import argparse
import time
import csv
import datetime
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data

import custom_transforms
import MaskResNet6
import MaskNet6
import PoseNetB6
import PoseNet6
import PoseExpNet
import back2future
from sequence_folders import SequenceFolder, ValSequenceFolder
from utils import tensor2array, save_checkpoint 
from loss_functions import smooth_loss, explainability_loss, depth_residual_mask, consensus_loss, flow_loss
from inverse_warp import inverse_warp, pose2flow, flow2oob, flow_warp
from flowutils.flowlib import flow_to_image

from path import Path
from tensorboardX import SummaryWriter
from itertools import chain
from tqdm import tqdm

parser = argparse.ArgumentParser(description='DeepFusion',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to pre-processed dataset')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default=None, metavar='PATH',
                    help='path to pre-trained Exp mask net model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained Exp flow net model')

parser.add_argument('--fix-masknet', dest='fix_masknet', action='store_true', help='do not train posenet')
parser.add_argument('--fix-posenet', dest='fix_posenet', action='store_true', help='do not train posenet')
parser.add_argument('--fix-flownet', dest='fix_flownet', action='store_true', help='do not train flownet')

parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 6')
parser.add_argument('--robust', dest='robust', action='store_true', help='train using robust losses')
parser.add_argument('-wrig', '--wrig', type=float, help='consensus imbalance weight', metavar='W', default=1.0)
parser.add_argument('-wbce', '--wbce', type=float, help='weight for binary cross entropy loss', metavar='W', default=0.5)
parser.add_argument('-wssim', '--wssim', type=float, help='weight for ssim loss', metavar='W', default=0.0)

parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--consensus-loss-weight', type=float, help='weight for mask consistancy', metavar='W', default=0.1)
parser.add_argument('-fl', '--flow-loss-weight', type=float, help='weight for flow consistancy', metavar='W', default=0.1)
parser.add_argument('--THRESH', '--THRESH', type=float, help='threshold for masks', metavar='W', default=0.01)


best_error = -1
n_iter = 0


def main():
    global args, best_error, n_iter
    args = parser.parse_args()
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path 
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
          custom_transforms.RandomRotate(),
          custom_transforms.RandomHorizontalFlip(),
          custom_transforms.RandomScaleCrop(),
          custom_transforms.ArrayToTensor(),
          normalize  ])
    training_writer = SummaryWriter(args.save_path)

    intrinsics = np.array([542.822841, 0, 315.593520, 0, 542.576870, 237.756098, 0, 0, 1]).astype(np.float32).reshape((3, 3))
    train_set = SequenceFolder(
        root = args.dataset_dir,
        intrinsics = intrinsics,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("=> creating model")
    mask_net = MaskNet6.MaskNet6().cuda()
    flow_net = back2future.Model(nlevels=args.nlevels).cuda()
    pose_net = PoseNetB6 .PoseNetB6 ().cuda()
    

    if args.pretrained_mask:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_mask)
        mask_net.load_state_dict(weights['state_dict'])
    else:
        mask_net.init_weights()
    mask_net = torch.nn.DataParallel(mask_net)

    if args.pretrained_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'])
    else:
        pose_net.init_weights()

    if args.pretrained_flow:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_flow)
        flow_net.load_state_dict(weights['state_dict'])
    else:
        flow_net.init_weights()

    print('=> setting adam solver')
    parameters = chain(mask_net.parameters(),pose_net.parameters(),flow_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    
    # training 
    best_error = 0
    train_loss = 0
    for epoch in tqdm(range(args.epochs)):
        if args.fix_flownet:
            for fparams in flow_net.parameters():
                fparams.requires_grad = False

        if args.fix_masknet:
            for fparams in mask_net.parameters():
                fparams.requires_grad = False

        if args.fix_posenet:
            for fparams in pose_net.parameters():
                fparams.requires_grad = False
        
        is_best = train_loss < best_error
        best_error = min(best_error, train_loss)
        save_checkpoint(
                args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': mask_net.module.state_dict()
                },  {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.state_dict()
                },{   
                    'epoch': epoch + 1,
                    'state_dict': flow_net.state_dict()
                },  {                
                    'epoch': epoch + 1,
                    'state_dict': optimizer.state_dict()
                },
                is_best)
        train_loss = train(train_loader, mask_net, pose_net, flow_net, optimizer, args.epoch_size, training_writer)



def train(train_loader, mask_net, pose_net, flow_net, optimizer, epoch_size, train_writer):
    global args, n_iter
    w1 = args.smooth_loss_weight
    w2 = args.mask_loss_weight
    w3 = args.consensus_loss_weight
    w4 = args.flow_loss_weight
    
    
    mask_net.train()
    pose_net.train()
    flow_net.train()
    average_loss = 0
    for i, (rgb_tgt_img, rgb_ref_imgs, depth_tgt_img, depth_ref_imgs, intrinsics, intrinsics_inv, pose_list) in enumerate(tqdm(train_loader)):
        rgb_tgt_img_var = Variable(rgb_tgt_img.cuda())
        # print(rgb_tgt_img_var.size())
        rgb_ref_imgs_var = [Variable(img.cuda()) for img in rgb_ref_imgs]
        # rgb_ref_imgs_var = [rgb_ref_imgs_var[0], rgb_ref_imgs_var[0], rgb_ref_imgs_var[1], rgb_ref_imgs_var[1]]
        depth_tgt_img_var = Variable(depth_tgt_img.unsqueeze(1).cuda())
        depth_ref_imgs_var = [Variable(img.unsqueeze(1).cuda()) for img in depth_ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda())
        intrinsics_inv_var = Variable(intrinsics_inv.cuda())
        # pose_list_var = [Variable(one_pose.float().cuda()) for one_pose in pose_list]

        explainability_mask = mask_net(rgb_tgt_img_var, rgb_ref_imgs_var)
        valid_pixle_mask = torch.where(depth_tgt_img_var==0, torch.zeros_like(depth_tgt_img_var), torch.ones_like(depth_tgt_img_var)) # zero is invalid
        # print(depth_test[0].sum())
        
        # print(explainability_mask[0].size()) #torch.Size([4, 2, 384, 512]) 
        # print()
        pose = pose_net(rgb_tgt_img_var, rgb_ref_imgs_var)

        # generate flow from camera pose and depth
        flow_fwd, flow_bwd, _ = flow_net(rgb_tgt_img_var, rgb_ref_imgs_var)
        flows_cam_fwd = pose2flow(depth_ref_imgs_var[1].squeeze(1), pose[:,1], intrinsics_var, intrinsics_inv_var) 
        flows_cam_bwd = pose2flow(depth_ref_imgs_var[0].squeeze(1), pose[:,0], intrinsics_var, intrinsics_inv_var) 
        rigidity_mask_fwd = (flows_cam_fwd - flow_fwd[0]).abs() 
        rigidity_mask_bwd = (flows_cam_bwd - flow_bwd[0]).abs()


        # loss 1: smoothness loss
        loss1 = smooth_loss(explainability_mask) + smooth_loss(flow_bwd) + smooth_loss(flow_fwd)

        # loss 2: explainability loss
        loss2 = explainability_loss(explainability_mask)

        # loss 3 consensus loss (the mask from networks and the mask from residual)
        depth_Res_mask, depth_ref_img_warped, depth_diff = depth_residual_mask(valid_pixle_mask, explainability_mask[0], rgb_tgt_img_var, rgb_ref_imgs_var, intrinsics_var, intrinsics_inv_var, depth_tgt_img_var, pose)              
        # print(depth_Res_mask[0].size(), explainability_mask[0].size())

        loss3  = consensus_loss(explainability_mask[0], rigidity_mask_bwd, rigidity_mask_fwd, args.THRESH, args.wbce) 

        # loss 4: flow loss
        loss4, flow_ref_img_warped, flow_diff = flow_loss(rgb_tgt_img_var, rgb_ref_imgs_var,[flow_bwd, flow_fwd], explainability_mask)

        # compute gradient and do Adam step
        loss = w1*loss1 + w2*loss2 + w3*loss3 + w4*loss4
        average_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visualization in tensorboard
        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('smoothness loss', loss1.item(), n_iter)
            train_writer.add_scalar('explainability loss', loss2.item(), n_iter)
            train_writer.add_scalar('consensus loss', loss3.item(), n_iter)
            train_writer.add_scalar('flow loss', loss4.item(), n_iter)
            train_writer.add_scalar('total loss', loss.item(), n_iter)
        if n_iter % (args.training_output_freq) == 0:
            train_writer.add_image('train Input', tensor2array(rgb_tgt_img_var[0]), n_iter)
            train_writer.add_image('train Exp mask Outputs ', 
                                    tensor2array(explainability_mask[0][0,0].data.cpu(), max_value=1, colormap='bone'), n_iter)
            train_writer.add_image('train depth Res mask ', 
                                    tensor2array(depth_Res_mask[0][0].data.cpu(), max_value=1, colormap='bone'), n_iter)
            train_writer.add_image('train depth ', 
                                    tensor2array(depth_tgt_img_var[0].data.cpu(), max_value=1, colormap='bone'), n_iter)
            train_writer.add_image('train valid pixel ', 
                                    tensor2array(valid_pixle_mask[0].data.cpu(), max_value=1, colormap='bone'), n_iter)
            train_writer.add_image('train after mask', tensor2array(rgb_tgt_img_var[0]*explainability_mask[0][0,0]), n_iter)
            train_writer.add_image('train depth diff', tensor2array(depth_diff[0]), n_iter)
            train_writer.add_image('train flow diff', tensor2array(flow_diff[0]), n_iter)
            train_writer.add_image('train depth warped img', tensor2array(depth_ref_img_warped[0]), n_iter)
            train_writer.add_image('train flow warped img', tensor2array(flow_ref_img_warped[0]), n_iter)
            train_writer.add_image('train Cam Flow Output',
                                    flow_to_image(tensor2array(flow_fwd[0].data[0].cpu())) , n_iter )
            train_writer.add_image('train Flow from Depth Output',
                                    flow_to_image(tensor2array(flows_cam_fwd.data[0].cpu())) , n_iter )
            train_writer.add_image('train Flow and Depth diff',
                                    flow_to_image(tensor2array(rigidity_mask_fwd.data[0].cpu())) , n_iter )

            


        n_iter += 1
    
    return average_loss/i

if __name__ == '__main__':

    main()
