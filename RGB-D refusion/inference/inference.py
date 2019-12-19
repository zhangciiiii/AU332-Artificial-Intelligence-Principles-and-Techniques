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
import PoseNetB6
from sequence_folders_for_inference import SequenceFolder
from utils import tensor2array, save_checkpoint
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
  
import cv2

from path import Path
from tensorboardX import SummaryWriter
from itertools import chain
from tqdm import tqdm

parser = argparse.ArgumentParser(description='DeepFusion',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to pre-processed dataset')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 6')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default=None, metavar='PATH', help='path to pre-trained masknet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained posenet model')

best_error = -1
n_iter = 0


def main():
    global args
    args = parser.parse_args()
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path 
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()


    
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
          # custom_transforms.RandomRotate(),
          # custom_transforms.RandomHorizontalFlip(),
          # custom_transforms.RandomScaleCrop(),
          custom_transforms.ArrayToTensor(),
          normalize  ])

    training_writer = SummaryWriter(args.save_path)

    intrinsics = np.array([542.822841, 0, 315.593520, 0, 542.576870, 237.756098, 0, 0, 1]).astype(np.float32).reshape((3, 3))
    
    inference_set = SequenceFolder(
        root = args.dataset_dir,
        intrinsics = intrinsics,
        transform=train_transform,
        train=False,
        sequence_length=args.sequence_length
    )

    print('{} samples found in {} train scenes'.format(len(inference_set), len(inference_set.scenes)))
    inference_loader = torch.utils.data.DataLoader(
        inference_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("=> creating model")
    mask_net = MaskResNet6.MaskResNet6().cuda()
    pose_net = PoseNetB6.PoseNetB6().cuda()
    mask_net = torch.nn.DataParallel(mask_net)

    masknet_weights = torch.load(args.pretrained_mask)# 
    posenet_weights = torch.load(args.pretrained_pose)
    mask_net.load_state_dict(masknet_weights['state_dict'])
    # pose_net.load_state_dict(posenet_weights['state_dict'])
    pose_net.eval()
    mask_net.eval()

    # training 

    for i, (rgb_tgt_img, rgb_ref_imgs, intrinsics, intrinsics_inv) in enumerate(tqdm(inference_loader)):
        #print(rgb_tgt_img)
        tgt_img_var = Variable(rgb_tgt_img.cuda(), volatile=True)
        ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in rgb_ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)

        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)
        
        after_mask = tensor2array(ref_imgs_var[0][0]*explainability_mask[0,0]).transpose(1,2,0)
        x = Image.fromarray(np.uint8(after_mask*255))
        x.save(args.save_path/str(i).zfill(3)+'multi.png')
        
        explainability_mask = (explainability_mask[0,0].detach().cpu()).numpy()
        # print(explainability_mask.shape)
        y = Image.fromarray(np.uint8(explainability_mask*255))
        y.save(args.save_path/str(i).zfill(3)+'mask.png')

        


if __name__ == '__main__':

    main()
