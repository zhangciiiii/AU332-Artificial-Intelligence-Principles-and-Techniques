# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import os
from tqdm import tqdm
import numpy as np
from path import Path
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

import custom_transforms
from inverse_warp import pose2flow
from datasets.validation_flow_video import ValidationFlow, ValidationFlowKitti2012
import models
from logger import AverageMeter
from PIL import Image
from torchvision.transforms import ToPILImage
from flowutils.flowlib import flow_to_image
from utils import tensor2array
from loss_functions import compute_all_epes, consensus_exp_masks, logical_or
import scipy.misc
import matplotlib.pyplot as plt

CMAP = 'plasma'
UNKNOWN_FLOW_THRESH = 1e7


parser = argparse.ArgumentParser(description='Evaluate optical flow on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/ps/project/datasets/AllFlowData/kitti/kitti2015',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispResNet6', 'DispNetS6'],
                    help='depth network architecture.')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseNetB6', choices=['PoseNet6','PoseNetB6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--masknet', dest='masknet', type=str, default='MaskNet6', choices=['MaskResNet6', 'MaskNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['Back2Future', 'FlowNetC6'],
                    help='flow network architecture.')

parser.add_argument('--THRESH', dest='THRESH', type=float, default=0.01, help='THRESH')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained posenet model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH', help='path to pre-trained flownet model')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default=None, metavar='PATH', help='path to pre-trained masknet model')

parser.add_argument('--nlevels', dest='nlevels', type=int, default=6, help='number of levels in multiscale. Options: 4|5')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', help='path to pre-trained Flow net model')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='path to output directory')


def main():
    global args
    args = parser.parse_args()
    args.pretrained_disp = Path(args.pretrained_disp)
    args.pretrained_pose = Path(args.pretrained_pose)
    args.pretrained_mask = Path(args.pretrained_mask)
    args.pretrained_flow = Path(args.pretrained_flow)

    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
        args.output_dir.makedirs_p()

        image_dir = args.output_dir/'images'
        gt_dir = args.output_dir/'gt'
        mask_dir = args.output_dir/'mask'
        viz_dir = args.output_dir/'viz'

        image_dir.makedirs_p()
        gt_dir.makedirs_p()
        mask_dir.makedirs_p()
        viz_dir.makedirs_p()

        output_writer = SummaryWriter(args.output_dir)

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    flow_loader_h, flow_loader_w = 256, 832
    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])
    if args.dataset == "kitti2015":
        N=len(os.listdir(args.kitti_dir))-4
        val_flow_set = ValidationFlow(root=args.kitti_dir, N=N,
                                sequence_length=5, transform=valid_flow_transform)


    val_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)

    disp_net = getattr(models, args.dispnet)().cuda()
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=4).cuda()
    mask_net = getattr(models, args.masknet)(nb_ref_imgs=4).cuda()
    flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()

    dispnet_weights = torch.load(args.pretrained_disp)
    posenet_weights = torch.load(args.pretrained_pose)
    masknet_weights = torch.load(args.pretrained_mask)
    flownet_weights = torch.load(args.pretrained_flow)
    disp_net.load_state_dict(dispnet_weights['state_dict'])
    pose_net.load_state_dict(posenet_weights['state_dict'])
    flow_net.load_state_dict(flownet_weights['state_dict'])
    mask_net.load_state_dict(masknet_weights['state_dict'])

    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    error_names = ['epe_total', 'epe_sp', 'epe_mv', 'Fl', 'epe_total_gt_mask', 'epe_sp_gt_mask', 'epe_mv_gt_mask', 'Fl_gt_mask']
    errors = AverageMeter(i=len(error_names))
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            tgt_img_var = Variable(tgt_img.cuda())
            ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
            intrinsics_var = Variable(intrinsics.cuda())
            intrinsics_inv_var = Variable(intrinsics_inv.cuda())

            flow_gt_var = Variable(flow_gt.cuda())
            obj_map_gt_var = Variable(obj_map_gt.cuda())

            disp = disp_net(tgt_img_var)
            depth = 1/disp
            pose = pose_net(tgt_img_var, ref_imgs_var)
            explainability_mask = mask_net(tgt_img_var, ref_imgs_var)


        if args.flownet=='Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])
        else:
            flow_fwd = flow_net(tgt_img_var, ref_imgs_var[2])

        flow_cam = pose2flow(depth.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var)
        flow_cam_bwd = pose2flow(depth.squeeze(1), pose[:,1], intrinsics_var, intrinsics_inv_var)

        rigidity_mask = 1 - (1-explainability_mask[:,1])*(1-explainability_mask[:,2]).unsqueeze(1) > 0.5
        rigidity_mask_census_soft = (flow_cam - flow_fwd).abs()#.normalize()
        rigidity_mask_census_u = rigidity_mask_census_soft[:,0] < args.THRESH
        rigidity_mask_census_v = rigidity_mask_census_soft[:,1] < args.THRESH
        rigidity_mask_census = (rigidity_mask_census_u).type_as(flow_fwd) * (rigidity_mask_census_v).type_as(flow_fwd)

        rigidity_mask_combined = 1 - (1-rigidity_mask.type_as(explainability_mask))*(1-rigidity_mask_census.type_as(explainability_mask))

        obj_map_gt_var_expanded = obj_map_gt_var.unsqueeze(1).type_as(flow_fwd)

        flow_fwd_non_rigid = (rigidity_mask_combined<=args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_fwd
        flow_fwd_rigid = (rigidity_mask_combined>args.THRESH).type_as(flow_cam).expand_as(flow_cam) * flow_cam
        total_flow = flow_fwd_rigid + flow_fwd_non_rigid

        rigidity_mask = rigidity_mask.type_as(flow_fwd)
        _epe_errors = compute_all_epes(flow_gt_var, flow_cam, flow_fwd, rigidity_mask_combined) + compute_all_epes(flow_gt_var, flow_cam, flow_fwd, (1-obj_map_gt_var_expanded) )
        errors.update(_epe_errors)

        tgt_img_np = tgt_img[0].numpy()
        rigidity_mask_combined_np = rigidity_mask_combined.cpu().data[0].numpy()
        gt_mask_np = obj_map_gt[0].numpy()


        if args.output_dir is not None:
            np.save(image_dir/str(i).zfill(3), tgt_img_np )
            np.save(gt_dir/str(i).zfill(3), gt_mask_np)
            np.save(mask_dir/str(i).zfill(3), rigidity_mask_combined_np)



        if (args.output_dir is not None):
            """
            ind = int(i//10)
            output_writer.add_image('val Dispnet Output Normalized', tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone'), ind)
            output_writer.add_image('val Input', tensor2array(tgt_img[0].cpu()), i)
            output_writer.add_image('val Total Flow Output', flow_to_image(tensor2array(total_flow.data[0].cpu())), ind)
            output_writer.add_image('val Rigid Flow Output', flow_to_image(tensor2array(flow_fwd_rigid.data[0].cpu())), ind)
            output_writer.add_image('val Non-rigid Flow Output', flow_to_image(tensor2array(flow_fwd_non_rigid.data[0].cpu())), ind)
            output_writer.add_image('val Rigidity Mask', tensor2array(rigidity_mask.data[0].cpu(), max_value=1, colormap='bone'), ind)
            output_writer.add_image('val Rigidity Mask Census', tensor2array(rigidity_mask_census.data[0].cpu(), max_value=1, colormap='bone'), ind)
            output_writer.add_image('val Rigidity Mask Combined', tensor2array(rigidity_mask_combined.data[0].cpu(), max_value=1, colormap='bone'), ind)
            """

            
            tgt_img_viz = tensor2array(tgt_img[0].cpu())
            depth_viz = tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone')
            # mask_viz = tensor2array(rigidity_mask_census_soft.data[0].prod(dim=0).cpu(), max_value=1, colormap='bone')
            mask_viz = tensor2array(rigidity_mask_combined.data[0].cpu(), max_value=1, colormap='bone')

            rigid_flow_viz = flow_to_image(tensor2array(flow_cam.data[0].cpu()))
            non_rigid_flow_viz = flow_to_image(tensor2array(flow_fwd_non_rigid.data[0].cpu()))
            total_flow_viz = flow_to_image(tensor2array(total_flow.data[0].cpu()))
            '''
            #row1_viz = np.hstack((tgt_img_viz, depth_viz, mask_viz))# hstack
            row1_viz = np.hstack((tgt_img_viz, tgt_img_viz, tgt_img_viz))
            row2_viz = np.hstack((rigid_flow_viz, non_rigid_flow_viz, total_flow_viz))

            row1_viz_im = Image.fromarray((255*row1_viz).astype('uint8'), mode='RGB')
            row2_viz_im = Image.fromarray((row2_viz).astype('uint8'), mode='RGB') 
            '''
            # print(disp[0].permute(1,2,0).size())

            # row1_viz = np.hstack((tgt_img_viz, depth_viz, mask_viz))# hstack
            row2_viz = np.hstack((rigid_flow_viz, non_rigid_flow_viz, total_flow_viz))
            # row2_viz_im = Image.fromarray((total_flow_viz).astype('uint8'),mode='P') 
            
            depth_map = disp[0].permute(1,2,0).cpu().numpy()
            depth_map = np.squeeze(depth_map)
            colored_map = _normalize_depth_for_display(depth_map, cmap=CMAP)

            tmp1 = total_flow.data[0].permute(1,2,0).cpu().numpy()
            tmp2 = tgt_img[0].permute(1,2,0).cpu().numpy()

            tmp1 = flow_2_image(tmp1)
            tmp2 = tmp2 * 255.0
            # print(tmp1)
            # tmp2 = flow_2_image(tmp2)

            result = np.vstack((tmp1,tmp2))

            scipy.misc.imsave(viz_dir/str(i).zfill(4)+'flow.png', result)

        

    print("Results")
    print("\t {:>10}, {:>10}, {:>10}, {:>6}, {:>10}, {:>10}, {:>10}, {:>10} ".format(*error_names))
    print("Errors \t {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors.avg))

def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / F_mag) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    f_err = n_err.sum()/valid_gt.sum();
    if type(f_err) == Variable: f_err = f_err.data
    return f_err[0]

def _gray2rgb(im, cmap=CMAP):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img


def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap=CMAP):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = _gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp

def flow_2_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))



    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return img

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(
        np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(
        np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(
        np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(
        np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


if __name__ == '__main__':
    main()

