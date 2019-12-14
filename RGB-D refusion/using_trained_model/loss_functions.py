import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp import inverse_warp, flow_warp, pose2flow
from ssim import ssim
epsilon = 1e-8

def smooth_loss(pred_mask):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_mask) not in [tuple, list]:
        pred_mask = [pred_mask]

    loss = 0
    weight = 1.

    for scaled_mask in pred_mask:
        dx, dy = gradient(scaled_mask)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # 2sqrt(2)
    return loss

def explainability_loss(mask):
    '''
    cross entropy with a ones_like matrix
    '''
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def logical_or(a, b):
    return 1 - (1 - a)*(1 - b)


def depth_residual_mask(valid_pixle_mask, explainability_mask, tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, pose, rotation_mode='euler', padding_mode='zeros', qch=0.5, wssim=0.5):
    def one_scale(depth):

        # assert(len(pose) == len(ref_imgs))
        

        mask_list = []
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
        # print(tgt_img_scaled.size()) [4, 3, 384, 512])


        for i, ref_img in enumerate(ref_imgs_scaled): # i=0, i=1

            # current_pose = pose[i]
            # print(current_pose.size())
            current_pose = pose[:, i]
            # print(current_pose.size())
            
            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            diff = (tgt_img_scaled - ref_img_warped).abs()
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped)

            # mean_reconstruction_loss  = ((1- wssim)*robust_l1_per_pix(diff, q=qch) + wssim*ssim_loss) .mean(1).mean(1).mean(1) # [B]
            # Res_mask = torch.where(diff.mean(1).unsqueeze(1)>(mean_reconstruction_loss.unsqueeze(1).unsqueeze(1).unsqueeze(1)), torch.ones_like(diff.mean(1).unsqueeze(1)),torch.zeros_like(diff.mean(1).unsqueeze(1)))
            # print(diff.mean(1).size(),mean_reconstruction_loss.unsqueeze(1).unsqueeze(1).size())
            # mask_list.append(Res_mask)
            with torch.no_grad():

                threshold =( valid_pixle_mask * (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) ) / ( valid_pixle_mask.sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) + 1 )
                #threshold_max =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* final_mask).sum(1).unsqueeze(1)).max(2)[0].unsqueeze(2).max(3)[0].unsqueeze(3)
                treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss).mean(1).unsqueeze(1)
                #treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss).sum(1).unsqueeze(1)
                threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
                #threshold_mask = 1-treshould_matrix/threshold_max
                # final_mask = threshold_mask * final_mask
                mask_list.append(torch.where(threshold_mask + (1-valid_pixle_mask)>0.5, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix)))
                
        return mask_list, ref_img_warped, diff

    if type(depth) not in [list, tuple]:
        depth = [depth]

    
    for d  in (depth):
        mask_list = one_scale(d)
    return mask_list

def flow_loss(tgt_img, ref_imgs, flows, explainability_mask, lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(explainability_mask, occ_masks, flows):
        reconstruction_loss = 0
        b, _, h, w = flows[0].size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]

            ref_img_warped = flow_warp(ref_img, current_flow)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * explainability_mask[:,i:i+1].expand_as(ssim_loss)

            if occ_masks is not None:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss += (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            #weight /= 2.83
            assert((reconstruction_loss == reconstruction_loss).item() == 1)
        
        return reconstruction_loss

    if type(flows[0]) not in [tuple, list]:
        if explainability_mask is not None:
            explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    for i in range(len(flows[0])):
        flow_at_scale = [uv[i] for uv in flows]
        occ_mask_at_scale_bw, occ_mask_at_scale_fw  = occlusion_masks(flow_at_scale[0], flow_at_scale[1])
        occ_mask_at_scale = torch.stack((occ_mask_at_scale_bw, occ_mask_at_scale_fw), dim=1)
        # occ_mask_at_scale = None
        loss += one_scale(explainability_mask[i], occ_mask_at_scale, flow_at_scale)
    ref_img_warped = flow_warp(ref_imgs[0], flows[0][0])
    diff = (tgt_img - ref_img_warped)
    return loss, ref_img_warped, diff

'''
def consensus_loss(explainability_mask, Res_mask):
    loss = 0
    # print(type(explainability_mask))
    for i, mask in enumerate(Res_mask):
        # print(mask.size(), explainability_mask[:,i,:,:].size())
        loss += nn.functional.binary_cross_entropy(explainability_mask[:,i,:,:], mask)
    return loss
'''

def consensus_loss(explainability_mask, census_mask_bwd, census_mask_fwd, THRESH, wbce):
    # Loop over each scale

    loss = 0.

    exp_mask_one_scale = explainability_mask
    census_mask_fwd_one_scale = (census_mask_fwd < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)
    census_mask_bwd_one_scale = (census_mask_bwd < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)

    #Using the pixelwise consensus term
    census_mask_fwd_one_scale = census_mask_fwd_one_scale
    census_mask_bwd_one_scale = census_mask_bwd_one_scale

    # OR gate for constraining only rigid pixels
    # exp_mask_fwd_one_scale = (exp_mask_one_scale[:,2].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
    # exp_mask_bwd_one_scale = (exp_mask_one_scale[:,1].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
    # census_mask_fwd_one_scale = 1- (1-census_mask_fwd_one_scale)*(1-exp_mask_fwd_one_scale)
    # census_mask_bwd_one_scale = 1- (1-census_mask_bwd_one_scale)*(1-exp_mask_bwd_one_scale)

    census_mask_fwd_one_scale = Variable(census_mask_fwd_one_scale.data, requires_grad=False)
    census_mask_bwd_one_scale = Variable(census_mask_bwd_one_scale.data, requires_grad=False)

    rigidity_mask_combined = torch.cat((census_mask_bwd_one_scale, census_mask_fwd_one_scale), dim=1)
    loss += weighted_binary_cross_entropy(exp_mask_one_scale, rigidity_mask_combined.type_as(exp_mask_one_scale), [wbce, 1-wbce])

    return loss

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output + epsilon)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + epsilon))
    else:
        loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

    return torch.neg(torch.mean(loss))

def depth_occlusion_masks(depth, pose, intrinsics, intrinsics_inv):
    flow_cam = [pose2flow(depth.squeeze(), pose[:,i], intrinsics, intrinsics_inv) for i in range(pose.size(1))]
    masks1, masks2 = occlusion_masks(flow_cam[1], flow_cam[2])
    masks0, masks3 = occlusion_masks(flow_cam[0], flow_cam[3])
    masks = torch.stack((masks0, masks1, masks2, masks3), dim=1)
    return masks

def occlusion_masks(flow_bw, flow_fw):
    mag_sq = flow_fw.pow(2).sum(dim=1) + flow_bw.pow(2).sum(dim=1)
    #flow_bw_warped = flow_warp(flow_bw, flow_fw)
    #flow_fw_warped = flow_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw
    flow_diff_bw = flow_bw + flow_fw
    occ_thresh =  0.08 * mag_sq + 1.0
    occ_fw = flow_diff_fw.sum(dim=1) > occ_thresh
    occ_bw = flow_diff_bw.sum(dim=1) > occ_thresh
    return occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)

def robust_l1(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    x = x.mean()
    return x

def robust_l1_per_pix(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    return x
