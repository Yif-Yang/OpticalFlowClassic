
import imageio
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt

import flowfilter.plot as fplot
import flowfilter.gpu.flowfilters as gpufilter
import torch
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output
# paths to image and ground truth data
basepath = '/my_dir/Data/cloud_gaming/training_3w_0316/training_gs_3w_270p/gs270p1/'
rgb_path = '/my_dir/Data/cloud_gaming/training_3w_0316/training_frames_3w/video1/'
rgb_path = '/my_dir/Data/cloud_gaming/training_3w_0316/training_frames_3w/video1/'

# GPU filter object with 2 pyramid levels
gpuF = gpufilter.PyramidalFlowFilter(270, 480, 2)
gpuF.gamma = [10, 50]                                   # gains for each level
gpuF.maxflow = 4.0                                      # maximum optical flow value
gpuF.smoothIterations = [2, 4]                          # smooth iterations per level

# print('maxflow: {0}'.format(gpuF.maxflow))
offset = 101
K = 1000
log_dir = '/home/FAREAST.yifanyang/container_yifan/output/cloud_gaming/vis_data'
def img_reader(img, size):
    img = cv2.imread(img)
    img = cv2.resize(img, size)
    return img
def compare(gt_img, pred_img, img_idx):
    gt_img = gt_img * 255.0
    gt_img = gt_img.astype(np.uint8)
    pred_img = pred_img * 255.0
    pred_img = pred_img.astype(np.uint8)
    ssim_now = compare_ssim(gt_img, pred_img, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
    psnr_now = compare_psnr(gt_img, pred_img, data_range=255)

    fname = '%s/gt_%d.png' % (log_dir, img_idx)
    cv2.imwrite(fname, gt_img)
    fname = '%s/pred_%d.png' % (log_dir, img_idx)
    cv2.imwrite(fname, pred_img)
    return ssim_now, psnr_now
avgET = np.zeros(K)
for k in range(offset, offset + K):

    ##########################################
    # COMPUTATION
    ##########################################

    # read and load next image to the filter
    # img = imageio.imread( basepath + f'{k}.png').astype(np.uint8)
    img = cv2.imread( basepath + f'{k}.png')
    rgb_img_now = img_reader(rgb_path + f'{k:05d}.png', [480, 270])
    if k > offset:
        rgb_img_prev = img_reader(rgb_path + f'{k - 1:05d}.png', [480, 270])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gpuF.loadImage(img)
    # compute new estimate of optical flow
    gpuF.compute()

    # return a Numpy ndarray with the new optical flow estimation
    flow = gpuF.getFlow()

    if k > offset:
        flow = torch.from_numpy(flow).unsqueeze(0).permute(0, 3, 1, 2)
        rgb_img_now = torch.from_numpy(rgb_img_now).unsqueeze(0).permute(0, 3, 1, 2) / 255
        rgb_img_prev = torch.from_numpy(rgb_img_prev).unsqueeze(0).permute(0, 3, 1, 2) / 255
        warp_out = flow_warp(rgb_img_prev, flow.permute(0, 2, 3, 1))
        warp_out_np = warp_out[0].data.cpu().numpy().transpose(1,2,0)
        rgb_img_now_np = rgb_img_now[0].data.cpu().numpy().transpose(1,2,0)
        ssim, psnr = compare(rgb_img_now_np, warp_out_np, k)
        print(ssim, psnr)
    # print(k, flow)

    # runtime in milliseconds
    avgET[k - offset] = gpuF.elapsedTime()


print('average elapsed time: {0} ms'.format(np.average(avgET)))