##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

cm1 = plt.cm.cividis
cm2 = plt.cm.viridis

def colored_depthmap(depth, d_min=None, d_max=None, cmap=plt.cm.jet):
    """
    Parameters
    ----------
    depth : numpy.ndarray 
            shape [batch_size, h, w] or [h, w]
    """
    if len(depth.shape) == 2:
        depth = np.expand_dims(depth, 0)
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth = (depth - d_min) / (d_max - d_min)
    b, h, w = depth.shape
    depth_color = np.zeros((b, h, w, 3))
    for d in range(depth_color.shape[0]):
        depth_color[d] = cmap(depth[d])[:, :, :3]
    return np.asarray(255 * depth_color, dtype=np.uint8)

def merge_images(rgb, depth_target, depth_pred, orientation='row'):
    """
    Parameters
    ----------
    rgb, depth_target, depth_pred : numpy.ndarray
                                      shape [batch_size, h, w, c]
    
    Return
    ------
    img_merge : numpy.ndarray
                shape [batch_size, h*3, w, c] or [batch_size, h, w*3, c]
    """
    axis = 2 if orientation == 'row' else 1
    img_merge = np.concatenate([rgb, depth_target, depth_pred], axis)
    return img_merge

def make_grid(images, nrow=4):
    """
    Parameters
    ----------
    images : numpy.ndarray
            shape [batch_size, h, w, c]
    nrow : int 
           Number of images displayed in each row of the grid.
           The Final grid size is (batch_size / nrow, nrow).
    
    Return
    ------
    images : numpy.ndarray
             shape [batch_size/nrow * h, nrow * w, c]
    """
    b, h, w, c = images.shape
    ncol = b // nrow
    assert b // nrow * nrow == b, "batch size of images can not be exactly divided by nrow"
    images = images.reshape(ncol, nrow * h, w, c)
    images = images.transpose(1, 0, 2, 3)
    images = images.reshape(nrow * h, ncol * w, c)
    return images
    
def imshow_rgb(images, nrow, ncol):
    """
    Parameters
    ----------
    images : numpy.ndarray 
             shape [h, w, c]
    """
    h, w, c = images.shape
    #fig = plt.figure(figsize=(w // 80 * ncol, h // 50 * nrow))
    fig = plt.figure()
    plt.imshow(images)
    return fig

def display_figure(writer, visuals, epoch):
    #['inputs', 'sim_map', 'depths', 'labels']
    m = min(next(iter(visuals.values())).shape[0], 4)
    images_npy = visuals['inputs'][:m].cpu().numpy().transpose(0, 2, 3, 1)
    depths_npy = visuals['depths'][:m].cpu().numpy().transpose(0, 2, 3, 1)
    labels_npy = visuals['labels'][:m].cpu().numpy().transpose(0, 2, 3, 1)
    b, h, w, c = images_npy.shape
    # display image gt and pred figure
    #images = F.interpolate(images, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
    #labels = F.interpolate(labels, size=(h // 2, w // 2), mode='nearest', align_corners=True)
    #depths = F.interpolate(depths, size=(h // 2, w // 2), mode='nearest', align_corners=True)
    images_colored = np.asarray(255 * images_npy, dtype=np.uint8)
    labels_colored = colored_depthmap(labels_npy.squeeze(), cmap=cm1)
    depths_colored = colored_depthmap(depths_npy.squeeze(), cmap=cm1)
    fuse = merge_images(images_colored, labels_colored, depths_colored, 'col')
    fuse = make_grid(fuse, nrow=1)
    fig1 = imshow_rgb(fuse, nrow=1, ncol=b)
    writer.add_figure('figure1-images', fig1, epoch)
    del fuse, images_npy, labels_npy, depths_npy
    # display similarity figure
    if 'sim_map' in visuals.keys():
        sim_map = visuals['sim_map'].cpu()
        sim_map = sim_map[:b]
        N = sim_map.shape[1]
        points = [N // 4, N // 2, 3 * N // 4]
        sim_pixels = sim_map[:, points]
        sim_pixels = sim_pixels.reshape((b, len(points), h//8, w//8))
        sim_pixels = sim_pixels.permute((1, 0, 2, 3)).reshape((-1, 1, h//8, w//8))
        sim_pixels = F.interpolate(sim_pixels, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        sim_pixels = sim_pixels.cpu().numpy().transpose(0, 2, 3, 1)
        sim_pixels = colored_depthmap(sim_pixels.squeeze(), cmap=cm2)
        sim_pixels = make_grid(sim_pixels, nrow=len(points))
        fig2 = imshow_rgb(sim_pixels, nrow=len(points), ncol=b)
        writer.add_figure('figure3-pixel_attentions', fig2, epoch)
        del sim_map, sim_pixels
    return