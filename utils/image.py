
import os, sys
import math, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio
from pdb import set_trace as st

# Misc
# img2mae = lambda x, y : torch.mean(torch.abs(x - y))
# img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

def contrast_loss(features, indexes):
    loss = torch.Tensor([0]).to(indexes.device)
    feat_dict = dict()
    for idx, i in enumerate(indexes):
        feat_dict[i.item()] = []
    # for idx, i in enumerate(indexes):
    #     feat_dict[i.item()].append(x[idx] for x in features])
    # pass


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x*mask - y*mask) ** 2) / (torch.sum(mask) + 1e-5)

def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x*mask - y*mask)) / (torch.sum(mask) + 1e-5)

def to8b(x):
    # return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)
    return (255*x).astype(np.uint8)

def export_images(rgbs, save_dir, H=0, W=0):
    rgb8s = []
    for i, rgb in enumerate(rgbs):
        # Resize
        if H > 0 and W > 0:
            rgb = rgb.reshape([H, W])

        filename = os.path.join(save_dir, '{:03d}.npy'.format(i))
        np.save(filename, rgb)
        
        # Convert to image
        rgb8 = to8b(rgb)
        filename = os.path.join(save_dir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)
        rgb8s.append(rgb8)
    
    return np.stack(rgb8s, 0)

def export_video(rgbs, save_path, fps=30, quality=8):
    imageio.mimwrite(save_path, to8b(rgbs), fps=fps, quality=quality)