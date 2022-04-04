import os, sys
import math, time, random

import numpy as np

import imageio
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from utils.image import to8b, img2mse, mse2psnr
from utils.ray import get_ortho_rays
import cv2
from pdb import set_trace as st

def eval_one_view(model, batch, near_far, device, stl_idx=None, bs=2, filter=False, **render_kwargs):
    '''Model inference
    '''
    model.eval()
    near, far = near_far
    with torch.no_grad():
        batch_rays = batch['rays'].cuda()
        # Run nerf
        # batch size = 1 in testing
        img_h, img_w = batch['rays'].shape[1:3]
        batch_rays = batch_rays.permute(1, 2, 0, 3)
        batch_rays = batch_rays.reshape([batch_rays.shape[0]*batch_rays.shape[1], 2, 3])
        batch_rays = torch.chunk(batch_rays, bs, dim=0)
        batch_rays = torch.stack(batch_rays)
        batch_rays = batch_rays.permute(0, 2, 1, 3)
        ret_dict = model(batch_rays, (near, far), stl_idx=stl_idx, test=True, **render_kwargs)
        for k, v in ret_dict.items():
            ret_dict[k] = v.cpu()

        ret_dict['rgb'] = ret_dict['rgb'].reshape([img_h, img_w, 3])
        ret_dict['disp'] = ret_dict['disp'].reshape([img_h, img_w, 1])
        ret_dict['acc'] = ret_dict['acc'].reshape([img_h, img_w, 1])

        metric_dict = {}
        if 'target_s' in batch:
            target_s = batch['target_s']
            ret_dict['target_s'] = target_s

            mse = img2mse(ret_dict['rgb'], target_s)
            metric_dict['mse'] = mse
            metric_dict['psnr'] = mse2psnr(mse)

        return ret_dict, metric_dict


def get_idx():
    '''smooth style embedding input
    '''
    stl_list = []
    for i in range(10000, -1, -150):
        i /= 10000
        j = 1 - i
        stl_idx = torch.Tensor([i, j, 0]).cuda()
        stl_list.append(stl_idx)

    for i in range(10000, -1, -150):
        i /= 10000
        j = 1 - i
        stl_idx = torch.Tensor([0, i, j]).cuda()
        stl_list.append(stl_idx)

    for i in range(10000, -1, -150):
        i /= 10000
        j = 1 - i
        stl_idx = torch.Tensor([j, 0, i]).cuda()
        stl_list.append(stl_idx)
    return stl_list


def linear_eval(model, dataset, save_dir=None,  expname=None, stl_idx=None, bs=1, device=None):
    '''Main function of conditional style interpolation
    '''
    near, far = dataset.near_far()
    rgbs, disps = [], []
    stl_idx_list = get_idx()
    for i, batch in enumerate(tqdm(dataset, desc='Rendering')):
        # if i >= 30:
        #     continue
        stl_idx = stl_idx_list[i]
        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), stl_idx=stl_idx, device=device, bs=bs)
        img, disp = ret_dict['rgb'].numpy(), ret_dict['disp'].numpy()
        rgbs.append(to8b(img))
        disps.append(to8b(disp / np.max(disp)))
    size = rgbs[-1].shape[:2]

    out = cv2.VideoWriter(os.path.join(save_dir, f"rgb_{expname}_linear_eval.mp4"), cv2.VideoWriter_fourcc('M','P','4','V'), 30, (400, 400), True)
    for i in range(len(rgbs)):
        rgb_img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
        out.write(cv2.resize(rgb_img, (400, 400)))
    out.release()


def evaluate(model_and_transformNet, dataset, device, save_dir=None, stl_idx=None, slice=-1, bs=1, **render_kwargs):
    '''Main function of  evaluation
    '''
    if isinstance(model_and_transformNet, list):
        model, transformer = model_and_transformNet
    else:
        model = model_and_transformNet
    near, far = dataset.near_far()
    total_mse = 0.
    for i, batch in enumerate(dataset):
        # if i == 1:
        #     return

        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), stl_idx=stl_idx, device=device, bs=bs, **render_kwargs)
        
        if ("mse" not in metric_dict.keys()):
            metric_dict["mse"] = torch.Tensor([0])
        if ("psnr" not in metric_dict.keys()):
            metric_dict["psnr"] = torch.Tensor([0])

        img, disp, acc = ret_dict['rgb'].numpy(), ret_dict['disp'].numpy(), ret_dict['acc'].numpy()
        print(f"[TEST] Iter {i+1}/{len(dataset)} MSE: {metric_dict['mse'].item()} PSNR: {metric_dict['psnr'].item()}")
        # accumulate mse
        total_mse += metric_dict['mse']

        if save_dir is not None:
            if stl_idx is not None:
                idx = stl_idx.cpu().numpy().tolist()
                idx = str(idx)
            else:
                idx = ""
            imageio.imwrite(os.path.join(save_dir, f'rgb_{i:03d}_{idx}.png'), to8b(img))
            imageio.imwrite(os.path.join(save_dir, f'disp_{i:03d}_{idx}.png'), to8b(disp / np.max(disp)))
            imageio.imwrite(os.path.join(save_dir, f'acc_{i:03d}_{idx}.png'), to8b(acc / np.max(acc)))

    total_mse = total_mse / len(dataset)
    total_psnr = mse2psnr(total_mse)
    print(f'[TEST] MSE: {total_mse.item()} PSNR: {total_psnr.item()}')
    return {'mse': total_mse.item(), 'psnr': total_psnr.item()}


def render_video(model, dataset, device, save_dir, suffix='', fps=30, quality=8, expname='', stl_idx=None, bs=2,  **render_kwargs):
    '''Render video
    '''
    near, far = dataset.near_far()
    rgbs, disps = [], []
    for i, batch in enumerate(tqdm(dataset, desc='Rendering')):
        # if i >= 1:
        #     continue
        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), stl_idx=stl_idx, device=device, bs=bs, filter=False, **render_kwargs)
        img, disp = ret_dict['rgb'].numpy(), ret_dict['disp'].numpy()
        rgbs.append(to8b(img))
        disps.append(to8b(disp / np.max(disp)))
    size = rgbs[-1].shape[:2]

    if stl_idx is not None:
        idx = stl_idx.cpu().numpy().tolist()
        idx = str(idx)
    else:
        idx = None

    out = cv2.VideoWriter(os.path.join(save_dir, f"rgb_{suffix}_{expname}_stl{idx}.mp4"), cv2.VideoWriter_fourcc('M','P','4','V'), fps, (400, 400), True)
    for i in range(len(rgbs)):
        rgb_img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
        out.write(cv2.resize(rgb_img, (400, 400)))
    out.release()