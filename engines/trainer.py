import os, sys
import pdb
import math, time, random

import numpy as np

import imageio
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.image import to8b, img2mse, mse2psnr, img2mae, contrast_loss
from models.nerf_net import NeRFNet
from engines.eval import eval_one_view, evaluate, render_video
import utils.style_utils as style_utils
from pdb import set_trace as st
import cv2

def interpolate(batch, mode='bilinear', size=[224, 224]):
    x = F.interpolate(batch, size)
    return x

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def train_one_epoch(model_and_VGG_and_TransformNet, optimizer, scheduler, train_loader, test_set, exhibit_set, summary_writer, global_step, max_steps,
    run_dir, device, i_print=100, i_img=500, log_img_idx=0, i_weights=10000, i_testset=50000, i_video=50000, args=None):

    model, teacher, VGG, transformer = model_and_VGG_and_TransformNet
    near, far = train_loader.dataset.near_far()

    stl_num = train_loader.dataset.style_num

    start_step = global_step
    time0 = time.time()
    for (batch_rays, target_s, style_s, idx, mask, stl_idx) in train_loader:
        model.train()

        # counter accumulate
        global_step += 1

        if args.scale_ps_step != -1:
            if global_step % args.scale_ps_step == 0:
                ps = max(1, train_loader.dataset.ps // 2)
                print(f"[Info]: Set ps from {train_loader.dataset.ps} to {ps}")
                train_loader.dataset.ps = ps

        # dataset pre-processing
        batch_rays, target_s, style_s, mask, stl_idx = \
            batch_rays.cuda(), target_s.cuda(), style_s.cuda(), mask.cuda(), stl_idx.cuda()

        # nerf forward
        if stl_idx[0].item() != 999:
            _stl_idx = F.one_hot(stl_idx, num_classes=stl_num).float()
        else:
            _stl_idx = None
        # _input: 2, 12, 12, 3
        _input = batch_rays.permute(0, 3, 1, 2, 4)
        ret_dict = model(_input, (near, far), stl_idx=_stl_idx, test=False) # no extraction
        if teacher is not None and (not args.self_distilled):
            ret_dict_teach = teacher(_input, (near, far), test=False)
        optimizer.zero_grad()
        # pre-process for VGG
        rgb_pred0 = ret_dict['rgb0']
        rgb_pred = ret_dict['rgb']
        rgb_pred0, rgb_pred, target_s, style_s = \
            rgb_pred0 * 255, rgb_pred * 255, target_s * 255, style_s * 255

        # _rgb_pred0, _rgb_pred, _target_s, _style_s, _mask = \
        #     rgb_pred0.detach().cpu().numpy()[0, ...], rgb_pred.detach().cpu().numpy()[0, ...], target_s.detach().cpu().numpy()[0, ...], style_s.detach().cpu().numpy()[0, ...], (mask*255).detach().cpu().numpy()[0, ...]
        # cv2.imwrite("logs/vis_/_rgb_pred0.png", _rgb_pred0.astype(np.uint8))
        # cv2.imwrite("logs/vis_/_rgb_pred.png", _rgb_pred.astype(np.uint8))
        # cv2.imwrite("logs/vis_/_target_s.png", _target_s.astype(np.uint8))
        # cv2.imwrite("logs/vis_/_style_s.png", _style_s.astype(np.uint8))
        # cv2.imwrite("logs/vis_/_mask.png", _mask.astype(np.uint8))
        # st()

        rgb_pred0, rgb_pred, target_s, style_s, mask = \
            rgb_pred0.permute(0, 3, 1, 2), rgb_pred.permute(0, 3, 1, 2), target_s.permute(0, 3, 1, 2), style_s.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)

        rgb_pred0, rgb_pred, target_s, style_s, mask = \
            torch.clamp(rgb_pred0, 0, 255, out=None), torch.clamp(rgb_pred, 0, 255, out=None), torch.clamp(target_s, 0, 255, out=None), torch.clamp(style_s, 0, 255, out=None), torch.clamp(mask, 0, 1, out=None)

        ### VGG forward ###

        rgb_pred0, rgb_pred, target_s, style_s, mask = \
            torch.clamp(rgb_pred0, 0, 255, out=None), torch.clamp(rgb_pred, 0, 255, out=None), torch.clamp(target_s, 0, 255, out=None), torch.clamp(style_s, 0, 255, out=None), torch.clamp(mask, 0, 1, out=None)

        _rgb_pred = interpolate(rgb_pred)
        _target_s = interpolate(target_s)
        _style_s = interpolate(style_s)
        _mask = interpolate(mask)

        rgb_pred_features = VGG(normalize_batch(_rgb_pred) * _mask)
        rgb_gt_features = VGG(normalize_batch(_target_s) * _mask)
        style_features = VGG(normalize_batch(_style_s))

        ### compute loss ###
        loss = 0

        # content loss
        content_loss = img2mse(rgb_gt_features['relu2_2'], rgb_pred_features['relu2_2'])

        # style loss
        style_loss = 0.
        gram_style = []
        for k, y in style_features.items():
            gram_style.append(gram_matrix(y))

        gram_pred = []
        for k, y in rgb_pred_features.items():
            gram_pred.append(gram_matrix(y))

        for gm_y, gm_s in zip(gram_pred, gram_style):
            style_loss += img2mse(gm_y, gm_s)
        # image loss
        img_loss = img2mse(rgb_pred, target_s)
        psnr = mse2psnr(img_loss)
        view_id = list(idx.cpu().numpy())
        if 'rgb0' in ret_dict:
            img_loss0 = img2mse(rgb_pred0, target_s)
            psnr0 = mse2psnr(img_loss0)

        # accumulate loss
        assert not ("style_v_0" in args.loss_terms and "style_v_all" in args.loss_terms)
        if "style_v_all" in args.loss_terms:
            content_loss *= (args.perceptual_weight * args.content_weight)
            style_loss *= (args.perceptual_weight * args.style_weight)
            loss += (content_loss + style_loss)
        else:
            content_loss = torch.Tensor([0]).cuda()
            style_loss = torch.Tensor([0]).cuda()

        if "fine" in args.loss_terms:
            img_loss *= args.rgb_weight
            loss += img_loss
        else:
            img_loss = torch.Tensor([0]).cuda()

        if "coarse" in args.loss_terms:
            img_loss0 *= args.rgb_weight
            loss += img_loss0
        else:
            img_loss0 = torch.Tensor([0]).cuda()

        if "density" in args.loss_terms:
            if not args.self_distilled:
                if teacher is None:
                    print(f"[Warning] teacher model is None, skip density loss")
                    d_loss = torch.Tensor([0]).cuda()
                else:
                    d_loss_c = img2mae(ret_dict['raw0'][..., -1].reshape(-1, 1), ret_dict_teach['raw0'][..., -1].reshape(-1, 1))
                    d_loss_f = img2mae(ret_dict['raw'][..., -1].reshape(-1, 1), ret_dict_teach['raw'][..., -1].reshape(-1, 1))
                    d_loss = args.d_weight * (d_loss_c + d_loss_f)
                    loss += d_loss
            else:
                d_loss = img2mae(ret_dict['raw'][..., -1].reshape(-1, 1), ret_dict['alpha_old0'].reshape(-1, 1))
                d_loss = args.d_weight * d_loss
                loss += d_loss
        else:
            d_loss = torch.Tensor([0]).cuda()

        if "contrast" in args.loss_terms:
            c_loss = contrast_loss(gram_pred, stl_idx)

        # Optimize
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        ############################
        ##### Rest is logging ######
        ############################

        # logging errors
        if (global_step % i_print == 0 and global_step > 0) or global_step == 200001:
            dt = time.time() - time0
            time0 = time.time()
            avg_time = dt / min(global_step - start_step, i_print)
            try:
                gamma = model.module.nerf_fine.mlp.gamma
                lemma = model.module.nerf_fine.mlp.lemma
            except:
                gamma = lemma = torch.Tensor([0])

            print(f"[TRAIN] Iter: {global_step}/{max_steps}, Gamma:{round(gamma.item(), 4)}, Lemma:{round(lemma.item(), 4)}, Total-Loss: {round(loss.item(), 4)} \n \
                Img-Loss: {round(img_loss.item(), 4)} Density-Loss: {round(d_loss.item(), 4)}\n \
                Style-Loss:{round(style_loss.item(), 4)} Content-Loss: {round(content_loss.item(), 4)} \n \
                PSNR: {round(psnr.item(), 4)} Average Time: {round(avg_time, 4)}")

            # log training metric
            summary_writer.add_scalar('train/loss', loss, global_step)
            summary_writer.add_scalar('train/psnr', psnr, global_step)

            # log learning rate
            lr_groups = {}
            for i, param in enumerate(optimizer.param_groups):
                lr_groups['group_'+str(i)] = param['lr']
            summary_writer.add_scalars('l_rate', lr_groups, global_step)

        # save checkpoint
        if global_step % i_weights == 0 and global_step > 0:
            path = os.path.join(run_dir, 'checkpoints', '{:08d}.ckpt'.format(global_step))
            print('Checkpointing at', path)
            save_checkpoint(path, global_step, model, optimizer)

        # test images
        if (global_step % i_testset == 0 and global_step > 0) or global_step == 200001:
            print("Evaluating test images ...")
            save_dir = os.path.join(run_dir, 'testset_{:08d}'.format(global_step))
            os.makedirs(save_dir, exist_ok=True)
            metric_dict = evaluate([model, transformer], test_set, device=device, save_dir=save_dir, fast_mode=args.fast_mode, stl_idx=_stl_idx, bs=args.batch_size)

            # log testing metric
            summary_writer.add_scalar('test/mse', metric_dict['mse'], global_step)
            summary_writer.add_scalar('test/psnr', metric_dict['psnr'], global_step)

        # exhibition video
        if global_step % i_video==0 and global_step > 0 and exhibit_set is not None:
            render_video(model, exhibit_set, device=device, save_dir=run_dir, suffix=str(global_step), expname=args.expname, fast_mode=args.fast_mode, stl_idx=_stl_idx, bs=args.batch_size)

        # End training if finished
        if global_step >= max_steps:
            print(f'Train ends at global_step={global_step}')
            break

    return global_step

def save_checkpoint(path, global_step, model, optimizer):
    save_dict = {
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(save_dict, path)