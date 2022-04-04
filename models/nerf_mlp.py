import os, sys
from turtle import forward
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from models.embedder import Embedder

from utils.error import *
from pdb import set_trace as st

class MLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, no_skip=False, skips=[4], use_viewdirs=False, 
                act_fn="relu", zero_viewdir=False, embed_mlp=False):
        """
        MLP backbone for NeRF
        """ 
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.no_skip = no_skip
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.zero_viewdir = zero_viewdir
        self.embed_mlp = embed_mlp
        if act_fn.lower() == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn.lower() == "silu":
            self.act_fn = nn.SiLU() # inplace True or False?
        elif act_fn.lower() == "mish":
            self.act_fn = nn.Mish()
        elif act_fn.lower() == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise NotImplementedError("not a valid activation function")
        if self.embed_mlp:
            self.embed_depth = 4
            self.mix_rgb_net = nn.ModuleList(
                [nn.Linear(2*W, W)] + [nn.Linear(W, W) for i in range(self.embed_depth-1)])
            self.mix_d_net = nn.ModuleList(
                [nn.Linear(2*W, W)] + [nn.Linear(W, 1)])
            self.gamma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
            self.lemma = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        if self.no_skip:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        else:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

            ### Implementation according to the paper
            # self.views_linears = nn.ModuleList(
            #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

            self.rgb_linear = nn.Linear(W//2, output_ch-1)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, style_feature=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.act_fn(h)
            if not self.no_skip:
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            feature = self.feature_linear(h)

            # Disable view direction for AM
            if self.zero_viewdir:
                input_views = torch.zeros_like(input_views).to(input_views.device)

            # AM for color and density from SIM and CIM
            if self.embed_mlp:
                for i, l in enumerate(self.mix_d_net[:-1]):
                    if i == 0:
                        feature_d_mix = self.mix_d_net[i](torch.cat([h, style_feature], dim=-1))
                        feature_d_mix = self.act_fn(feature_d_mix)
                    else:
                        feature_d_mix = self.mix_d_net[i](feature_d_mix, dim=-1)
                        feature_d_mix = self.act_fn(feature_d_mix)
                alpha = self.lemma * alpha + (1 - self.lemma) * self.mix_d_net[-1](feature_d_mix)

                for i, l in enumerate(self.mix_rgb_net[:-1]):
                    if i == 0:
                        feature_mix = self.mix_rgb_net[i](torch.cat([feature, style_feature], dim=-1))
                        feature_mix = self.act_fn(feature_mix)
                    else:
                        feature_mix = self.mix_rgb_net[i](feature_mix)
                        feature_mix = self.act_fn(feature_mix)
                feature = self.gamma * feature + (1 - self.gamma) * self.mix_rgb_net[-1](feature_mix)

            # previous code
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = self.act_fn(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# Point query with embedding
class NeRFMLP(nn.Module):
    
    def __init__(self, input_dim=3, output_dim=4, net_depth=8, net_width=256, no_skip=False, act_fn="relu", skips=[4],
        viewdirs=True, use_embed=True, multires=10, multires_views=4, netchunk=1024*64, fix_weight=False, 
        zero_viewdir=False, embed_mlp=False, offset_mlp=False, embed_posembed=False, stl_num=None):

        super().__init__()

        self.chunk = netchunk
        self.embed_mlp = embed_mlp
        self.offset_mlp = offset_mlp
        self.embed_posembed = embed_posembed
        # self.embed_mlp = True
        # self.offset_mlp = False
        print(f"> Style Implicit Module: {self.embed_mlp}")
        print(f"> Cat position embedding to SIM: {self.embed_posembed}")

        # Provide empty periodic_fns to specify identity embedder
        periodic_fns = []
        if use_embed:
            periodic_fns = [torch.sin, torch.cos]

        self.embedder = Embedder(input_dim, multires, multires-1, periodic_fns,log_sampling=True, include_input=True)
        input_ch = self.embedder.out_dim

        input_ch_views = 0
        self.embeddirs = None
        if viewdirs:
            self.embeddirs = Embedder(input_dim, multires_views, multires_views-1, periodic_fns, log_sampling=True, include_input=True)
            input_ch_views = self.embeddirs.out_dim

        output_ch = output_dim
        self.mlp = MLP(net_depth, net_width, no_skip=no_skip, act_fn=act_fn, skips=skips, input_ch=input_ch,
            output_ch=output_ch, input_ch_views=input_ch_views, use_viewdirs=viewdirs, zero_viewdir=zero_viewdir, embed_mlp=self.embed_mlp)
        
        if self.embed_mlp:
            self.embed_depth = 4
            self.embed_dim = stl_num
            W = net_width
            if self.embed_posembed:
                self.embed_net = nn.ModuleList(
                    [nn.Linear(self.embed_dim, W)] + [nn.Linear(W+63+self.embed_dim, W)] + [nn.Linear(W+self.embed_dim, W) for i in range(self.embed_depth-2)])
            else:
                self.embed_net = nn.ModuleList(
                    [nn.Linear(self.embed_dim, W)] + [nn.Linear(W, W) for i in range(self.embed_depth-1)])
            self.act_fn = nn.ReLU()

    def batchify(self, inputs):
        """Single forward feed that applies to smaller batches.
        """
        query_batches = []
        for i in range(0, inputs.shape[0], self.chunk):
            end = min(i+self.chunk, inputs.shape[0])
            h = self.mlp(inputs[i:end]) # [N_chunk, C]
            query_batches.append(h)
        outputs = torch.cat(query_batches, 0) # [N_pts, C]
        return outputs

    def forward(self, inputs, viewdirs=None, stl_idx=None, **kwargs):
        """Prepares inputs and applies network.
        inputs: shape:[1024, 64, 3]
        viewdirs: shape:[1024, 3]
        """
        # Flatten
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [N_pts, C]
        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        # Batchify
        output_chunks = []
        for i in range(0, inputs_flat.shape[0], self.chunk):
            end = min(i+self.chunk, inputs_flat.shape[0])
            embedded = self.embedder(inputs_flat[i:end])
            style_feature = None
            
            # Style Implicit Module, to learn the conditional style embedding 
            if self.embed_mlp:
                # add the position embedding to learned conditional style feature 
                if self.embed_posembed:
                    _stl_idx = stl_idx.expand(end-i, stl_idx.shape[-1])
                    for ii, l in enumerate(self.embed_net):
                        if ii == 1:
                            stl_embed = self.embed_net[ii](torch.cat([stl_embed, embedded, _stl_idx], -1))
                            stl_embed = self.act_fn(stl_embed)
                        elif ii == 0:
                            stl_embed = self.embed_net[ii](_stl_idx)
                            stl_embed = self.act_fn(stl_embed)
                        else:
                            # stl_embed = self.embed_net[ii](stl_embed)
                            stl_embed = self.embed_net[ii](torch.cat([stl_embed, _stl_idx], -1))
                            stl_embed = self.act_fn(stl_embed)
                    style_feature = stl_embed
                else:
                    for ii, l in enumerate(self.embed_net):
                        stl_embed = self.embed_net[ii](stl_embed)
                        stl_embed = self.act_fn(stl_embed)
                    style_feature = stl_embed.expand([end-i, stl_embed.shape[-1]])

            embedded = self.embedder(inputs_flat[i:end])

            # append view direction embedding
            if self.embeddirs is not None:
                embedded_dirs = self.embeddirs(input_dirs_flat[i:end])
                embedded = torch.cat([embedded, embedded_dirs], -1)
            h = self.mlp(embedded, style_feature=style_feature) # [N_chunk, C]
            output_chunks.append(h)
        outputs_flat = torch.cat(output_chunks, 0) # [N_pts, C]

        # Unflatten
        sh = list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        return torch.reshape(outputs_flat, sh)


class EmbedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass