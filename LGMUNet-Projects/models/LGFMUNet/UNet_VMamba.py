from einops import rearrange
from copy import deepcopy
from torch import nn
import torch
import numpy as np
from thop import profile, clever_format
import torch
from torchsummary import summary

import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    # mamba_ssm
    try:
        import os as _os, sys as _sys
        _proj_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        if _proj_root not in _sys.path:
            _sys.path.insert(0, _proj_root)
        from models.selective_scan.test_selective_scan_easy import build_api_selective_scan, selective_scan_ref
        selective_scan_fn = build_api_selective_scan(chunksize=64)
    except Exception as _e:
        print(f"Warning: Failed to import selective_scan_fn: {_e}. "
              f"Please install mamba_ssm: pip install mamba_ssm")
        selective_scan_fn = None
        selective_scan_ref = None
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"  # Mamba


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...

    return flops




# MLP module with GELU activation and Dropoutclass Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Convolution stem projection block with CNN and activationclass project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x


# Deconvolution stem projection block with transposed conv and activation
class project_up(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x


# Full convolution stem with number of blocks determined by patch_sizeclass PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block = int(np.log2(patch_size[0]))
        self.project_block = []
        self.dim = [int(embed_dim) // (2 ** i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim = self.dim[::-1]  # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim

        for i in range(self.num_block)[:-1]:
            self.project_block.append(project(self.dim[i], self.dim[i + 1], 2, 1, nn.GELU, nn.LayerNorm, False))
        self.project_block.append(project(self.dim[-2], self.dim[-1], 2, 1, nn.GELU, nn.LayerNorm, True))
        self.project_block = nn.ModuleList(self.project_block)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        for blk in self.project_block:
            x = blk(x)

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

# Symmetric decoder final projection layerclass final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.num_block=int(np.log2(patch_size[0]))-2
        self.project_block=[]
        self.dim_list=[int(dim)//(2**i) for i in range(self.num_block+1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i],self.dim_list[i+1],nn.GELU,nn.LayerNorm,False))
        self.project_block=nn.ModuleList(self.project_block)
        self.up_final=nn.ConvTranspose2d(self.dim_list[-1],num_class,4,4)

    def forward(self,x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x)
        return x


# Downsampling moduleclass PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x

# Upsampling moduleclass Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose2d(dim, dim // 2, 2, 2)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up(x)
        return x


# Custom layer normalizationclass LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# SS2D
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        if selective_scan_fn is None:
            raise RuntimeError(
                "selective_scan_fn is not available. Install mamba_ssm: "
                "pip install mamba_ssm"
            )
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# Mamba
class LGMamba(nn.Module):
    """
    Corresponds to original MSABlock but uses SS2D instead of self-attention.

    Architecture:
    1. Input features
    2. Save residual connection (shortcut)
    3. Layer normalization
    4. Dual-branch processing:
       a. SS2D scan branch
       b. Depthwise convolution branch
    5. Sum of both branches
    6. Projection layer + Dropout
    7. Residual connection: shortcut + processed result
    """

    def __init__(self, dim, d_state=16, drop_rate=0.):
        """
        Args:
            dim (int): Input channel dimension
            d_state (int): SS2D state space dimension
            drop_rate (float): Dropout probability
        """
        super().__init__()
        self.dim = dim

        # (MSABlocknorm1)
        self.norm = nn.LayerNorm(dim)

        # SS2D ()
        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=3,
            expand=2,
            dropout=drop_rate
        )

        # Depthwise convolution branch        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Dropout (WindowAttentionprojproj_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        """
:
 x: [B, H, W, C] ()
:
 x: [B, H, W, C] 
        """
        shortcut = x

        # Layer normalization        x = self.norm(x)

        # ===== =====
        # SS2D ([B, H, W, C])
        ss2d_branch = self.ss2d(x)

        # ()
        dw_branch = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        dw_branch = self.dwconv(dw_branch)
        dw_branch = dw_branch.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # ===== =====
        # WindowAttention x = x + dw
        x = ss2d_branch + dw_branch

        # ===== Dropout =====
        # WindowAttention proj proj_drop
        x = self.proj(x)
        x = self.proj_drop(x)

        # ===== =====
        # MSABlock shortcut + x
        x = shortcut + x

        return x

# Overall block designclass LGBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, d_state=16, drop_rate=0.):
        super().__init__()
        self.dim = dim

        # ===== =====
        # Depthwise convolution        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # (LayerNorm)
        self.norm_conv = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # MLP (2)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer scale parameter        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

        # ===== =====
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ===== LGMamba =====
        self.lgmamba = LGMamba(
            dim=dim,
            d_state=d_state,
            drop_rate=drop_rate
        )

    def forward(self, x):
        """
:
 x: [B, C, H, W] 
:
 x: [B, C, H, W] 
        """
        input = x

        # ===== =====
        # Depthwise convolution        x = self.dwconv(x)
        x = self.norm_conv(x)

        # MLP
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scaling        if self.gamma is not None:
            x = self.gamma * x

        # ===== =====
        # :
        x = input.permute(0, 2, 3, 1) + self.drop_path(x)  # [B, H, W, C]

        # ===== LGMamba =====
        # Ensure contiguous        x = x.contiguous()
        x = self.lgmamba(x)

        # ===== =====
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]

        return x

# Encoder layerclass BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, drop_path=0.,
                 d_state=16, drop_rate=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.downsample = downsample

        # LGBlock
        self.blocks = nn.ModuleList([
            LGBlock(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                d_state=d_state,
                drop_rate=drop_rate
            )
            for i in range(depth)
        ])

        # Downsample layer        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)

    def forward(self, x, H, W):
        """
 x: [B, C, H, W] 
 H, W: 
:
 x: [B, C, H, W]
 x_down: [B, 2C, H/2, W/2] ()
        """
        # LGBlock
        for blk in self.blocks:
            x = blk(x)  # [B, C, H, W]

        # Downsample processing        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

# Decoder layerclass BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, drop_path=0.,
                 d_state=16, drop_rate=0., upsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.upsample = upsample

        # LGBlock
        self.blocks = nn.ModuleList([
            LGBlock(
                dim=dim // 2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                d_state=d_state,
                drop_rate=drop_rate
            )
            for i in range(depth)
        ])

        # Upsample layer        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=nn.LayerNorm)

    def forward(self, x, skip, H, W):
        """
 x: [B, C, H, W] 
 skip: [B, C_skip, H_skip, W_skip] 
 H, W: 
:
 x: [B, C//2, 2H, 2W]
        """
        # Upsample operation        if self.upsample is not None:
            x = self.upsample(x, H, W)  # [B, C//2, 2H, 2W]
            H, W = H * 2, W * 2

        if skip is not None:
            # Add size adjustment logic            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            if x.shape[1] != skip.shape[1]:
                skip = nn.Conv2d(skip.shape[1], x.shape[1], kernel_size=1)(skip)
            x = x + skip

        # LGBlock
        for blk in self.blocks:
            x = blk(x)  # [B, C, H, W]

        return x, H, W


# Encoderclass Encoder(nn.Module):
    def __init__(
            self,
            pretrain_img_size=[224, 224],
            patch_size=[4, 4],
            in_chans=1,
            embed_dim=96,
            depths=[3, 3, 3, 3],
            d_state=16,  # Mamba
            drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # 1.
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 3.
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Compute current layer input resolution            H = pretrain_img_size[0] // patch_size[0] // (2 ** i_layer)
            W = pretrain_img_size[1] // patch_size[1] // (2 ** i_layer)

            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(H, W),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                d_state=d_state,  # Mamba
                drop_rate=drop_rate,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # 4.
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # 5.
        for i_layer in out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
 """"""
        # Initial patch embedding        x = self.patch_embed(x)
        down = []

        # Get initial spatial dimensions        Wh, Ww = x.size(2), x.size(3)
        x = self.pos_drop(x)

        # Process layer by layer        for i in range(self.num_layers):
            layer = self.layers[i]
            # : (, H, W, , H, W)
            x_out, H, W, x_down, Wh, Ww = layer(x, Wh, Ww)

            # Save current layer output            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')

                # : [B, C, H, W] -> [B, H, W, C]
                x_out = x_out.permute(0, 2, 3, 1)
                x_out = norm_layer(x_out)

                # Convert back to original dimension format                out = x_out.permute(0, 3, 1, 2).contiguous()
                down.append(out)

            # Update input to downsampled output            x = x_down

        return down


class Decoder(nn.Module):
    def __init__(
            self,
            pretrain_img_size,
            embed_dim,
            encoder_depth,
            patch_size=[4, 4],
            depths=[3, 3, 3],
            d_state=16,
            drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))][::-1]

        # Compute bottleneck channels        bottleneck_dim = embed_dim * 2 ** (encoder_depth - 1)

        # （）
        decoder_dims = [bottleneck_dim // (2 ** i) for i in range(self.num_layers)]

        # （）
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Compute current layer resolution            scale_factor = encoder_depth - 1 - i
            H = pretrain_img_size[0] // patch_size[0] // (2 ** scale_factor)
            W = pretrain_img_size[1] // patch_size[1] // (2 ** scale_factor)

            layer = BasicLayer_up(
                dim=decoder_dims[i],
                input_resolution=(H, W),
                depth=depths[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                d_state=d_state,
                drop_rate=drop_rate,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)

        # 3.
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        """
        Args:
            x: Bottleneck features [B, C, H, W]
            skips: Encoder skip connections (ordered: shallow->deep)
        """
        # Reverse skip connections: shallow->deep
        skips = skips[::-1]

        outs = []
        H, W = x.size(2), x.size(3)
        x = self.pos_drop(x)

        # （）
        for i, layer in enumerate(self.layers):
            # Get corresponding skip connection            skip = skips[i] if i < len(skips) else None

            # Decoder layer processing            x, H, W = layer(x, skip, H, W)
            outs.append(x)

        return outs

# Overall LGMUNet model
class LGMUNet(nn.Module):
    def __init__(
            self,
            config=None,
            num_input_channels=3,
            embedding_dim=96,
            d_state=16,
            num_classes=1,
            deep_supervision=False,
            conv_op=nn.Conv2d
    ):
        super(LGMUNet, self).__init__()

        # Process config parameters
        if config is None:
            img_size = (448, 448)
            convolution_stem_down = 8
            blocks_num = [3, 3, 3, 3]
            drop_rate = 0.1
        else:
            img_size = config.hyper_parameter.crop_size
            convolution_stem_down = config.hyper_parameter.convolution_stem_down
            blocks_num = config.hyper_parameter.blocks_num
            drop_rate = config.hyper_parameter.drop_rate

        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision
        self.embed_dim = embedding_dim
        self.depths = blocks_num
        self.img_size = img_size
        self.patch_size = [convolution_stem_down, convolution_stem_down]

        # 1.
        self.encoder = Encoder(
            pretrain_img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.num_input_channels,
            embed_dim=self.embed_dim,
            depths=self.depths,
            d_state=d_state,
            drop_rate=drop_rate
        )

        # 2.
        decoder_depths = self.depths[::-1][1:]
        self.decoder = Decoder(
            pretrain_img_size=self.img_size,
            embed_dim=self.embed_dim,
            encoder_depth=len(self.depths),
            patch_size=self.patch_size,
            depths=decoder_depths,
            d_state=d_state,
            drop_rate=drop_rate
        )

        # 3. -
        self.final = nn.ModuleList()
        # （）
        # Formula: embed_dim * 2^(total_layers - 2 - layer_index)
        decoder_out_dims = [
            self.embed_dim * (2 ** (len(self.depths) - 2 - i))
            for i in range(len(self.depths) - 1)
        ]

        for dim in decoder_out_dims:
            self.final.append(
                final_patch_expanding(dim, self.num_classes, patch_size=self.patch_size)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        seg_outputs = []

        # Encoder path        skips = self.encoder(x)

        # Bottleneck as decoder input        neck = skips[-1]

        # Decoder path        # Note: skip last encoder feature map (bottleneck) as it is already the input
        decoder_outs = self.decoder(neck, skips[:-1])

        # Apply final upsampling for each decoder output        for i in range(len(decoder_outs)):
            # Apply corresponding upsampling head            seg_out = self.final[i](decoder_outs[i])
            seg_outputs.append(self.sigmoid(seg_out))

        # Deep supervision handling        if self.do_ds:
            # Training: return all scale outputs            return seg_outputs[::-1]
        else:
            # /
            return seg_outputs[-1]


# Create model instancemodel = LGMUNet(
    num_input_channels=3,    # ()
    embedding_dim=96,        # ()
    num_classes=1,           # ()
    deep_supervision=False
)

# Print model structureprint("="*50)
print(":")
#print(model)
print("="*50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # GPU
input_tensor = torch.randn(1, 3, 448, 448).to(device)  # GPU


# Forward pass
with torch.no_grad():
    output = model(input_tensor)
 print(f": {input_tensor.shape}")
 print(f": {output.shape}")

# Compute model parameter countstotal_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*50)
print(f": {total_params:,}")
print(f": {trainable_params:,}")

# FLOPs
flops, params = profile(model, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")

print("="*50)
print(f"FLOPs: {flops}")
print(f": {params}")

# Test model output rangeprint("="*50)
print(":")
print(f": {output.min().item():.4f}")
print(f": {output.max().item():.4f}")
print(f": {output.mean().item():.4f}")