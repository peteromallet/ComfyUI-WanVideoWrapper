import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import comfy.model_management as mm
from utils import log
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from typing import Optional, List
import inspect

from enhance_a_video.enhance import get_feta_scores
from enhance_a_video.globals import is_enhance_enabled

# SageAttention
try:
    from sageattention import sageattn
except:
    pass

from diffusers.models.modeling_utils import ModelMixin
from einops import repeat, rearrange

try:
    from flex_attention.flex_attention import flex_attention
except:
    pass

from comfy.ldm.flux.math import apply_rope as apply_rope_comfy

def rope_riflex(pos, dim, theta, L_test, k, temporal):
    if temporal:
        freqs = pos * torch.exp(
            torch.linspace(
                0, -math.log(theta), dim // 2, dtype=torch.float32, device=pos.device))
    else:
        freqs = pos * torch.exp(
            k *
            torch.linspace(
                0, -math.log(theta), dim // 2, dtype=torch.float32, device=pos.device))
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)

def rope_params(dim, theta=10000, L_test=81, k=1):
    pos = torch.arange(L_test).to(torch.float32)
    freqs_i = torch.exp(
        k *
        torch.linspace(0, -math.log(theta), dim, dtype=torch.float32, device=pos.device))
    freqs = pos.unsqueeze(1) * freqs_i.unsqueeze(0)
    return torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)

class WanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=None):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, self.w.shape, self.w, self.b, self.eps)

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=None):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class WanAttention(nn.Module):
    def __init__(self, dim, num_heads, text_len, eps, attention_mode, main_device, offload_device, dtype):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.text_len = text_len
        self.attention_mode = attention_mode
        self.main_device = main_device
        self.offload_device = offload_device
        self.dtype = dtype

    def forward(self, x, context, freqs):
        with torch.autocast(device_type=mm.get_autocast_device(self.main_device), dtype=self.dtype, enabled=True):
            B, T, C = x.shape
            q = self.q(x)
            k = self.k(context)
            v = self.v(context)
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            if freqs is not None:
                q = apply_rope_comfy(q, freqs).to(q)
                k = apply_rope_comfy(k, freqs).to(k)

            if "sdpa" in self.attention_mode:
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            elif "flash_attn" in self.attention_mode:
                try:
                    from flash_attn import flash_attn_func
                    x = flash_attn_func(q, k, v, causal=False)
                except Exception as e:
                    log.warning(f"Could not use flash_attn: {e}, falling back to sdpa")
                    x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            elif "sage" in self.attention_mode:
                x = sageattn(q, k, v, None, False)

            x = x.transpose(1, 2).reshape(B, T, C)
        return self.o(x)
    
class WanModel(nn.Module):
    def __init__(self, dim, ffn_dim, eps, freq_dim, in_dim, model_type, out_dim, text_len, num_heads, num_layers, 
                 attention_mode, main_device, offload_device, teacache_coefficients, vace_layers=None, 
                 vace_in_dim=None, inject_sample_info=False, add_ref_conv=False, in_dim_ref_conv=None, add_control_adapter=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.eps = eps
        self.freq_dim = freq_dim
        self.in_dim = in_dim
        self.model_type = model_type
        self.out_dim = out_dim
        self.text_len = text_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_mode = attention_mode
        self.main_device = main_device
        self.offload_device = offload_device
        self.vace_layers = vace_layers
        self.inject_sample_info = inject_sample_info
        self.add_ref_conv = add_ref_conv
        self.add_control_adapter = add_control_adapter
        self.dtype = torch.bfloat16 # for rope

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.rope_embedder = RopeEmbedder(dim=dim, num_heads=num_heads)
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, text_len=text_len, eps=eps, attention_mode=attention_mode, main_device=main_device, offload_device=offload_device, dtype=self.dtype) for _ in range(num_layers)
        ])
        if vace_layers is not None:
            self.vace_blocks = nn.ModuleList([
                VACEBlock(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=eps, attention_mode=attention_mode, main_device=main_device, offload_device=offload_device) for _ in range(15)
            ])
            self.vace_embedding = nn.Conv2d(vace_in_dim, dim, kernel_size=1, stride=1, padding=0)
            self.vace_pos_embedding = nn.Parameter(torch.randn(1, 1024, dim))

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim, bias=False)
        self.teacache_coefficients = teacache_coefficients
        self.enable_teacache = False
        self.teacache_state = TeaCacheState()
        self.use_non_blocking = True
        self.controlnet = None

        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.motion_speed_embedding = nn.Embedding(256, dim)
            self.camera_motion_embedding = nn.Embedding(256, dim)
        if add_ref_conv:
            self.ref_conv = nn.Conv3d(in_dim_ref_conv, dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))

        if add_control_adapter:
            from controlnet.ldm.modules.attention import SpatialTransformer
            from controlnet.ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
            from controlnet.ldm.modules.diffusionmodules.util import conv_nd, linear
            control_model_config = {
                "in_channels": 4,
                "model_channels": 320,
                "out_channels": 4,
                "num_res_blocks": 2,
                "attention_resolutions": [4, 2, 1],
                "dropout": 0,
                "channel_mult": [1, 2, 4, 4],
                "num_classes": "sequential",
                "use_checkpoint": False,
                "image_size": 32,
                "num_heads": 8,
                "num_head_channels": 64,
                "num_heads_upsample": -1,
                "use_scale_shift_norm": True,
                "resblock_updown": True,
                "use_new_attention_order": False,
                "use_spatial_transformer": True,
                "transformer_depth": 1,
                "context_dim": 768,
                "n_embed": None,
                "legacy": False,
                "disable_self_attentions": None,
                "num_attention_blocks": None,
                "disable_middle_self_attn": False,
                "use_linear_in_transformer": False,
            }
            self.control_adapter = SpatialTransformer(**control_model_config)

    def forward(self, x, context, y=None, clip_fea=None, is_uncond=False, 
                current_step=0, seq_len=None, freqs=None, t=None,
                control_lora_enabled=False, camera_embed=None, unianim_data=None, fun_ref=None, fun_camera=None,
                pred_id=None, vace_data=None, audio_proj=None, audio_context_lens=None, audio_scale=None,
                pcd_data=None, controlnet=None):
        pass

class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, text_len, eps, attention_mode, main_device, offload_device, dtype):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanAttention(dim, num_heads, text_len, eps, attention_mode, main_device, offload_device, dtype)
        self.norm2 = WanLayerNorm(dim, eps)
        self.cross_attn = WanAttention(dim, num_heads, text_len, eps, attention_mode, main_device, offload_device, dtype)
        self.norm3 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )

    def forward(self, x, context, freqs):
        x = self.self_attn(self.norm1(x), x, freqs) + x
        x = self.cross_attn(self.norm2(x), context, freqs) + x
        x = self.ffn(self.norm3(x)) + x
        return x

class RopeEmbedder(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = None
        self.num_frames = None

    def forward(self, B, T, C, device, dtype):
        pos = torch.arange(T, device=device).unsqueeze(1)
        if self.k is None:
            freqs = rope_params(self.head_dim, L_test=T, k=1)
        else:
            freqs = rope_riflex(pos, self.head_dim, 10000, self.num_frames, self.k, True)
        freqs_cis = freqs.to(dtype)
        return freqs_cis.view(1, T, 1, -1)
    
class TeaCacheState:
    def __init__(self):
        self.states = {}

    def get(self, pred_id, step, default=None):
        if pred_id not in self.states or 'cache' not in self.states[pred_id] or step not in self.states[pred_id]['cache']:
            return default
        return self.states[pred_id]['cache'][step]

    def set(self, pred_id, step, value):
        if pred_id not in self.states:
            self.states[pred_id] = {'cache': {}}
        self.states[pred_id]['cache'][step] = value

    def clear(self, pred_id):
        if pred_id in self.states:
            self.states[pred_id].clear()
            
    def clear_all(self):
        self.states.clear()

    def add_skipped_step(self, pred_id, step):
        if pred_id not in self.states:
            self.states[pred_id] = {'skipped_steps': []}
        elif 'skipped_steps' not in self.states[pred_id]:
            self.states[pred_id]['skipped_steps'] = []
        self.states[pred_id]['skipped_steps'].append(step) 