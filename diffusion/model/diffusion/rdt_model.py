# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.````
# --------------------------------------------------------
# References:``````````
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from collections import OrderedDict
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
# 也许是比diffusion policy那篇更好的transformer

#################################################################################
#               Embedding Layers for Timesteps and Condition Inptus             #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    总的来说,这个类的作用就是将timestep,freq转化为一个embedding,
    然后将这个embedding通过mlp输出一个形状为(hidden_size ,hidden_size)的数据，那么意义是什么？
    是:
    将标量时间步转换为丰富的特征表示
    让模型能够"理解"不同扩散阶段的特点
    指导模型在不同阶段采取不同的去噪策略
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        # standard mlp with linear layers and activation and linear output
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep2embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # freqs 从高频到低频，范围从 1 到 1/max_period
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        # ----计算时间步和频率的外积----
        args = t[:, None].float() * freqs[None]# 增加一个维度,前面那个所有行增加，也就是增加一个列，后面的相当于增加一个行
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)# 合并cos和sin捕捉周期特征
        # 如果 t=[0,50,999], freqs=[f1,f2,...,f128]
        # args 形状为 (3, 128)，每行是时间步与所有频率的乘积
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)# 全是zero的like某种形状的 就是zeros_like
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep2embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # 草，我懂了，传入的c里面是包含了lan和img的，[B，[lan,img]],并且这两个都有着相同的Dimention，
        # 所以看似是直接ubind了，其实巧妙的把这两个分开了
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # permute改变原有维度的顺序，将2这个维度提到了最前面，这样就可以将2这个维度解绑
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


#################################################################################
#                                 RDT Block                                     #
#################################################################################
class RDTBlock(nn.Module):
    """
    A RDT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, 
            act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, mask=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x
        
        origin_x = x
        x = self.norm2(x)
        x = self.cross_attn(x, c, mask)
        x = x + origin_x
                
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, 
                                  embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim
    
    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)
    
    return c_pos_emb

class RDT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # 用nn.parameter来创建可训练的参数,每次会被保存到.ckpt,会被优化器更新
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon+3, hidden_size))
        # Language conditions
        self.lang_cond_pos_embed = nn.Parameter(
            torch.zeros(1, max_lang_cond_len, hidden_size))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len, hidden_size))

        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                ('ctrl_freq', 1),
                ('state', 1),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(
            torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
            
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        Forward pass of RDT.
        
        x: (B, T, D), state + action token sequence, T = horizon + 1,
            dimension D is assumed to be the same as the hidden size.
        freq是RDT相比CNN-based的一个重要扩展,因为它允许模型
        - 更精确地控制动作执行的时间特性
        - 更灵活地处理不同频率的控制信号
        - 在训练和推理时更高效地处理不同频率的控制信号
        freq: (B,), a scalar indicating control frequency.
        t: (B,) or (1,), diffusion timesteps.
        lang_c: (B, L_lang, D) or None, language condition tokens (variable length),
            dimension D is assumed to be the same as the hidden size.
        img_c: (B, L_img, D) or None, image condition tokens (fixed length),
            dimension D is assumed to be the same as the hidden size.
        lang_mask: (B, L_lang) or None, language condition mask (True for valid).
        img_mask: (B, L_img) or None, image condition mask (True for valid).
        """
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)
        # Append timestep to the input tokens
        print(f't is {t.shape}')
        print(f'freq is {freq.shape}')
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)               # x原本的(B, T+1, D)
        print(f'x is {x.shape}')
        print(f'x_pos_embed is {self.x_pos_embed.shape}')
        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        # Note the lang is of variable length
        print(f'lang_cond_pos_embed[:, :lang_c.shape[1]] is {self.lang_cond_pos_embed[:, :lang_c.shape[1]].shape}')
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        # Forward pass
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):#一个block就是一个RDT的模块
            c, mask = conds[i%2], masks[i%2]# 交替注入避免信息淹没：1）防止大量的图像token压倒少量的文本token 2）确保模型能够充分关注语言指令 3）更好的特征融合
            x = block(x, c, mask)                       # (B, T+1, D)
        # Inject the language condition at the final layer
        x = self.final_layer(x)                         # (B, T+1, out_channels)

        # Only preserve the action tokens
        x = x[:, -self.horizon:]
        return x
