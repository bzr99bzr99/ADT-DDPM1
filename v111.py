import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# -------------------------
# 工具函数 (Utilities)
# -------------------------
def get_norm(dim: int, groups: int = 8) -> nn.GroupNorm:
    """自动计算合理的 GroupNorm 分组数"""
    if dim < groups:
        return nn.GroupNorm(1, dim)
    return nn.GroupNorm(min(groups, dim // 2) if dim >= 2 and dim % groups == 0 else 1, dim)


# -------------------------
# 位置编码模块 (Positional Embedding)
# -------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale
        # 预计算因子，避免 forward 中重复计算
        half_dim = dim // 2
        self.register_buffer('inv_freq', torch.exp(torch.arange(half_dim) * -(math.log(10000) / half_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] or [B, 1]
        emb = torch.outer(x.squeeze() * self.scale, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -------------------------
# 自适应扩张卷积 (Adaptive Dilation Conv)
# -------------------------
class AdaptiveDilationConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation_rates: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 reduction_ratio: int = 4):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=d * (kernel_size // 2), dilation=d, bias=False)
            for d in dilation_rates
        ])

        # 动态权重生成器
        hidden_dim = max(in_channels // reduction_ratio, 8)
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(dilation_rates))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # 计算注意力权重: (B, C, L) -> (B, L, C) -> (B, L, K) -> (B, K, L)
        scores = self.gate(x.transpose(1, 2)).transpose(1, 2)
        weights = F.softmax(scores, dim=1)  # (B, K, L)

        # 并行计算所有路径: List[(B, C_out, L)] -> (B, K, C_out, L)
        stacked_outputs = torch.stack([conv(x) for conv in self.paths], dim=1)

        # 加权融合: (B, K, C, L) * (B, K, 1, L) -> Sum over K -> (B, C, L)
        return torch.sum(stacked_outputs * weights.unsqueeze(2), dim=1)


# -------------------------
# 改进的残差块 (Residual Block)
# -------------------------
class ResBlock1D_Adaptive(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: Optional[int] = None, cond_emb_dim: Optional[int] = None,
                 norm_groups: int = 8):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            AdaptiveDilationConv1D(in_channels, out_channels),
            get_norm(out_channels, norm_groups),
            nn.SiLU()
        )

        # 嵌入投影层
        self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.cond_proj = nn.Linear(cond_emb_dim, out_channels) if cond_emb_dim else None

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            get_norm(out_channels, norm_groups),
            nn.SiLU()
        )

        # Shortcut
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None,
                cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x)

        # 注入时间与条件嵌入
        if self.time_proj is not None and time_emb is not None:
            h = h + self.time_proj(F.silu(time_emb))[..., None]
        if self.cond_proj is not None and cond_emb is not None:
            h = h + self.cond_proj(F.silu(cond_emb))[..., None]

        h = self.block2(h)
        return h + self.shortcut(x)


# -------------------------
# 注意力模块 (Attention Modules)
# -------------------------
class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, max_len_full: int = 512):
        super().__init__()
        self.dim = dim
        self.max_len_full = max_len_full
        self.norm = get_norm(dim)
        self.mhsa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.local_attn = nn.Sequential(
            nn.Conv1d(dim, dim, 7, padding=3, groups=dim),
            nn.Conv1d(dim, dim, 1),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        res = x
        x_norm = self.norm(x)

        if l <= self.max_len_full:
            x_perm = x_norm.transpose(1, 2)
            out, _ = self.mhsa(x_perm, x_perm, x_perm)
            out = out.transpose(1, 2)
        else:
            out = self.local_attn(x_norm)

        return res + self.gamma * out


class CrossAttention1D(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, inner_dim: int, num_heads: int = 1):
        super().__init__()
        self.heads = num_heads
        # [修复] 显式保存 head_dim，避免 forward 中出现两个未知维度
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), context: (B, L_c, C_c)
        b, c, l = x.shape

        # Q: (B, L, inner_dim) -> (B, L, heads, head_dim) -> (B, heads, L, head_dim)
        q = self.to_q(x.transpose(1, 2)).view(b, l, self.heads, -1).transpose(1, 2)

        k = self.to_k(context).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, -1, self.heads, self.head_dim).transpose(1, 2)

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Output projection
        out = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        return x + self.gamma * self.to_out(out).transpose(1, 2)


# -------------------------
# UNet 构件 (UNet Components)
# -------------------------
class DownBlock(nn.Module):
    """编码层块：ResBlock -> Attn -> Downsample"""

    def __init__(self, in_dim: int, out_dim: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.res = ResBlock1D_Adaptive(in_dim, out_dim, time_dim, cond_dim)
        self.attn = TemporalSelfAttention(out_dim)
        self.down = nn.Conv1d(out_dim, out_dim, 4, stride=2, padding=1)

    def forward(self, x, t_emb, c_emb):
        x = self.res(x, t_emb, c_emb)
        x_skip = self.attn(x)  # 此处输出作为 skip connection
        x_down = self.down(x_skip)
        return x_down, x_skip


class UpBlock(nn.Module):
    """解码层块：Upsample -> Concat -> Attn -> ResBlock"""

    def __init__(self, in_dim: int, out_dim: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_dim, out_dim, 4, stride=2, padding=1)
        self.attn = TemporalSelfAttention(out_dim * 2)  # 处理 concat 后的特征
        self.res = ResBlock1D_Adaptive(out_dim * 2, out_dim, time_dim, cond_dim)

    def forward(self, x, skip, t_emb, c_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.attn(x)
        return self.res(x, t_emb, c_emb)


# -------------------------
# 完整的 UNet1D 模型
# -------------------------
class UNet1D(nn.Module):
    def __init__(self, time_emb_dim=128, base_channels=64, channel_mults=(1, 2, 4, 8, 8),
                 cond_emb_dim=128, num_classes=10, num_diffusion_timesteps=500,
                 cross_attn_inner_dim_ratio=1):
        super().__init__()

        # 1. 嵌入层
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Embedding(num_classes, cond_emb_dim, padding_idx=0),
            nn.Linear(cond_emb_dim, cond_emb_dim * 4), nn.SiLU(),
            nn.Linear(cond_emb_dim * 4, cond_emb_dim)
        )
        self.alphas = nn.Parameter(torch.zeros(num_diffusion_timesteps))

        eff_mults = channel_mults[:-1]
        dims = [base_channels * m for m in eff_mults]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(64,128), (128,256), (256,512)]

        self.init_conv = nn.Conv1d(1, dims[0], 1)

        # 3. 编码器 (Encoder)
        self.downs = nn.ModuleList([])
        # Layer 1 特殊处理（通道不变）
        self.downs.append(DownBlock(dims[0], dims[0], time_emb_dim, cond_emb_dim))
        # Layer 2-4
        for ind, outd in in_out:
            self.downs.append(DownBlock(ind, outd, time_emb_dim, cond_emb_dim))

        # 4. 瓶颈层 (Bottleneck)
        mid_dim = dims[-1]  # 对应 Layer 4 的输出通道 (chs[3])
        self.mid_res1 = ResBlock1D_Adaptive(mid_dim, mid_dim, time_emb_dim, cond_emb_dim)
        self.mid_attn_self = TemporalSelfAttention(mid_dim)
        self.mid_attn_cross = CrossAttention1D(mid_dim, cond_emb_dim, mid_dim * cross_attn_inner_dim_ratio)
        self.mid_res2 = ResBlock1D_Adaptive(mid_dim, mid_dim, time_emb_dim, cond_emb_dim)

        # 5. 解码器 (Decoder)
        self.ups = nn.ModuleList([])
        # 对应 Encoder Layer 4, 3, 2, 1 的逆序
        # Upsample 路径: mid -> L4 -> L3 -> L2 -> L1
        # Dec 2 (chs[3]->chs[3]), Dec 3 (chs[3]->chs[2]), Dec 4 (chs[2]->chs[1]), Dec 5 (chs[1]->chs[0])

        # Decoder 2: mid -> dims[-1] (保持通道)
        self.ups.append(UpBlock(mid_dim, mid_dim, time_emb_dim, cond_emb_dim))

        # Decoder 3, 4, 5
        rev_in_out = list(zip(dims[1:], dims[:-1]))[::-1]  # [(512,256), (256,128), (128,64)]
        for ind, outd in rev_in_out:
            self.ups.append(UpBlock(ind, outd, time_emb_dim, cond_emb_dim))

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _get_alpha(self, t):
        """获取特定时间步的混合权重"""
        if t.ndim == 0:
            idx = t.long().view(1)
        else:
            idx = t.long()
        return self.alphas[idx].view(-1, 1, 1)

    def forward(self, x, t, c):
        # 嵌入计算
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(c)
        c_ctx = c_emb.unsqueeze(1)
        alpha = self._get_alpha(t)

        x = self.init_conv(x)

        # Encoder: 自动处理 Skip Connections
        skips = []
        for block in self.downs:
            x, skip = block(x, t_emb, c_emb)
            skips.append(skip)

        # Bottleneck
        x = self.mid_res1(x, t_emb, c_emb)
        # 混合注意力：自注意力与交叉注意力的加权融合
        out_self = self.mid_attn_self(x)
        out_cross = self.mid_attn_cross(x, c_ctx)
        x = (1 - alpha) * out_self + alpha * out_cross
        x = self.mid_res2(x, t_emb, c_emb)

        # Decoder
        for block in self.ups:
            skip = skips.pop()
            x = block(x, skip, t_emb, c_emb)

        return self.final_conv(x)