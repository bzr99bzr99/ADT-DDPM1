import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from Config import Config

import torch
import math


def get_cosine_schedule_betas(timesteps, s=0.008):
    """
    标准的 Cosine Schedule (Nichol & Dhariwal, 2021)

    参数:
    timesteps: 总的时间步数 T
    s: 偏移量 (shift)，用于防止 t=0 时 beta 太小导致数值不稳定。默认 0.008。

    返回:
    betas: 形状为 (timesteps,) 的 tensor
    """
    # 1. 生成从 0 到 T 的时间步，一共 T+1 个点
    # 因为我们需要利用 t-1 时刻来计算 t 时刻的 beta，所以多算一个点
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)

    # 2. 计算 Alpha_bar (累积信号保留率) 的余弦曲线
    # f(t) = cos^2( ((t/T + s) / (1+s)) * pi/2 )
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2

    # 确保 t=0 时值为 1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # 3. 根据公式反推 Beta
    # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # 4. 裁剪数值
    # 论文建议将 beta 上限裁剪到 0.999，防止 t=T 附近出现数值奇点
    return torch.clip(betas, 0, 0.999).float()


# 扩散过程工具类
class GaussianDiffusion:
    def __init__(self):
        # 创建方差调度
        self.timesteps = Config.timesteps

        # 使用修正后的 Cosine Schedule
        self.betas = get_cosine_schedule_betas(self.timesteps).to(Config.device)

        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

    # 前向扩散过程
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t].reshape(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    # --- 新增的方法：预测原始信号x_0 ---
    def predict_start_from_noise(self, x_t, t, noise):
        """
        根据x_t（带噪声信号）、时间步t和预测的噪声，反推预测的原始干净信号x_0。
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(x_t.device).reshape(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(x_t.device).reshape(-1, 1, 1)

        # 应用公式计算x_0_pred
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t

        return x_0_pred


def samplefft(model, num_samples=8, c=None, fft=None):
    """反向扩散采样过程 with 进度条显示"""
    diffusion = GaussianDiffusion()
    with torch.no_grad():
        # 优化：直接在目标 device 上生成噪声，避免 CPU->GPU 传输
        x_t = torch.randn((num_samples, Config.dim, Config.seq_len), device=Config.device)

        # 使用 tqdm 显示进度
        for t in tqdm(reversed(range(Config.timesteps)), total=Config.timesteps, desc='生成进度'):
            t_batch = torch.full((num_samples,), t, device=Config.device).long()

            # 预测噪声
            if c is None:
                pred_noise = model(x_t, t_batch)
            else:
                pred_noise = model(x_t, t_batch, c, fft)

            # 计算系数
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alpha_bars[t]
            beta_t = diffusion.betas[t]

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            # 反向过程计算
            x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise

    return x_t


def sample(model, num_samples=8, c=None):
    """反向扩散采样过程 with 进度条显示"""
    diffusion = GaussianDiffusion()
    with torch.no_grad():
        # 优化：直接在目标 device 上生成噪声，避免 CPU->GPU 传输
        x_t = torch.randn((num_samples, Config.dim, Config.seq_len), device=Config.device)

        # 使用 tqdm 显示进度
        for t in tqdm(reversed(range(Config.timesteps)), total=Config.timesteps, desc='生成进度'):
            t_batch = torch.full((num_samples,), t, device=Config.device).long()

            # 预测噪声
            if c is None:
                pred_noise = model(x_t, t_batch)
            else:
                pred_noise = model(x_t, t_batch, c)

            # 计算系数
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alpha_bars[t]
            beta_t = diffusion.betas[t]

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            # 反向过程计算
            x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise

    return x_t