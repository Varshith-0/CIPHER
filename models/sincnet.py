import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv1d(nn.Module):
    """
    Sinc-based 1D Convolutional Neural Network layer.
    Allows learning of band-pass filters from raw signals.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=256, in_channels=1, min_low_hz=1.0, min_band_hz=1.0):
        super(SincConv1d, self).__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.in_channels = in_channels
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize frequencies similarly to Mel-scale (can be linear here as well)
        hz = np.linspace(min_low_hz, sample_rate / 2 - min_band_hz, out_channels + 1)
        
        # Frequencies are divided by sample_rate/2 (normalized to [0, 1])
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        # Hamming window
        n_lin = torch.linspace(0, (kernel_size/2)-1, steps=int((kernel_size/2)))
        self.window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        
        n = (kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate

    def forward(self, waveforms):
        """
        waveforms: `(batch_size, in_channels, time)`
        Returns: `(batch_size, out_channels * in_channels, time)`
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window = self.window.to(waveforms.device)
        
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        
        # Prepare filters for grouped convolution
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        filters = filters.repeat(self.in_channels, 1, 1)
        
        return F.conv1d(waveforms, filters, stride=1, padding=self.kernel_size // 2, groups=self.in_channels)
        
class MultiresolutionSpectralFrontend(nn.Module):
    """
    Multiresolution Frontend fusing SincNet features + Raw CNN features + Wavelet-like scattering.
    """
    def __init__(self, in_channels, d_model=256, sample_rate=256, sinc_channels=32):
        super().__init__()
        self.sinc_conv = SincConv1d(
            out_channels=sinc_channels, 
            kernel_size=31, 
            sample_rate=sample_rate, 
            in_channels=in_channels,
            min_low_hz=0.5,
            min_band_hz=2.0
        )
        # Sinc output will have `in_channels * sinc_channels` channels
        self.sinc_min_low_hz = 0.5
        self.sinc_min_band_hz = 2.0
        self.sinc_conv.min_low_hz = 0.5
        self.sinc_conv.min_band_hz = 2.0
        
        # Standard convolution branch to capture non-parametric local features
        self.std_conv = nn.Conv1d(in_channels, in_channels * 16, kernel_size=15, padding=7, groups=in_channels)
        
        self.proj = nn.Sequential(
            nn.Linear(in_channels * sinc_channels + in_channels * 16, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: (Batch, Seq_len, Channels)
        """
        x_t = x.transpose(1, 2) # (Batch, Channels, Seq_len)
        
        sinc_feats = self.sinc_conv(x_t)
        sinc_feats = F.gelu(sinc_feats)
        
        std_feats = self.std_conv(x_t)
        std_feats = F.gelu(std_feats)
        
        combined = torch.cat([sinc_feats, std_feats], dim=1)
        combined = combined.transpose(1, 2) # (Batch, Seq_len, Features)
        
        out = self.proj(combined)
        return out
