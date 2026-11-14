# SPDX-License-Identifier: MIT
"""Owned pure-torch models for PLAN2-06 broader admission (no model packages).

A diffusion-style noise-prediction UNet2D (conv + GroupNorm + non-causal SDPA
attention + up/down-sampling + skips) and next-token RNN/LSTM/GRU language
models. Random-initialized from config; no weights fetched; no diffusers. These
exercise the native conv (MIOpen), attention (SDPA) and recurrent kernels.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(tokens, tokens, tokens)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class UNet2D(nn.Module):
    """Diffusion-style noise-prediction UNet (32x32, 3 channels)."""

    def __init__(self, in_channels=3, base=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down1 = ResBlock(base, base)
        self.downsample1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.down2 = ResBlock(base, base * 2)
        self.downsample2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.mid1 = ResBlock(base * 2, base * 2)
        self.mid_attn = SpatialAttention(base * 2)
        self.mid2 = ResBlock(base * 2, base * 2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = ResBlock(base * 2 + base * 2, base * 2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ResBlock(base * 2 + base, base)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        d1 = self.down1(x0)
        d2 = self.down2(self.downsample1(d1))
        m = self.mid1(self.downsample2(d2))
        m = self.mid2(self.mid_attn(m))
        u2 = self.up2(torch.cat([self.upsample2(m), d2], dim=1))
        u1 = self.up1(torch.cat([self.upsample1(u2), d1], dim=1))
        return self.out_conv(F.silu(self.out_norm(u1)))


class RecurrentLM(nn.Module):
    """Next-token language model with a torch-native recurrent cell."""

    def __init__(self, cell, vocab=64, emb=16, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.rnn = getattr(nn, cell)(emb, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
        self.cell = cell

    def forward(self, ids):
        out, _ = self.rnn(self.embed(ids))
        return self.head(out)


def unet_inputs(device, dtype, batch=2, size=32, channels=3):
    generator = torch.Generator(device=device).manual_seed(20261114)
    x = torch.randn(
        batch, channels, size, size, device=device, dtype=dtype, generator=generator
    )
    target = torch.randn_like(x, generator=generator)
    return x, target


def unet_loss(model, inputs):
    x, target = inputs
    return (model(x) - target).square().mean()


def lm_inputs(device, batch=2, length=16, vocab=64):
    generator = torch.Generator(device=device).manual_seed(20261114)
    return torch.randint(0, vocab, (batch, length), device=device, generator=generator)


def lm_loss(model, ids):
    logits = model(ids)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    )
