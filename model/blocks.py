"""Restormer building blocks and small convolution blocks.

Architecture references: swz30/Restormer, AMG-Fuse and DSPFusion official code.
This implementation keeps MDTA/GDFN explicit, uses NCHW channel LayerNorm,
and performs attention statistics in float32 under mixed precision.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def no_autocast(x):
    return torch.autocast(x.device.type, enabled=False) if x.device.type in ("cpu", "cuda") else nullcontext()


def statistics_dtype(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def validate_pyramid(widths, heads, blocks, levels):
    if not (len(widths) == len(heads) == len(blocks) == levels):
        raise ValueError(f"Expected {levels} widths, heads and encoder block counts.")
    if any(not isinstance(c, int) or c < 2 or c % 2 for c in widths):
        raise ValueError("Widths must be positive even integers >= 2.")
    if any(widths[s] != 2 * widths[s - 1] for s in range(1, levels)):
        raise ValueError("Pixel-rearrangement pyramids require doubling channel widths.")
    if any(not isinstance(h, int) or h < 1 or c % h for c, h in zip(widths, heads)):
        raise ValueError("Every width must be divisible by its positive head count.")
    if any(not isinstance(n, int) or n < 1 for n in blocks):
        raise ValueError("Every encoder stage must contain at least one block.")


class LayerNorm2d(nn.Module):
    """WithBias LayerNorm over channels at each pixel, as in Restormer."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        with no_autocast(x):
            z = statistics_dtype(x)
            mean = z.mean(1, keepdim=True)
            variance = z.var(1, keepdim=True, unbiased=False)
            z = (z - mean) * torch.rsqrt(variance + self.eps)
            z = z * self.weight.to(z.dtype)[None, :, None, None] + self.bias.to(z.dtype)[None, :, None, None]
        return z.to(x.dtype)


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention: channel, rather than token, attention."""
    def __init__(self, channels, heads):
        super().__init__()
        if heads < 1 or channels % heads:
            raise ValueError("MDTA channels must be divisible by heads.")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(3 * channels, 3 * channels, 3, padding=1, groups=3 * channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = [z.reshape(b, self.heads, c // self.heads, h * w) for z in qkv.chunk(3, 1)]
        with no_autocast(q):
            q, k, v = (statistics_dtype(z) for z in (q, k, v))
            q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
            attention = (q @ k.transpose(-2, -1) * self.temperature.to(q.dtype)).softmax(-1)
            out = (attention @ v).reshape(b, c, h, w)
        return self.project_out(out.to(qkv.dtype))


class GDFN(nn.Module):
    def __init__(self, channels, expansion=2.66):
        super().__init__()
        if not math.isfinite(expansion) or expansion <= 0:
            raise ValueError("FFN expansion must be positive and finite.")
        hidden = int(channels * expansion)
        if hidden < 1:
            raise ValueError("FFN hidden width is zero.")
        self.project_in = nn.Conv2d(channels, 2 * hidden, 1, bias=False)
        self.dwconv = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        a, b = self.dwconv(self.project_in(x)).chunk(2, 1)
        return self.project_out(F.gelu(a) * b)


class RestormerBlock(nn.Module):
    def __init__(self, channels, heads, expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TransformerStage(nn.Sequential):
    def __init__(self, channels, heads, count, expansion=2.66, checkpoint_blocks=False):
        if not isinstance(count, int) or count < 0:
            raise ValueError("Block count must be a non-negative integer.")
        super().__init__(*(RestormerBlock(channels, heads, expansion) for _ in range(count)))
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, x):
        for block in self:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class LightConvResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.body = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False), nn.GELU(),
                                  nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        return x + self.body(x)


class Downsample(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))


class Upsample(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, 2 * channels, 3, padding=1, bias=False), nn.PixelShuffle(2))


class StateHead(nn.Sequential):
    def __init__(self, in_channels, hidden):
        super().__init__(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
                         nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1))


def pad_to_multiple(x, multiple=8):
    # 8 = 2**3 pixel unshuffles: three pyramid downsamples inside the networks.
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if not (ph or pw):
        return x
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode)
