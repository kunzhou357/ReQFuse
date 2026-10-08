"""Post-restoration error estimation and quality-controlled cross-modality exchange.

Cross-attention follows Restormer/DSPFusion channel attention and AMG-Fuse's
convolutional projections. Source K/V, rather than recipient residuals, are
quality-gated. Local windows constrain relationship statistics, not alignment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import LayerNorm2d, LightConvResidualBlock, StateHead, no_autocast, statistics_dtype


class PostRestorationErrorHead(nn.Module):
    """Regress normalized error before any nonnegative display/gating transform."""
    def __init__(self, image_channels, feature_channels, hidden,
                 initial_error=0.05, error_scale=0.1):
        super().__init__()
        if not math.isfinite(initial_error) or initial_error <= 0:
            raise ValueError("initial_error must be positive.")
        if not math.isfinite(error_scale) or error_scale <= 0:
            raise ValueError("error_scale must be positive.")
        self.error_scale = error_scale
        self.feature_norm = LayerNorm2d(feature_channels)
        self.head = StateHead(2 * image_channels + 1 + feature_channels, hidden)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, initial_error/error_scale)

    def forward(self, restored, modification, demand, feature):
        # Detach upstream tensors BEFORE this head's learnable normalization.
        feature = self.feature_norm(feature.detach())
        features = torch.cat((restored.detach(), modification.detach(), demand.detach(), feature), 1)
        logits = self.head(features)
        with no_autocast(logits):
            raw = statistics_dtype(logits)
            error = self.error_scale * raw.relu()
        return {"raw": raw, "error": error}


def quality_from_error(error, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Quality temperature must be finite and positive.")
    with no_autocast(error):
        return torch.exp(-statistics_dtype(error) / temperature)


class WindowChannelCrossAttention(nn.Module):
    def __init__(self, channels, heads, window_size):
        super().__init__()
        if heads < 1 or channels % heads or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("Invalid channel heads or window size.")
        self.heads, self.window_size = heads, window_size
        self.norm_target, self.norm_source = LayerNorm2d(channels), LayerNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.q_dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.kv = nn.Conv2d(channels, 2*channels, 1, bias=False)
        self.kv_dwconv = nn.Conv2d(2*channels, 2*channels, 3, padding=1, groups=2*channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def _partition(self, x, wh, ww):
        b, c, h, w = x.shape
        return x.reshape(b, c, h//wh, wh, w//ww, ww).permute(0, 2, 4, 1, 3, 5).reshape(-1, self.heads, c//self.heads, wh*ww)

    def forward(self, target, source, source_quality):
        if target.shape != source.shape or source_quality.shape != (source.shape[0], 1, *source.shape[-2:]):
            raise ValueError("Cross-attention requires equal source/target and B1HW quality.")
        b, c, h, w = target.shape
        q = self.q_dwconv(self.q(self.norm_target(target)))
        k, v = self.kv_dwconv(self.kv(self.norm_source(source))).chunk(2, 1)
        dtype = v.dtype
        # Gate after normalization/projection; no bias can resurrect a zero donor.
        k, v = k * source_quality.detach(), v * source_quality.detach()
        wh = ww = self.window_size
        ph, pw = (-h) % wh, (-w) % ww
        q, k, v = (F.pad(z, (0, pw, 0, ph)) for z in (q, k, v))
        hp, wp = h+ph, w+pw
        q, k, v = (self._partition(z, wh, ww) for z in (q, k, v))
        with no_autocast(q):
            q, k, v = (statistics_dtype(z) for z in (q, k, v))
            q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
            attention = (q @ k.transpose(-2, -1) * self.temperature.to(q.dtype)).softmax(-1)
            out = (attention @ v).reshape(b, hp//wh, wp//ww, c, wh, ww)
            out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, hp, wp)[..., :h, :w]
        return self.project_out(out.to(dtype))


class QualityMerge(nn.Module):
    def __init__(self, channels, with_messages):
        super().__init__()
        self.with_messages = with_messages
        self.project = nn.Conv2d((4 if with_messages else 2)*channels, channels, 1, bias=False)
        self.refine = LightConvResidualBlock(channels)

    def forward(self, visible, infrared, q_visible, q_infrared, messages=()):
        if len(messages) != (2 if self.with_messages else 0):
            raise ValueError("QualityMerge message count does not match its construction.")
        groups = (q_visible * visible, q_infrared * infrared, *messages)
        return self.refine(self.project(torch.cat(groups, 1)))
