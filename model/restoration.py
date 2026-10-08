"""Modification-demand-controlled restoration for a single modality."""
import math

import torch
from torch import nn

from .blocks import (Downsample, LayerNorm2d, StateHead, TransformerStage,
                     Upsample, no_autocast, statistics_dtype, validate_pyramid)


class SelectiveRestormer(nn.Module):
    def __init__(self, in_channels, widths, heads, encoder_blocks, decoder_blocks,
                 refinement_blocks, expansion, state_hidden, residual_bound,
                 checkpoint_blocks=False):
        super().__init__()
        validate_pyramid(widths, heads, encoder_blocks, 3)
        if len(decoder_blocks) != 2 or any(n < 1 for n in decoder_blocks):
            raise ValueError("Restorer decoder_blocks must be [coarse, fine] positive counts.")
        if in_channels not in (1, 3) or not math.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("Restorer requires RGB/gray input and a positive residual bound.")
        self.residual_bound = residual_bound
        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False)
        self.encoder = nn.ModuleList(TransformerStage(c, h, n, expansion, checkpoint_blocks)
                                     for c, h, n in zip(widths, heads, encoder_blocks))
        self.down = nn.ModuleList(Downsample(c) for c in widths[:-1])
        self.up = nn.ModuleList(Upsample(c) for c in reversed(widths[1:]))
        self.reduce = nn.ModuleList(nn.Conv2d(2*c, c, 1, bias=False) for c in reversed(widths[:-1]))
        self.decoder = nn.ModuleList(TransformerStage(widths[s], heads[s], n, expansion, checkpoint_blocks)
                                    for s, n in zip((1, 0), decoder_blocks))
        self.refinement = TransformerStage(widths[0], heads[0], refinement_blocks, expansion, checkpoint_blocks)
        # Bound the feature scale shared by the demand, residual and error heads.
        # Input RGB/IR values retain their original brightness/color scale.
        self.head_norm = LayerNorm2d(widths[0])
        self.demand_head = StateHead(widths[0] + in_channels, state_hidden)
        self.residual_head = nn.Conv2d(widths[0], in_channels, 3, padding=1)
        nn.init.zeros_(self.demand_head[-1].weight)
        nn.init.zeros_(self.demand_head[-1].bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, image, demand_strength=1.0):
        x, skips = self.stem(image), []
        for s, stage in enumerate(self.encoder):
            x = stage(x)
            skips.append(x)
            if s < 2:
                x = self.down[s](x)
        for j, s in enumerate((1, 0)):
            x = self.decoder[j](self.reduce[j](torch.cat((self.up[j](x), skips[s]), 1)))
        feature = self.head_norm(self.refinement(x))
        demand_logits = self.demand_head(torch.cat((feature, image), 1))
        residual_logits = self.residual_head(feature)
        with no_autocast(demand_logits):
            demand = statistics_dtype(demand_logits).sigmoid()
            residual = self.residual_bound * statistics_dtype(residual_logits).tanh()
        # The gate is detached: reconstruction/fusion cannot reduce their loss
        # by closing it; only the auxiliary demand supervision moves it.
        applied = (1-demand_strength) + demand_strength*demand.detach()
        restored = image + applied * residual
        return {"restored": restored, "demand": demand,
                "demand_logits": demand_logits, "feature": feature}
