"""Restoration and state supervision, separate from the reusable four fusion losses."""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from .loss import SobelGradient, to_luminance


def region_means(loss_map, mask):
    """Per-image means and presence flags; zero-area regions stay finite."""
    if loss_map.shape != mask.shape:
        raise ValueError("Regional loss maps and masks must have matching B1HW shapes.")
    area = mask.flatten(1).sum(1)
    means = (loss_map*mask).flatten(1).sum(1)/area.clamp_min(1e-8)
    return means, area > 0


def masked_mean(loss_map, mask):
    """Average over images that contain this region, excluding absent regions."""
    means, present = region_means(loss_map, mask)
    return (means*present).sum()/present.sum().clamp_min(1)


def balanced_mean(loss_map, mask):
    """Equal weight for damaged/intact regions within each image, then batch."""
    damaged, has_damaged = region_means(loss_map, mask)
    intact, has_intact = region_means(loss_map, 1-mask)
    count = has_damaged.float()+has_intact.float()
    return ((damaged*has_damaged+intact*has_intact)/count.clamp_min(1)).mean()


class AuxiliaryLoss(nn.Module):
    def __init__(self, reconstruction_weight=5.0, keep_weight=2.0, demand_weight=1.0,
                 error_weight=1.0, demand_scale=0.1, average_kernel=5,
                 gradient_weight=0.25, sobel_divisor=8.0, error_scale=0.1):
        super().__init__()
        self.weights = {"reconstruction":reconstruction_weight,"keep":keep_weight,"demand":demand_weight,"error":error_weight}
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError("Auxiliary weights must be finite and non-negative.")
        if (not all(math.isfinite(v) for v in (demand_scale, error_scale, gradient_weight, sobel_divisor))
                or demand_scale <= 0 or error_scale <= 0 or sobel_divisor <= 0
                or average_kernel < 1 or average_kernel % 2 == 0 or gradient_weight < 0):
            raise ValueError("Invalid auxiliary target parameters.")
        self.demand_scale,self.average_kernel = demand_scale,average_kernel
        self.error_scale = error_scale
        self.gradient_weight,self.sobel_divisor = gradient_weight,sobel_divisor
        self.sobel = SobelGradient()

    def error_map(self, image, reference):
        if image.shape != reference.shape:
            raise ValueError("Error targets require same-modality matching shapes.")
        image,reference = image.float(),reference.float()
        intensity = (image-reference).abs().mean(1,keepdim=True)
        gradient = (self.sobel(to_luminance(image))-self.sobel(to_luminance(reference))).abs()/self.sobel_divisor
        pad = self.average_kernel//2
        def pool(x):
            return F.avg_pool2d(F.pad(x,(pad,)*4,mode="replicate"),self.average_kernel,1)
        return pool(intensity)+self.gradient_weight*pool(gradient)

    def forward(self, outputs, batch):
        device = outputs["restored"]["visible"].device
        context = torch.autocast(device.type,enabled=False) if device.type in ("cpu","cuda") else nullcontext()
        with context:
            terms = {key:[] for key in self.weights}
            for modality in ("visible","infrared"):
                restored = outputs["restored"][modality].float()
                source,reference = batch[modality].float(),batch[f"hq_{modality}"].float()
                if restored.shape != reference.shape or source.shape != reference.shape:
                    raise ValueError("Restoration supervision must match source channel and spatial shape.")
                mask = batch[f"mask_{modality}"].float()
                if mask.shape != restored[:, :1].shape or not torch.all((mask >= 0)&(mask <= 1)):
                    raise ValueError("Degradation masks must be B1HW in [0,1].")
                valid = 1-mask
                pixel_error = ((restored-reference).square()+1e-6).sqrt().mean(1,keepdim=True)
                gradient = (self.sobel(to_luminance(restored))-self.sobel(to_luminance(reference))).abs().mean()/self.sobel_divisor
                rec = balanced_mean(pixel_error, mask)+self.gradient_weight*gradient
                differences = (restored-source).abs().mean(1,keepdim=True)
                keep = masked_mean(differences, valid)
                with torch.no_grad():
                    demand_target = (self.error_map(source,reference)/self.demand_scale).clamp(0,1)
                    error_target = self.error_map(restored.detach(),reference).detach()
                terms["reconstruction"].append(rec)
                terms["keep"].append(keep)
                demand_map = F.binary_cross_entropy_with_logits(
                    outputs["demand_logits"][modality].float(), demand_target, reduction="none")
                # Supervise the unbounded quantity: negative raw errors still
                # receive a recovery gradient even when the display ReLU is zero.
                error_map = F.smooth_l1_loss(outputs["error_raw"][modality].float(),
                                           error_target/self.error_scale, reduction="none")
                terms["demand"].append(balanced_mean(demand_map, mask))
                terms["error"].append(balanced_mean(error_map, mask))
            terms = {k:torch.stack(v).mean() for k,v in terms.items()}
            total = sum(self.weights[k]*v for k,v in terms.items())
        return total,terms
