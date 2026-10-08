"""Online sampling of the same nine operators used by the offline generator.

Each affected modality receives one operator; low-light includes shot/read
noise. Masks supervise training and are never network inputs.
"""
import hashlib
import json
import math

import numpy as np
import torch

from .synthetic_degradation import (
    INFRARED_TYPES, LEVELS, PRESETS, PROTOCOL_VERSION, VISIBLE_TYPES,
    SyntheticDegrader,
)

DEFAULT_DEGRADATION = {
    "pair_probabilities": [0.2, 0.3, 0.3, 0.2],  # clean, VI only, IR only, both
    "local_probability": 0.7,
    "visible_types": list(VISIBLE_TYPES),
    "infrared_types": list(INFRARED_TYPES),
    "levels": list(LEVELS),
    "level_probabilities": [0.25, 0.5, 0.25],
    "context_margin": 16,
    "protocol_version": PROTOCOL_VERSION,
    "preset_signature": hashlib.sha256(json.dumps(PRESETS, sort_keys=True).encode()).hexdigest(),
}


def uniform(generator, low=0.0, high=1.0):
    return low + (high-low)*torch.rand((), generator=generator).item()


def randint(generator, low, high):
    return int(torch.randint(low, high, (), generator=generator))


def _choice(probabilities, generator):
    draw, cumulative = uniform(generator), 0.0
    for i, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return i
    return len(probabilities)-1


def _probabilities(values, length, name):
    if len(values) != length or any(not math.isfinite(p) or p < 0 for p in values) or abs(sum(values)-1) > 1e-6:
        raise ValueError(f"{name} must contain {length} probabilities summing to one.")


class LocalDegrader:
    """CPU CHW adapter; operator math and severity presets have one source."""
    def __init__(self, config=None):
        self.config = dict(DEFAULT_DEGRADATION)
        if config:
            unknown = set(config)-set(self.config)
            if unknown:
                raise ValueError(f"Unknown degradation options: {sorted(unknown)}")
            self.config.update(config)
        c = self.config
        _probabilities(c["pair_probabilities"], 4, "pair_probabilities")
        if not math.isfinite(c["local_probability"]) or not 0 <= c["local_probability"] <= 1:
            raise ValueError("local_probability must be in [0,1].")
        for name, allowed in (("visible_types", VISIBLE_TYPES), ("infrared_types", INFRARED_TYPES), ("levels", LEVELS)):
            if not c[name] or len(set(c[name])) != len(c[name]) or set(c[name])-set(allowed):
                raise ValueError(f"Invalid {name}.")
        _probabilities(c["level_probabilities"], len(c["levels"]), "level_probabilities")
        if isinstance(c["context_margin"], bool) or not isinstance(c["context_margin"], int) or c["context_margin"] < 0:
            raise ValueError("context_margin must be a nonnegative integer.")
        for key in ("protocol_version", "preset_signature"):
            if c[key] != DEFAULT_DEGRADATION[key]:
                raise ValueError(f"{key} does not match the shared degradation protocol.")
        self.operator = SyntheticDegrader()

    def _one(self, image, modality, enabled, generator):
        if not enabled:
            return image.clone(), image.new_zeros(1, *image.shape[-2:])
        kinds = self.config[f"{modality}_types"]
        kind = kinds[randint(generator, 0, len(kinds))]
        level = self.config["levels"][_choice(self.config["level_probabilities"], generator)]
        global_degradation = uniform(generator) >= self.config["local_probability"]
        rng = np.random.default_rng(randint(generator, 0, 2**63-1))
        array = image.detach().float().permute(1, 2, 0).numpy()
        if modality == "infrared":
            array = array[..., 0]
        result = self.operator(array, kind, level, rng, modality=modality,
                               global_degradation=global_degradation)
        array = result.image[..., None] if modality == "infrared" else result.image
        degraded = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).to(image.dtype)
        mask = torch.from_numpy(result.mask.copy()).unsqueeze(0).to(image.dtype)
        return degraded, mask

    def __call__(self, visible, infrared, generator):
        if visible.device.type != "cpu" or infrared.device.type != "cpu":
            raise ValueError("Online degradation belongs to CPU dataset workers.")
        if visible.ndim != 3 or infrared.ndim != 3 or visible.shape[0] != 3 or infrared.shape[0] != 1:
            raise ValueError("Expected CHW RGB visible and grayscale infrared tensors.")
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError("Registered VI/IR dimensions must match.")
        case = _choice(self.config["pair_probabilities"], generator)
        vi, mv = self._one(visible, "visible", case in (1, 3), generator)
        ir, mi = self._one(infrared, "infrared", case in (2, 3), generator)
        return vi, ir, mv, mi
