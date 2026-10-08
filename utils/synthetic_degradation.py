"""Shared online/offline RGB/thermal corruption operators and severity presets.

Arrays use HWC RGB or HW grayscale, float32 in [0, 1]. Original public-dataset
images are the clean/HQ references. LocalDegrader samples these operators in
dataset workers; generate_degradation.py freezes them for reproducible tests.

Sources: DSPFusion appendix B.1; ControlFusion section 3 and its dataset
generation script. The three-level partition, snow particles and sparse
weather alpha blending are implementation choices, not an exact reproduction
of AWMM-100K.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


PROTOCOL_VERSION = "paper_presets_v1"
LEVELS = ("light", "medium", "heavy")
VISIBLE_TYPES = ("low_light", "noise", "blur", "rain", "snow", "haze")
INFRARED_TYPES = ("noise", "stripe", "low_contrast")

# 8-bit noise units; conversion to [0,1] happens in the operators below.
# Noise sigma=10 and blur sigma=2.6 are from DSPFusion; noise [5,20],
# haze beta [0.5,2], illumination gamma [0.5,3] are from ControlFusion.
# Medium stripe sigma=6 is within the official script's [5,7].
PRESETS = {
    "noise": {
        "light": {"sigma_255": 5.0, "poisson_peak": 120.0},
        "medium": {"sigma_255": 10.0, "poisson_peak": 80.0},
        "heavy": {"sigma_255": 20.0, "poisson_peak": 50.0},
    },
    "blur": {
        "light": {"kernel_size": 21, "sigma": 1.2},
        "medium": {"kernel_size": 21, "sigma": 2.0},
        "heavy": {"kernel_size": 21, "sigma": 2.6},
    },
    "low_light": {
        level: {"gamma": gamma, "poisson_peak": 1000.0, "read_std": 0.0005}
        for level, gamma in zip(LEVELS, (1.5, 2.0, 3.0))
    },
    "stripe": {
        level: {"sigma_255": sigma}
        for level, sigma in zip(LEVELS, (3.0, 6.0, 10.0))
    },
    "low_contrast": {
        level: {"factor": factor}
        for level, factor in zip(LEVELS, (0.8, 0.5, 0.3))
    },
    "haze": {
        level: {"beta": beta, "airlight": 0.75, "distance_cap": 0.7}
        for level, beta in zip(LEVELS, (0.5, 1.0, 2.0))
    },
    "rain": {
        level: {"coverage": coverage, "opacity": opacity, "length": [12, 30],
                "width": [1, 2], "angle_degrees": [-20.0, 20.0]}
        for level, coverage, opacity in zip(LEVELS, (0.02, 0.05, 0.08), (0.15, 0.30, 0.45))
    },
    "snow": {
        level: {"coverage": coverage, "opacity": opacity, "radius": [1, 6]}
        for level, coverage, opacity in zip(LEVELS, (0.03, 0.06, 0.10), (0.50, 0.70, 0.85))
    },
}


@dataclass
class DegradationResult:
    image: np.ndarray
    mask: np.ndarray       # Binary damaged-region support; supervises the losses.
    envelope: np.ndarray   # Soft region; distinct from sparse rain/snow support.
    parameters: dict


def _smooth_step(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def local_envelope(shape, rng, stripe=False, area=0.4):
    """A repeatable soft region; stripe corruption stays column-correlated."""
    h, w = shape
    if h < 1 or w < 1 or not 0 < area < 1:
        raise ValueError("Invalid image shape or local area.")
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    if stripe:
        width = max(1, round(w * area))
        left = int(rng.integers(0, w - width + 1))
        distance = np.minimum(xx - (left - 0.5), (left + width - 0.5) - xx)
        return _smooth_step(distance / max(1.0, w * 0.04) + 0.5).astype(np.float32)
    score = np.full((h, w), -np.inf, dtype=np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        rx, ry = rng.uniform(0.20, 0.45, 2)
        radius = np.sqrt(((xx / max(w, 1) - cx) / rx) ** 2 +
                         ((yy / max(h, 1) - cy) / ry) ** 2)
        score = np.maximum(score, 1.0 - radius)
    threshold = np.quantile(score, 1.0 - area)
    return _smooth_step((score - threshold) / 0.16 + 0.5).astype(np.float32)


def _srgb_to_linear(image):
    return np.where(image <= 0.04045, image / 12.92,
                    ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(image):
    image = np.clip(image, 0.0, 1.0)
    return np.where(image <= 0.0031308, image * 12.92,
                    1.055 * image ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _put_sprite(canvas, sprite, x, y):
    h, w = canvas.shape
    sh, sw = sprite.shape
    left, top = x - sw // 2, y - sh // 2
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left + sw), min(h, top + sh)
    if x1 > x0 and y1 > y0:
        region = canvas[y0:y1, x0:x1]
        np.maximum(region, sprite[y0-top:y1-top, x0-left:x1-left], out=region)


def _weather_layer(shape, kind, params, rng, envelope):
    """Draw neutral streaks/flakes until their measured coverage is reached."""
    h, w = shape
    region = envelope >= 0.5
    if not region.any():
        region = envelope > 0
    canvas = np.zeros((h, w), dtype=np.float32)
    sprites = []
    if kind == "rain":
        for _ in range(12):
            length = int(rng.integers(params["length"][0], params["length"][1] + 1))
            width = int(rng.integers(params["width"][0], params["width"][1] + 1))
            angle = math.radians(float(rng.uniform(*params["angle_degrees"])))
            size = length + 5
            sprite = np.zeros((size, size), dtype=np.uint8)
            c, dx, dy = size // 2, math.sin(angle) * length / 2, math.cos(angle) * length / 2
            cv2.line(sprite, (round(c-dx), round(c-dy)), (round(c+dx), round(c+dy)),
                     255, width, cv2.LINE_AA)
            sprites.append(sprite.astype(np.float32) / 255.0)
    else:
        for radius in range(params["radius"][0], params["radius"][1] + 1):
            axis = np.arange(-radius, radius + 1, dtype=np.float32)
            radius_sq = (axis[:, None] ** 2 + axis[None, :] ** 2) / max(radius ** 2, 1)
            sprite = np.maximum(0.0, (np.exp(-3.0 * radius_sq) - math.exp(-3.0)) /
                                (1.0 - math.exp(-3.0)))
            sprites.append(sprite.astype(np.float32))
    threshold = 0.02 / params["opacity"]
    target = max(1, math.ceil(int(region.sum()) * params["coverage"]))
    support = 0
    particles = 0
    # Coverage is measured inside the selected region, not over the entire image.
    region_y, region_x = np.nonzero(region)
    while support < target:
        for _ in range(min(8, target - support)):
            position = int(rng.integers(0, len(region_x)))
            sprite = sprites[int(rng.integers(0, len(sprites)))]
            _put_sprite(canvas, sprite, int(region_x[position]), int(region_y[position]))
            particles += 1
        support = int(np.count_nonzero((canvas * envelope > threshold) & region))
        if particles > max(1000, 10 * h * w):
            raise RuntimeError("Could not construct the requested weather coverage.")
    return canvas, {"actual_coverage": support / int(region.sum()), "particles": particles}


class SyntheticDegrader:
    """One explicit corruption, with deterministic RNG supplied by the caller."""

    def __call__(self, image, kind, level, rng, *, modality="visible",
                 global_degradation=True, envelope=None, depth=None, depth_is_inverse=True):
        image = np.asarray(image, dtype=np.float32)
        if modality not in ("visible", "infrared"):
            raise ValueError("modality must be visible or infrared.")
        allowed = VISIBLE_TYPES if modality == "visible" else INFRARED_TYPES
        if kind not in ("clean", *allowed) or level not in LEVELS:
            raise ValueError(f"Unsupported {modality} degradation/level: {kind}/{level}")
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] != 3):
            raise ValueError("Expected HW gray or HWC RGB image.")
        if (modality == "visible") != (image.ndim == 3):
            raise ValueError("Visible inputs must be RGB; infrared inputs must be grayscale.")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Images must be finite and in [0,1].")
        h, w = image.shape[:2]
        if kind == "clean":
            zero = np.zeros((h, w), dtype=np.float32)
            return DegradationResult(image.copy(), zero, zero.copy(), {})
        if not isinstance(global_degradation, bool) or not isinstance(depth_is_inverse, bool):
            raise TypeError("Binary options must be True or False.")
        if envelope is None:
            envelope = (np.ones((h, w), dtype=np.float32) if global_degradation else
                        local_envelope((h, w), rng, stripe=(kind == "stripe")))
        envelope = np.asarray(envelope, dtype=np.float32)
        if envelope.shape != (h, w) or not np.isfinite(envelope).all() or \
                envelope.min() < 0 or envelope.max() > 1 or not envelope.any():
            raise ValueError("Envelope must be a nonempty HW array in [0,1].")
        if kind == "stripe" and not np.all(envelope == envelope[:1, :]):
            raise ValueError("Stripe envelope must be constant down each column.")
        params = dict(PRESETS[kind][level])
        params["envelope_area"] = float(np.mean(envelope >= 0.5))
        weight = envelope[..., None] if image.ndim == 3 else envelope
        support = envelope > 0
        if kind == "noise":
            base = np.clip(image + rng.normal(0.0, params["sigma_255"] / 255.0, image.shape), 0, 1)
            degraded = (rng.poisson(base * params["poisson_peak"]) / params["poisson_peak"]).astype(np.float32)
        elif kind == "blur":
            size = params["kernel_size"]
            degraded = cv2.GaussianBlur(image, (size, size), params["sigma"],
                                        borderType=cv2.BORDER_REFLECT_101)
        elif kind == "low_contrast":
            mean = float(image.mean())
            params["reference_mean"] = mean
            degraded = mean + params["factor"] * (image - mean)
        elif kind == "stripe":
            columns = rng.normal(0.0, params["sigma_255"] / 255.0, (1, w)).astype(np.float32)
            degraded = image + columns
        elif kind == "low_light":
            # Retinex-style illumination attenuation: I/L * L^gamma.
            # Smooth max-RGB is a lightweight illumination estimate, not LIME.
            illumination = cv2.GaussianBlur(image.max(axis=2), (0, 0), 5.0,
                                             borderType=cv2.BORDER_REFLECT_101)
            gain = np.maximum(illumination, 1e-4) ** (params["gamma"] - 1.0)
            linear = _srgb_to_linear(image)
            signal = _srgb_to_linear(image * gain[..., None])
            noisy = rng.poisson(signal * params["poisson_peak"]) / params["poisson_peak"]
            noisy += rng.normal(0.0, params["read_std"], image.shape)
            result = _linear_to_srgb(linear + weight * (np.clip(noisy, 0, 1) - linear))
            params["illumination_estimator"] = "smooth_max_rgb_sigma5"
            params["mean_gain"] = float(gain.mean())
        elif kind == "haze":
            if depth is None:
                distance = np.full((h, w), params["distance_cap"], dtype=np.float32)
                params["depth_mode"] = "uniform_distance"
            else:
                distance = np.asarray(depth, dtype=np.float32)
                if distance.shape != (h, w) or not np.isfinite(distance).all():
                    raise ValueError("Depth must be a finite HW array matching the image.")
                if distance.min() < 0 or distance.max() > 1:
                    span = float(np.ptp(distance))
                    if span <= 1e-8:
                        raise ValueError("Unnormalised constant depth cannot be interpreted.")
                    distance = (distance - distance.min()) / span
                distance = cv2.blur(distance, (22, 22), borderType=cv2.BORDER_REFLECT_101)
                if depth_is_inverse:
                    distance = 1.0 - distance
                distance = np.clip(distance, 0, params["distance_cap"])
                params["depth_mode"] = "inverse_depth_map" if depth_is_inverse else "distance_map"
            transmission = np.exp(-params["beta"] * envelope * distance)
            result = image * transmission[..., None] + params["airlight"] * (1-transmission[..., None])
            params["mean_transmission"] = float(transmission.mean())
        else:
            layer, measured = _weather_layer((h, w), kind, params, rng, envelope)
            alpha = layer * params["opacity"] * envelope
            result = image * (1-alpha[..., None]) + 0.9 * alpha[..., None]
            support = alpha > 0
            params.update(measured)
        if kind not in ("low_light", "haze", "rain", "snow"):
            result = image + weight * (np.clip(degraded, 0, 1) - image)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        # Preserve complete-region pixels exactly, including nonlinear RGB round trips.
        result[~support] = image[~support]
        params["support_area"] = float(np.mean(support))
        return DegradationResult(result, support.astype(np.float32), envelope.copy(), params)
