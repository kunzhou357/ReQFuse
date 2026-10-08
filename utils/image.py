"""Image IO; float CHW tensors in [0, 1] everywhere inside the project."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def read_image(path, channels=3):
    if channels not in (1,3):
        raise ValueError("read_image supports RGB (3) or grayscale (1).")
    with Image.open(path) as image:
        array = np.array(image.convert("RGB" if channels == 3 else "L"), dtype=np.float32, copy=True) / 255.0
    if channels == 1:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(image, path):
    image = image.detach().float().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("save_image expects one image, not a batch.")
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("save_image expects CHW RGB or gray.")
    array = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    if image.shape[0] == 1:
        array = array[..., 0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
