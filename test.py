"""Fuse image pairs; normal output contains only rgb/ and gray/."""
import sys
from types import SimpleNamespace
from itertools import islice

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.fusion import MODEL_VERSION, RobustFusionNet
from utils.dataset import PairedFusionDataset
from utils.experiment import choose_device, move_batch, read_checkpoint, resolve_path, strict_load
from utils.image import save_image
from utils.loss import to_luminance

# ===================== EDIT HERE =====================
VISIBLE_DIR = "data/test_MSRS/vi_snow/2/vi"
INFRARED_DIR = "data/test_MSRS/ir"
CHECKPOINT_PATH = "ckpt/latest.pth"
OUTPUT_DIR = "results/test_vi_snow"
# =====================================================


def _build_settings():
    return SimpleNamespace(checkpoint=CHECKPOINT_PATH, visible_dir=VISIBLE_DIR,
        infrared_dir=INFRARED_DIR, output=OUTPUT_DIR,
        device="auto", cpu_threads=4, limit=None, progress=True)


@torch.no_grad()
def run(args):
    if args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Thread count and image limit must be positive.")
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    state = read_checkpoint(args.checkpoint, expected_implementation=MODEL_VERSION)
    model = RobustFusionNet(state["config"]["model"]).to(device)
    strict_load(model, state["model"])
    model.eval()
    dataset = PairedFusionDataset(str(resolve_path(args.visible_dir)), str(resolve_path(args.infrared_dir)), mode="inference")
    output = resolve_path(args.output)
    for name in ("rgb", "gray"):
        (output/name).mkdir(parents=True, exist_ok=True)
    print(f"Loaded epoch={state['epoch']}, device={device}, pairs={len(dataset)}")
    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    progress = tqdm(islice(loader, total), total=total, desc="Fusing", unit="image",
                    dynamic_ncols=True, disable=not args.progress)
    count = 0
    try:
        for batch in progress:
            batch = move_batch(batch, device)
            sample = batch["sample_id"][0]
            fused = model(batch["visible"], batch["infrared"], return_aux=False)
            save_image(fused, output/"rgb"/(sample+".png"))
            save_image(to_luminance(fused), output/"gray"/(sample+".png"))
            count += 1
            progress.set_postfix(sample=sample)
    finally:
        progress.close()
    print(f"Saved {count} image pairs: {output/'rgb'}, {output/'gray'}")
    return count


def main():
    run(_build_settings())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit test_MSRS.py, then run: python test_MSRS.py")
    main()
