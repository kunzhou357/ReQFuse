"""Epoch-based two-stage training. Edit the short block below and run train_MSRS.py.

Normal output consists only of latest.pth and one checkpoint every 10 epochs.
"""
from copy import deepcopy
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.fusion import DEFAULT_MODEL_CONFIG, MODEL_VERSION, RobustFusionNet
from utils.auxiliary_loss import AuxiliaryLoss
from utils.dataset import EpochBatchSampler, PairedFusionDataset
from utils.degradation import DEFAULT_DEGRADATION
from utils.experiment import (amp_settings, atomic_checkpoint, autocast_context,
    capture_rng, choose_device, dataset_fingerprint, dataset_options,
    epoch_joint_strength, epoch_learning_rate, move_batch, read_checkpoint,
    resolve_path, restore_rng, seed_everything, strict_load, training_signature)
from utils.loss import FusionLoss

# ===================== EDIT HERE =====================
VISIBLE_DIR = "data/train_MSRS/vis"        # any folder of visible images
INFRARED_DIR = "data/train_MSRS/ir"
OUTPUT_DIR = "outputs/reqfuse"

BATCH_SIZE = 4
PATCH_SIZE = 128
RESTORATION_EPOCHS = 40                   # stage 1: restoration and state estimation
FUSION_EPOCHS = 80                        # stage 2: joint restoration and fusion
LEARNING_RATE = 2e-4
RESUME_TRAINING = False                   # True: resume from OUTPUT_DIR/latest.pth

ONLINE_DEGRADATION = True                 # True: degrade the HQ originals on the fly
HQ_VISIBLE_DIR = None                     # required only when ONLINE_DEGRADATION = False
HQ_INFRARED_DIR = None
# =====================================================


def _build_settings():
    if not isinstance(ONLINE_DEGRADATION, bool) or not isinstance(RESUME_TRAINING, bool):
        raise TypeError("ONLINE_DEGRADATION and RESUME_TRAINING must be True or False.")
    if not ONLINE_DEGRADATION and (not HQ_VISIBLE_DIR or not HQ_INFRARED_DIR):
        raise ValueError("ONLINE_DEGRADATION=False requires both HQ directories.")
    device = choose_device()
    precision = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.type == "cuda" else "none"
    return {
        "implementation": {"version": MODEL_VERSION, "training_protocol": "epochs_shared_degradation",
                           "demand_curriculum": "epoch_joint_strength_detached",
                           "demand_supervision": "logits_bce_region_balanced",
                           "error_supervision": "linear_scaled_smooth_l1_region_balanced"},
        "model": deepcopy(DEFAULT_MODEL_CONFIG),
        "data": {"train_MSRS": {
            "mode": "synthetic" if ONLINE_DEGRADATION else "paired",
            "visible_dir": VISIBLE_DIR, "infrared_dir": INFRARED_DIR,
            "hq_visible_dir": None if ONLINE_DEGRADATION else HQ_VISIBLE_DIR,
            "hq_infrared_dir": None if ONLINE_DEGRADATION else HQ_INFRARED_DIR,
            "mask_visible_dir": None, "mask_infrared_dir": None, "patch_size": PATCH_SIZE,
        }},
        "training": {
            "seed": 100, "deterministic": False, "device": str(device), "amp": precision,
            "cpu_threads": 4, "workers": 4 if device.type == "cuda" else 0,
            "batch_size": BATCH_SIZE, "accumulation_steps": 2,
            "restoration_epochs": RESTORATION_EPOCHS, "fusion_epochs": FUSION_EPOCHS,
            "transition_epochs": min(5, FUSION_EPOCHS),
            "lr": LEARNING_RATE, "min_lr": min(1e-6, LEARNING_RATE), "lr_warmup_epochs": 3,
            "weight_decay": 1e-4, "betas": [0.9, 0.999], "grad_clip": 1.0,
            "save_every_epochs": 10, "progress": True,
        },
        "degradation": deepcopy(DEFAULT_DEGRADATION),
        # SSIM/CbCr use averages: weights account for sums in the references.
        "fusion_loss": {"intensity_weight": 10.0, "gradient_weight": 2.0,
                        "ssim_weight": 2.0, "color_weight": 20.0, "require_color": True},
        "auxiliary_loss": {"reconstruction_weight": 5.0, "keep_weight": 2.0,
                           "demand_weight": 1.0, "error_weight": 1.0, "demand_scale": 0.1,
                           "average_kernel": 5, "gradient_weight": 0.25, "sobel_divisor": 8.0,
                           "error_scale": 0.1},
        "experiment": {"output_dir": OUTPUT_DIR, "resume_training": RESUME_TRAINING},
    }


def check_training(settings):
    for key in ("batch_size", "accumulation_steps", "restoration_epochs", "fusion_epochs",
                "cpu_threads", "save_every_epochs"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("workers", "transition_epochs", "lr_warmup_epochs"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    if settings["lr"] <= 0 or not 0 <= settings["min_lr"] <= settings["lr"] or settings["grad_clip"] <= 0:
        raise ValueError("Invalid learning rate or gradient clipping.")


def run(config, resume=None):
    """Train full epochs; a checkpoint is written after every epoch."""
    settings = config["training"]
    check_training(settings)
    total_epochs = settings["restoration_epochs"]+settings["fusion_epochs"]
    torch.set_num_threads(settings["cpu_threads"])
    seed_everything(settings["seed"], settings["deterministic"])
    device = choose_device(settings["device"])
    dtype = amp_settings(device, settings["amp"])
    dataset = PairedFusionDataset(**dataset_options(config["data"]["train_MSRS"], settings["seed"], True, config["degradation"]))
    if dataset.mode == "inference":
        raise ValueError("Training requires synthetic or paired reference mode.")
    fingerprint = dataset_fingerprint(dataset)
    model = RobustFusionNet(config["model"]).to(device)
    config["model"] = model.config
    if config.get("implementation", {}).get("version") != MODEL_VERSION:
        raise ValueError("Settings must describe the current network implementation.")
    signature = training_signature(config)
    fusion_criterion = FusionLoss(**config["fusion_loss"]).to(device)
    auxiliary_criterion = AuxiliaryLoss(**config["auxiliary_loss"]).to(device)
    if auxiliary_criterion.error_scale != model.config["error_scale"]:
        raise ValueError("Model and auxiliary error regression scales must match.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
        weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    output = resolve_path(config["experiment"]["output_dir"])
    latest = output/"latest.pth"
    if resume is None and config["experiment"].get("resume_training", False):
        resume = latest
    if latest.exists() and not resume:
        raise FileExistsError(f"{latest} exists; set RESUME_TRAINING=True or use a new OUTPUT_DIR.")
    completed, updates, history = 0, 0, []
    if resume:
        state = read_checkpoint(resume, expected_implementation=MODEL_VERSION)
        if state["training_signature"] != signature or state["dataset_fingerprint"] != fingerprint:
            raise ValueError("Resume parameters/data differ. Keep the original recipe and source data.")
        completed = state["epoch"]
        if not isinstance(completed, int) or not 0 <= completed <= total_epochs:
            raise ValueError("Invalid checkpoint epoch.")
        expected_updates = completed*((len(dataset)+settings["batch_size"]*settings["accumulation_steps"]-1)//
                                      (settings["batch_size"]*settings["accumulation_steps"]))
        if state["step"] != expected_updates:
            raise ValueError("Checkpoint epoch and optimizer update count are inconsistent.")
        strict_load(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        updates, history = state["step"], state["history"]
        if len(history) != completed:
            raise ValueError("Checkpoint epoch history is inconsistent.")
        restore_rng(state["rng"])
    if completed >= total_epochs:
        print(f"Already completed epoch {completed}/{total_epochs}.")
        return {"epoch": completed, "step": updates, "history": history}
    output.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={parameters:,} pairs={len(dataset)} "
          f"restoration={settings['restoration_epochs']} fusion={settings['fusion_epochs']} epochs "
          f"batch={settings['batch_size']} accumulation={settings['accumulation_steps']}")
    for epoch in range(completed, total_epochs):
        model.train()
        restoration_only = epoch < settings["restoration_epochs"]
        phase = "restoration" if restoration_only else "fusion"
        phase_epoch = epoch+1 if restoration_only else epoch-settings["restoration_epochs"]+1
        phase_total = settings[f"{phase}_epochs"]
        strength = epoch_joint_strength(epoch, settings)
        lr = epoch_learning_rate(epoch, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampler = EpochBatchSampler(len(dataset), settings["batch_size"], epoch, settings["seed"])
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings["workers"],
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(settings["seed"]+999+epoch))
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} | {phase} {phase_epoch}/{phase_total}",
                        unit="batch", dynamic_ncols=True, disable=not settings["progress"])
        sums, seen, group_samples = {}, 0, 0
        optimizer.zero_grad(set_to_none=True)
        try:
            for batch_index, batch in enumerate(progress):
                if batch_index % settings["accumulation_steps"] == 0:
                    # Weight the tail by its actual samples; no duplication/drop.
                    group_samples = min(settings["batch_size"]*settings["accumulation_steps"],
                                        len(dataset)-batch_index*settings["batch_size"])
                batch = move_batch(batch, device)
                count = batch["visible"].shape[0]
                with autocast_context(device, dtype):
                    outputs = model(batch["visible"], batch["infrared"], quality_strength=strength,
                                    demand_strength=strength, restoration_only=restoration_only)
                    auxiliary, _ = auxiliary_criterion(outputs, batch)
                    fusion = auxiliary.new_zeros(()) if restoration_only else fusion_criterion(
                        outputs["fused"], batch["hq_visible"], batch["hq_infrared"])
                    total = auxiliary+strength*fusion
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, batch {batch_index+1}.")
                scaler.scale(total*(count/group_samples)).backward()
                for name, value in (("loss", total), ("fusion_loss", fusion), ("auxiliary_loss", auxiliary)):
                    sums[name] = sums.get(name, 0.0)+float(value.detach())*count
                seen += count
                if (batch_index+1) % settings["accumulation_steps"] == 0 or batch_index+1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                progress.set_postfix(loss=f"{sums['loss']/seen:.4f}", lr=f"{lr:.2g}")
        finally:
            progress.close()
        if seen != len(dataset):
            raise RuntimeError("An epoch must visit every training pair exactly once.")
        completed = epoch+1
        record = dict(epoch=completed, phase=phase, lr=lr, joint_strength=strength,
                      **{name:value/seen for name,value in sums.items()})
        history.append(record)
        print(f"Epoch {completed}/{total_epochs} | {phase} | loss={record['loss']:.5f} "
              f"fusion={record['fusion_loss']:.5f} aux={record['auxiliary_loss']:.5f} lr={lr:.3g}")
        state = {"format_version": 3, "implementation_version": MODEL_VERSION,
                 "model": model.state_dict(), "config": config, "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": completed, "step": updates,
                 "history": history, "rng": capture_rng(), "training_signature": signature,
                 "dataset_fingerprint": fingerprint, "parameters": parameters,
                 "torch_version": str(torch.__version__)}
        atomic_checkpoint(state, latest)
        if completed % settings["save_every_epochs"] == 0:
            atomic_checkpoint(state, output/f"epoch_{completed:03d}.pth")
    print(f"Saved: {latest}")
    return {"epoch": completed, "step": updates, "history": history}


def main():
    run(_build_settings())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit train_MSRS.py, then run: python train_MSRS.py")
    main()
