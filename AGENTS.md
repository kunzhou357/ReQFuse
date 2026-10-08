# AGENTS.md

PyTorch research codebase for robust infrared+visible image fusion (ReQFuse, `MODEL_VERSION = "restormer_v2"` in model/fusion.py). No CI, linter, formatter, or test suite is configured. The three root scripts are the only runnable entry points.

## Environment

- Run everything with the conda env interpreter: `& "D:\miniconda3\envs\img_fusion\python.exe" train.py` (Python 3.12, torch 2.5.1+cu124, CUDA available). The `python` on PATH (3.14) does **not** have torch.
- Dependencies are listed in `requirements.txt` (torch, numpy, opencv-python, Pillow, tqdm), with loose floors only.

## Entry points — edit-and-run, no CLI args

Each script is configured by editing the constants between the `==== EDIT HERE ====` markers at the top of the file, then running it with **no arguments** — any CLI arg aborts with SystemExit.

- `train.py` — two-stage training: `RESTORATION_EPOCHS` restoration-only epochs, then `FUSION_EPOCHS` joint epochs (fusion loss ramps in over `transition_epochs`). Saves `OUTPUT_DIR/latest.pth` every epoch plus `epoch_XXX.pth` every 10 epochs.
- `test.py` — batch-1 inference from `CHECKPOINT_PATH` (default `ckpt/latest.pth`); writes `rgb/` and `gray/` PNGs to `OUTPUT_DIR`.
- `generate_degradation.py` — offline paired VI/IR degradation generator; records a `run_info.json` fingerprint per output dir and refuses to reuse a dir whose settings/sources differ. Its fingerprint hashes the entry file itself, so refactoring it invalidates fingerprints of existing output dirs (regenerate into new dirs).

## Training gotchas

- `train.py` raises `FileExistsError` if `OUTPUT_DIR/latest.pth` exists and `RESUME_TRAINING=False`.
- Resume strictly validates a training signature plus a dataset fingerprint built from file sizes and mtimes — editing the recipe/config or re-copying source images invalidates resume.
- `ONLINE_DEGRADATION=True` (default) reads HQ originals from `VISIBLE_DIR`/`INFRARED_DIR` and degrades on the fly; `False` requires both `HQ_*_DIR` and pre-degraded pairs from `generate_degradation.py`.
- `AuxiliaryLoss.error_scale` must equal model config `error_scale` or training aborts.

## Checkpoints

- Only `format_version 3` (epoch-based, resumable) is accepted; there is no legacy step-based support.
- Loads are always `strict=True`; implementation version mismatches fail loudly rather than degrading silently.
- Checkpoints saved before the ablation-flag cleanup carry `use_demand`/`use_quality`/`gate_private`/`gate_messages`/`local_exchange` in `config["model"]`; `model_config()` (model/fusion.py) drops these via `LEGACY_MODEL_OPTIONS` so they still load. Never rename model parameters or `MODEL_VERSION` — both are compatibility surfaces.

## Layout & data conventions

- Relative paths in configs resolve against the repo root (`resolve_path` in utils/experiment.py), independent of the current working directory.
- `model/` and `utils/` are import-only packages with relative imports and no `__init__.py`; do not run files inside them directly.
- VI/IR datasets must be strictly paired by relative-path sample ID (filename stem, indexed recursively) and registered (identical dimensions); every loader validates this and lists mismatches.
- Paired training mode auto-discovers `mask_vi`/`mask_ir` as sibling directories of the visible/infrared dirs.
- Training data defaults to `data/train_MSRS/vis` (named `vis`, not `vi`) and `data/train_MSRS/ir`. Test data lives under `data/test_MSRS/<condition>/<level 1-3>/{vi,ir}` (e.g. `vi_snow`, `ir_stripe`), with shared clean modalities at `data/test_MSRS/vi` and `data/test_MSRS/ir`; `results/` mirrors this per condition.
- Degradation operators are defined once in `utils/synthetic_degradation.py` (nine operators, three severity levels) and consumed by both the online sampler (`utils/degradation.py`, used by dataset workers) and the offline `generate_degradation.py` — keep them in sync there, never duplicate. Their numeric behavior and RNG draw order are load-bearing for training reproducibility.
- Degradation masks supervise training only; they are never network inputs. The model's forward returns only `fused`, `restored`, `demand_logits` and `error_raw`; training consumes exactly these.
