"""Edit the settings below and run: python generate_degradation.py.

Public-dataset originals are clean/HQ references. Images are not resized.
Outputs contain paired VI/IR PNGs, support masks, soft envelopes and previews.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version
from tqdm import tqdm

from utils.dataset import image_index
from utils.experiment import resolve_path
from utils.synthetic_degradation import (INFRARED_TYPES, LEVELS as VALID_LEVELS,
    PRESETS, PROTOCOL_VERSION, VISIBLE_TYPES, SyntheticDegrader, local_envelope)


# ===================== EDIT HERE =====================
VISIBLE_DIR = "../datasets/train_MSRS/vis"     # folder of clean visible images
INFRARED_DIR = "../datasets/train_MSRS/ir"     # folder of clean infrared images
OUTPUT_DIR = "data/degraded/train_MSRS"

VI_DEGRADATION = "low_light"              # clean / low_light / noise / blur / rain / snow / haze
IR_DEGRADATION = "clean"                  # clean / noise / stripe / low_contrast
LEVELS = ["light", "medium", "heavy"]     # e.g. ["medium"] generates a single severity
GLOBAL_DEGRADATION = True                 # True: full image; False: local, ~40% area
GENERATE_ALL_TYPES = False                # True: all nine single-modality conditions; False: the pair above

# Optional, used for haze only. None produces uniform haze without a depth model.
DEPTH_DIR = None                          # folder of same-name PNG or .npy depth maps
DEPTH_IS_INVERSE = True                   # True: larger values are nearer; False: farther
# =====================================================


def _build_settings():
    for name in ("GLOBAL_DEGRADATION", "GENERATE_ALL_TYPES", "DEPTH_IS_INVERSE"):
        if not isinstance(globals()[name], bool):
            raise TypeError(f"{name} must be True or False.")
    if not isinstance(LEVELS, (list, tuple)) or not LEVELS or \
            any(level not in VALID_LEVELS for level in LEVELS) or len(set(LEVELS)) != len(LEVELS):
        raise ValueError(f"LEVELS must select distinct values from {VALID_LEVELS}.")
    if VI_DEGRADATION not in ("clean", *VISIBLE_TYPES) or IR_DEGRADATION not in ("clean", *INFRARED_TYPES):
        raise ValueError("Invalid VI_DEGRADATION or IR_DEGRADATION.")
    conditions = ([(kind, "clean") for kind in VISIBLE_TYPES] +
                  [("clean", kind) for kind in INFRARED_TYPES]) if GENERATE_ALL_TYPES else \
                  [(VI_DEGRADATION, IR_DEGRADATION)]
    return dict(visible_dir=VISIBLE_DIR, infrared_dir=INFRARED_DIR, output=OUTPUT_DIR,
                conditions=conditions, levels=list(LEVELS), global_degradation=GLOBAL_DEGRADATION,
                depth_dir=DEPTH_DIR, depth_is_inverse=DEPTH_IS_INVERSE,
                seed=20261007, preview_count=6, limit=None, progress=True)


def _seed(base, sample_id, key):
    # Stable across Python processes, directory traversal and selected conditions.
    value = f"{base}|{sample_id}|{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _read(path, visible):
    with Image.open(path) as image:
        return np.array(image.convert("RGB" if visible else "L"), dtype=np.float32) / 255.0


def _save(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def _preview(vi, ir, degraded_vi, degraded_ir, path, title):
    panels = [(vi, "VI clean"), (degraded_vi, "VI degraded"),
              (ir, "IR clean"), (degraded_ir, "IR degraded")]
    h, w = vi.shape[:2]
    scale = min(1.0, 480 / w)
    pw, ph = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (pw * 2, (ph + 24) * 2 + 24), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill="white")
    for index, (array, label) in enumerate(panels):
        panel = Image.fromarray(np.rint(array * 255).astype(np.uint8)).convert("RGB")
        panel = panel.resize((pw, ph), Image.Resampling.BILINEAR)
        x, y = (index % 2) * pw, 24 + (index // 2) * (ph + 24)
        draw.text((x + 6, y + 4), label, fill="white")
        canvas.paste(panel, (x, y + 24))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _depth_index(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Depth directory does not exist: {directory}")
    index = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"):
            key = path.relative_to(directory).with_suffix("").as_posix()
            if key in index:
                raise ValueError(f"Duplicate depth sample ID: {key}")
            index[key] = path
    if not index:
        raise ValueError(f"No depth maps in {directory}.")
    return index


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(settings):
    cv2.setNumThreads(1)
    vi_root, ir_root, output = (resolve_path(settings[k]).resolve() for k in ("visible_dir", "infrared_dir", "output"))
    for source in (vi_root, ir_root):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("OUTPUT_DIR must be separate from both source directories.")
    vi_index, ir_index = image_index(vi_root), image_index(ir_root)
    if set(vi_index) != set(ir_index):
        raise ValueError(f"Unpaired dataset: missing IR={sorted(set(vi_index)-set(ir_index))[:5]}, "
                         f"missing VI={sorted(set(ir_index)-set(vi_index))[:5]}.")
    sample_ids = sorted(vi_index)
    if settings["limit"] is not None:
        if not isinstance(settings["limit"], int) or settings["limit"] < 1:
            raise ValueError("Image limit must be positive.")
        sample_ids = sample_ids[:settings["limit"]]
    depth_root = resolve_path(settings["depth_dir"]).resolve() if settings["depth_dir"] is not None else None
    depth_index = _depth_index(depth_root) if depth_root is not None and any(
        vi == "haze" for vi, _ in settings["conditions"]) else None
    if depth_index is not None:
        missing = set(sample_ids) - set(depth_index)
        if missing:
            raise ValueError(f"Missing depth maps: {sorted(missing)[:5]}")
    # Preflight all image pairs before creating any output.
    for sample in sample_ids:
        with Image.open(vi_index[sample]) as vi, Image.open(ir_index[sample]) as ir:
            if vi.size != ir.size:
                raise ValueError(f"Unregistered image dimensions for {sample}: {vi.size} vs {ir.size}")
    fingerprint = hashlib.sha256()
    for index in (vi_index, ir_index, depth_index or {}):
        for sample in sample_ids:
            if sample in index:
                path = index[sample]
                stat = path.stat()
                fingerprint.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    source_fingerprint = fingerprint.hexdigest()
    operator_sha = hashlib.sha256((Path(__file__).parent / "utils/synthetic_degradation.py").read_bytes()).hexdigest()
    entry_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    degrader = SyntheticDegrader()
    scope = "global" if settings["global_degradation"] else "local"
    generated = []
    for vi_kind, ir_kind in settings["conditions"]:
        condition = f"vi_{vi_kind}__ir_{ir_kind}"
        for level in settings["levels"]:
            root = output / condition / scope / level
            signature = dict(protocol=PROTOCOL_VERSION, operator_sha256=operator_sha, entry_sha256=entry_sha,
                numpy_version=np.__version__, opencv_version=cv2.__version__, pillow_version=pillow_version,
                visible_dir=str(vi_root), infrared_dir=str(ir_root), source_fingerprint=source_fingerprint,
                sample_ids=sample_ids, vi_type=vi_kind, ir_type=ir_kind, level=level, scope=scope,
                seed=settings["seed"],
                depth_dir=str(depth_root) if depth_index is not None else None,
                depth_is_inverse=settings["depth_is_inverse"] if depth_index is not None else None,
                hq_convention="public_dataset_originals_are_hq")
            info_path = root / "run_info.json"
            if info_path.exists():
                previous = json.loads(info_path.read_text(encoding="utf-8"))
                if previous.get("settings") != signature:
                    raise ValueError(f"Existing output has different settings/sources: {root}. Use another OUTPUT_DIR.")
            elif root.exists() and any(root.iterdir()):
                raise ValueError(f"Nonempty output without generation record: {root}. Use another OUTPUT_DIR.")
            root.mkdir(parents=True, exist_ok=True)
            _json(info_path, {"settings": signature, "complete": False})
            record_path = root / "records.csv"
            temporary_records = record_path.with_suffix(".csv.tmp")
            with temporary_records.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "vi_source", "ir_source", "depth_source",
                    "vi_seed", "ir_seed", "mask_seed", "vi_parameters", "ir_parameters"])
                writer.writeheader()
                progress = tqdm(sample_ids, desc=f"{condition} {scope} {level}", unit="pair",
                                dynamic_ncols=True, disable=not settings["progress"])
                for position, sample in enumerate(progress):
                    vi, ir = _read(vi_index[sample], True), _read(ir_index[sample], False)
                    mask_seed = _seed(settings["seed"], sample, f"mask:{condition}")
                    envelope = None if settings["global_degradation"] else local_envelope(
                        ir.shape, np.random.default_rng(mask_seed), stripe=ir_kind == "stripe")
                    depth = None
                    if vi_kind == "haze" and depth_index is not None:
                        path = depth_index[sample]
                        depth = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else _read(path, False)
                    vi_seed, ir_seed = (_seed(settings["seed"], sample, f"{modality}:{kind}")
                                       for modality, kind in (("visible", vi_kind), ("infrared", ir_kind)))
                    dv = degrader(vi, vi_kind, level, np.random.default_rng(vi_seed),
                        modality="visible", global_degradation=settings["global_degradation"],
                        envelope=envelope, depth=depth, depth_is_inverse=settings["depth_is_inverse"])
                    di = degrader(ir, ir_kind, level, np.random.default_rng(ir_seed),
                        modality="infrared", global_degradation=settings["global_degradation"], envelope=envelope)
                    for folder, array in (("vi", dv.image), ("ir", di.image), ("mask_vi", dv.mask),
                                          ("mask_ir", di.mask), ("envelope_vi", dv.envelope), ("envelope_ir", di.envelope)):
                        _save(array, root / folder / (sample + ".png"))
                    if position < settings["preview_count"]:
                        _preview(vi, ir, dv.image, di.image, root / "previews" / (sample + ".png"),
                                 f"{condition} | {scope} | {level} | {sample}")
                    writer.writerow(dict(sample_id=sample, vi_source=str(vi_index[sample]), ir_source=str(ir_index[sample]),
                        depth_source=str(depth_index[sample]) if vi_kind == "haze" and depth_index is not None else "",
                        vi_seed=vi_seed, ir_seed=ir_seed, mask_seed=mask_seed,
                        vi_parameters=json.dumps(dv.parameters, sort_keys=True),
                        ir_parameters=json.dumps(di.parameters, sort_keys=True)))
            temporary_records.replace(record_path)
            _json(info_path, {"settings": signature, "complete": True, "pairs": len(sample_ids),
                              "presets": {kind: PRESETS[kind][level] for kind in (vi_kind, ir_kind) if kind != "clean"}})
            generated.append(root)
            print(f"Saved {len(sample_ids)} pairs: {root}")
    return generated


def main():
    run(_build_settings())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit the settings at the top of generate_degradation.py, then run the file directly.")
    main()
