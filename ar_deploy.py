from __future__ import annotations

import argparse
import calendar
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform

# Make relative paths work whether launched from repo root or DeepSatModels/
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

AR_ROIG = [
    [-94.7610, 36.6652],
    [-94.7610, 32.8376],
    [-89.5522, 36.6652],
    [-89.5522, 32.8376],
]
AR_LON_MIN = min(AR_ROIG[0][0], AR_ROIG[1][0])
AR_LON_MAX = max(AR_ROIG[2][0], AR_ROIG[3][0])
AR_LAT_MIN = min(AR_ROIG[1][1], AR_ROIG[3][1])
AR_LAT_MAX = max(AR_ROIG[0][1], AR_ROIG[2][1])
AR_GRID_N = 20

SAMPLE_REQUIREMENTS: dict[int, int] = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 1,
    11: 1,
    12: 1,
}

SELECTED_BANDS: dict[str, list[str]] = {
    "10m": ["B2", "B3", "B4", "B8"],
    "20m": ["B5", "B6", "B7", "B8A", "B11", "B12"],
    "SCL": ["SCL"],
}

STATEWIDE_REQUIRED_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
    "SCL",
    "TCI_R",
    "TCI_G",
    "TCI_B",
]

FASTDIFFSR_DEFAULT_CONFIG = "fastdiffsr/config/sr_fastdiffsr_infer_x4_planet.json"
_TQDM_IMPORT_WARNED = False


def _read_jsonc(fp: str | Path) -> dict:
    text = ""
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            text += line.split("//")[0] + "\n"
    return json.loads(text)


def _resolve_fastdiffsr_resume_state(checkpoint_path: str | Path) -> str:
    """
    Accepts one of:
      - checkpoint dir containing *_gen.pth
      - *_gen.pth or *_opt.pth file
      - resume-state prefix path (.../Ixxxx_Eyyyy)
    Returns resume-state prefix path expected by FastDiffSR.
    """
    cp = Path(checkpoint_path)

    def _strip_suffix(p: Path) -> str:
        s = str(p)
        if s.endswith("_gen.pth"):
            return s[: -len("_gen.pth")]
        if s.endswith("_opt.pth"):
            return s[: -len("_opt.pth")]
        return s

    if cp.is_dir():
        cands = []
        for fp in cp.glob("*_gen.pth"):
            m = re.match(r"I(\d+)_E(\d+)_gen\.pth$", fp.name)
            if m:
                cands.append((int(m.group(1)), int(m.group(2)), fp))
        if not cands:
            raise SystemExit(f"No '*_gen.pth' checkpoints found in: {cp}")
        cands.sort(key=lambda x: (x[0], x[1]))
        return _strip_suffix(cands[-1][2])

    if cp.suffix == ".pth":
        if str(cp).endswith("_gen.pth") or str(cp).endswith("_opt.pth"):
            return _strip_suffix(cp)
        raise SystemExit(f"Unsupported checkpoint file name (expected *_gen.pth or *_opt.pth): {cp}")

    gen_fp = Path(f"{cp}_gen.pth")
    if gen_fp.exists():
        return str(cp)

    raise SystemExit(
        "Unable to resolve FastDiffSR checkpoint. "
        f"Provide a checkpoint dir, *_gen.pth/_opt.pth file, or resume prefix. Got: {checkpoint_path}"
    )


def _build_fastdiffsr_model(*, config_path: str, resume_state: str, device_ids: list[int]):
    fastdiffsr_root = APP_DIR / "fastdiffsr"
    if not fastdiffsr_root.exists():
        raise SystemExit(f"FastDiffSR code root not found: {fastdiffsr_root}")

    if str(fastdiffsr_root) not in sys.path:
        sys.path.insert(0, str(fastdiffsr_root))

    # Ensure the intended package is used for import name "model".
    if "model" in sys.modules:
        mod_file = getattr(sys.modules["model"], "__file__", "") or ""
        if str(fastdiffsr_root) not in mod_file:
            del sys.modules["model"]

    fastdiffsr_model = importlib.import_module("model")

    opt = _read_jsonc(config_path)
    opt["phase"] = "val"
    opt["gpu_ids"] = device_ids
    opt["distributed"] = len(device_ids) > 1
    opt["enable_wandb"] = False
    opt.setdefault("path", {})
    opt["path"]["resume_state"] = resume_state

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)

    diffusion = fastdiffsr_model.create_model(opt)
    beta_sched = ((opt.get("model", {}) or {}).get("beta_schedule", {}) or {}).get("val")
    if beta_sched is None:
        beta_sched = ((opt.get("model", {}) or {}).get("beta_schedule", {}) or {}).get("train")
    if beta_sched is None:
        raise SystemExit("FastDiffSR config missing model.beta_schedule.{val|train}.")
    diffusion.set_new_noise_schedule(beta_sched, schedule_phase="val")
    return diffusion


def _read_rgb10m_image(meta_dir: Path, day: date, *, cloud_frac_max: float = 0.10) -> tuple[np.ndarray, Path] | None:
    import cv2

    day_s = day.strftime("%Y-%m-%d")
    b2_fp = meta_dir / day_s / f"B2_{day_s}.tif"
    b3_fp = meta_dir / day_s / f"B3_{day_s}.tif"
    b4_fp = meta_dir / day_s / f"B4_{day_s}.tif"
    scl_fp = meta_dir / day_s / f"SCL_{day_s}.tif"
    if not (b2_fp.exists() and b3_fp.exists() and b4_fp.exists() and scl_fp.exists()):
        return None

    with rasterio.open(b2_fp) as src:
        b2 = src.read(1)
    with rasterio.open(b3_fp) as src:
        b3 = src.read(1)
    with rasterio.open(b4_fp) as src:
        b4 = src.read(1)
    with rasterio.open(scl_fp) as src:
        scl = src.read(1)

    if scl.shape != b2.shape:
        scl = cv2.resize(scl, (b2.shape[1], b2.shape[0]), interpolation=cv2.INTER_NEAREST)

    cloud_frac = float((scl > 7).sum()) / float(scl.size)
    if cloud_frac > float(cloud_frac_max):
        return None

    # Natural-color ordering: R=B4, G=B3, B=B2
    rgb = np.dstack([b4, b3, b2]).astype(np.float32)
    return rgb, b2_fp


def _stretch_to_uint8(rgb: np.ndarray, *, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(rgb.shape[2]):
        ch = rgb[..., c]
        lo = float(np.percentile(ch, p_low))
        hi = float(np.percentile(ch, p_high))
        if hi <= lo:
            hi = lo + 1.0
        x = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        out[..., c] = (x * 255.0).astype(np.uint8)
    return out


def _pad_rgb_reflect_or_edge(tile: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    h, w = tile.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile
    mode = "reflect" if (h > 1 and w > 1) else "edge"
    return np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)], mode=mode)


def _tile_positions(size: int, tile_size: int, step: int) -> list[int]:
    if size <= tile_size:
        return [0]
    out = list(range(0, max(1, size - tile_size + 1), step))
    last = max(0, size - tile_size)
    if out[-1] != last:
        out.append(last)
    return out


def _center_heavy_weight_map(size: int) -> np.ndarray:
    if size <= 1:
        return np.ones((size, size), dtype=np.float32)
    ramp = np.hanning(size).astype(np.float32)
    if float(ramp.max()) <= 0.0:
        ramp = np.ones((size,), dtype=np.float32)
    ramp = np.clip(ramp, 1.0e-3, None)
    return np.outer(ramp, ramp).astype(np.float32)


def _infer_fastdiffsr_rgb(
    diffusion,
    rgb_lr: np.ndarray,
    *,
    scale: int,
    lr_tile_size: int = 64,
    lr_overlap: int = 0,
) -> np.ndarray:
    import cv2
    import torch

    if lr_tile_size <= 0:
        raise SystemExit(f"lr_tile_size must be > 0, got {lr_tile_size}")
    if scale <= 0:
        raise SystemExit(f"scale must be > 0, got {scale}")
    if lr_overlap < 0:
        raise SystemExit(f"lr_overlap must be >= 0, got {lr_overlap}")
    if lr_overlap >= lr_tile_size:
        raise SystemExit(f"lr_overlap must be < lr_tile_size ({lr_tile_size}), got {lr_overlap}")

    h, w, _ = rgb_lr.shape
    rgb8 = _stretch_to_uint8(rgb_lr)
    sr_tile_size = lr_tile_size * scale
    sr_accum = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight_accum = np.zeros((h * scale, w * scale, 1), dtype=np.float32)
    step = max(1, lr_tile_size - lr_overlap)
    weight_full = _center_heavy_weight_map(sr_tile_size)[..., None]

    y_positions = _tile_positions(h, lr_tile_size, step)
    x_positions = _tile_positions(w, lr_tile_size, step)

    for iy in y_positions:
        for ix in x_positions:
            tile_lr = rgb8[iy : iy + lr_tile_size, ix : ix + lr_tile_size, :]
            vh, vw = tile_lr.shape[:2]
            tile_lr = _pad_rgb_reflect_or_edge(tile_lr, target_h=lr_tile_size, target_w=lr_tile_size)
            tile_sr_in = cv2.resize(tile_lr, (sr_tile_size, sr_tile_size), interpolation=cv2.INTER_CUBIC)

            sr_t = torch.from_numpy(np.transpose(tile_sr_in, (2, 0, 1))).float() / 255.0
            sr_t = sr_t * 2.0 - 1.0
            sr_t = sr_t.unsqueeze(0)  # (1, 3, H, W)

            batch = {"SR": sr_t, "HR": sr_t.clone()}
            diffusion.feed_data(batch)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals(need_LR=False)

            sr_tile = visuals["SR"].squeeze(0).clamp(-1, 1).permute(1, 2, 0).numpy()
            sr_tile = np.clip(((sr_tile + 1.0) * 127.5).round(), 0, 255).astype(np.uint8)
            sr_tile = sr_tile[: vh * scale, : vw * scale, :]

            oy = iy * scale
            ox = ix * scale
            weight = weight_full[: vh * scale, : vw * scale, :]
            sr_accum[oy : oy + (vh * scale), ox : ox + (vw * scale), :] += sr_tile.astype(np.float32) * weight
            weight_accum[oy : oy + (vh * scale), ox : ox + (vw * scale), :] += weight

    np.maximum(weight_accum, 1.0e-6, out=weight_accum)
    sr_out = np.clip(np.round(sr_accum / weight_accum), 0, 255).astype(np.uint8)
    return sr_out


def _write_rgb_geotiff(*, out_fp: Path, rgb: np.ndarray, template_fp: Path, scale: int):
    with rasterio.open(template_fp) as src:
        profile = src.profile
        transform = src.transform * rasterio.Affine.scale(1.0 / float(scale), 1.0 / float(scale))

    profile.update(
        count=3,
        dtype=rasterio.uint8,
        height=int(rgb.shape[0]),
        width=int(rgb.shape[1]),
        nodata=None,
        transform=transform,
        compress="DEFLATE",
    )
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(np.transpose(rgb, (2, 0, 1)))


def _choose_gtiff_blocksize(length: int, preferred: int = 512) -> int:
    size = min(int(preferred), int(length))
    while size > 16 and (size % 16 != 0):
        size -= 1
    return max(16, size)


def _maybe_tqdm(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if not bool(enabled):
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        global _TQDM_IMPORT_WARNED
        if not _TQDM_IMPORT_WARNED:
            print("Progress requested, but tqdm is not available. Falling back to plain output.")
            _TQDM_IMPORT_WARNED = True
        return iterable


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _month_range(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _max_doy_for_month(year: int, month: int) -> int:
    _, end = _month_range(year, month)
    return _parse_ymd(end).timetuple().tm_yday


def _clamp_model_month(month: int) -> int:
    return max(6, min(11, int(month)))


def _default_model_paths(model_month: int) -> tuple[str, str]:
    mm = int(model_month)
    saved_cfg = f"models/saved_models/AR23_focal_{mm:02d}mo/config_file.yaml"
    if Path(saved_cfg).exists():
        cfg = saved_cfg
    else:
        cfg = f"configs/Arkansas/TSViT_AR23_{mm:02d}mo_focal.yaml"

    ckpt = f"models/saved_models/AR23_focal_{mm:02d}mo/best.pth"
    if not Path(ckpt).exists() and mm == 11 and Path("models/saved_models/AR23_focal_11mo/best.pth").exists():
        ckpt = "models/saved_models/AR23_focal_11mo/best.pth"
    return cfg, ckpt


def _list_meta_patches(raw_root: Path) -> list[str]:
    out = []
    for d in raw_root.iterdir():
        if d.is_dir() and re.fullmatch(r"\d+_\d+", d.name):
            out.append(d.name)
    return sorted(out, key=lambda s: (int(s.split("_")[0]), int(s.split("_")[1])))


def _list_dates(meta_dir: Path, *, start_day: str, end_day: str) -> list[date]:
    start = _parse_ymd(start_day)
    end = _parse_ymd(end_day)
    out: list[date] = []
    for d in meta_dir.iterdir():
        if not d.is_dir():
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name):
            continue
        try:
            dt = _parse_ymd(d.name)
        except Exception:
            continue
        if start <= dt <= end:
            out.append(dt)
    return sorted(out)


def _list_valid_fastdiffsr_dates(
    *,
    meta_dir: Path,
    start_day: str,
    end_day: str,
    cloud_frac_max: float,
) -> list[date]:
    valid: list[date] = []
    for d in _list_dates(meta_dir, start_day=start_day, end_day=end_day):
        try:
            got = _read_rgb10m_image(meta_dir, d, cloud_frac_max=float(cloud_frac_max))
        except Exception:
            got = None
        if got is not None:
            valid.append(d)
    return valid


def _plan_statewide_anchor_dates(
    *,
    raw_root: Path,
    meta_patches: list[str],
    start_day: str,
    end_day: str,
    cloud_frac_max: float,
) -> tuple[dict[str, date], dict]:
    """
    Pick one month-wide anchor date (latest date with maximum valid tile coverage), then
    assign each tile the anchor date if available, otherwise the nearest valid in-month date.
    """
    valid_dates_by_mp: dict[str, list[date]] = {}
    date_counts: dict[date, int] = {}
    status_counts = {
        "meta_patch_missing": 0,
        "no_dates_in_month": 0,
        "no_valid_observation": 0,
        "valid_tiles": 0,
    }

    for mp in [str(x) for x in meta_patches]:
        meta_dir = raw_root / mp
        if not meta_dir.exists():
            status_counts["meta_patch_missing"] += 1
            continue

        month_dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
        if not month_dates:
            status_counts["no_dates_in_month"] += 1
            continue

        valid_dates = _list_valid_fastdiffsr_dates(
            meta_dir=meta_dir,
            start_day=start_day,
            end_day=end_day,
            cloud_frac_max=float(cloud_frac_max),
        )
        if not valid_dates:
            status_counts["no_valid_observation"] += 1
            continue

        status_counts["valid_tiles"] += 1
        valid_dates = sorted(valid_dates)
        valid_dates_by_mp[mp] = valid_dates
        for d in valid_dates:
            date_counts[d] = int(date_counts.get(d, 0)) + 1

    if not date_counts:
        summary = {
            "strategy": "statewide_anchor",
            "anchor_date": None,
            "anchor_coverage_tiles": 0,
            "planned_tiles": 0,
            "anchor_assigned_tiles": 0,
            "fallback_assigned_tiles": 0,
            "status_counts": status_counts,
            "top_dates": [],
            "fallback_gap_days_hist": {},
        }
        return {}, summary

    anchor_date = max(date_counts.keys(), key=lambda d: (int(date_counts[d]), d.toordinal()))
    plan: dict[str, date] = {}
    anchor_assigned = 0
    fallback_assigned = 0
    gap_hist: dict[str, int] = {}

    for mp in [str(x) for x in meta_patches]:
        valid_dates = valid_dates_by_mp.get(mp)
        if not valid_dates:
            continue
        if anchor_date in valid_dates:
            chosen = anchor_date
            anchor_assigned += 1
            gap_days = 0
        else:
            chosen = min(
                valid_dates,
                key=lambda d: (abs((d - anchor_date).days), -d.toordinal()),
            )
            fallback_assigned += 1
            gap_days = abs((chosen - anchor_date).days)

        plan[mp] = chosen
        k = str(int(gap_days))
        gap_hist[k] = int(gap_hist.get(k, 0)) + 1

    top_dates = sorted(date_counts.items(), key=lambda kv: (-int(kv[1]), -kv[0].toordinal()))[:10]
    summary = {
        "strategy": "statewide_anchor",
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "anchor_coverage_tiles": int(date_counts.get(anchor_date, 0)),
        "planned_tiles": len(plan),
        "anchor_assigned_tiles": int(anchor_assigned),
        "fallback_assigned_tiles": int(fallback_assigned),
        "status_counts": status_counts,
        "top_dates": [
            {"date": d.strftime("%Y-%m-%d"), "tiles": int(c)}
            for d, c in top_dates
        ],
        "fallback_gap_days_hist": gap_hist,
    }
    return plan, summary


def _resample_dates(dates: list[date]) -> list[date]:
    by_month: dict[int, list[date]] = {}
    for d in dates:
        by_month.setdefault(d.month, []).append(d)

    keep: list[date] = []
    for m in sorted(by_month.keys()):
        k = int(SAMPLE_REQUIREMENTS.get(m, 0))
        if k <= 0:
            continue
        ds = sorted(by_month[m])
        if not ds:
            continue
        if k == 1:
            keep.append(min(ds, key=lambda x: abs(x.day - 15)))
        else:
            keep.append(ds[0])
            if ds[-1] != ds[0]:
                keep.append(ds[-1])
    return sorted(set(keep))


def _read_satellite_image(meta_dir: Path, day: date) -> dict[str, dict[str, np.ndarray]] | None:
    import cv2

    day_s = day.strftime("%Y-%m-%d")
    sat: dict[str, dict[str, np.ndarray]] = {k: {} for k in SELECTED_BANDS}

    for res, bands in SELECTED_BANDS.items():
        for b in bands:
            fp = meta_dir / day_s / f"{b}_{day_s}.tif"
            if not fp.exists():
                return None
            with rasterio.open(fp) as src:
                data = src.read(1)
            if res == "SCL" and (data > 7).sum() / data.size > 0.10:
                return None
            sat[res][b] = data

    # Upsample 20m + SCL to 10m grid
    h, w = sat["10m"]["B2"].shape
    for b in SELECTED_BANDS["20m"]:
        sat["20m"][b] = cv2.resize(sat["20m"][b], (w, h), interpolation=cv2.INTER_NEAREST)
    scl_raw = sat["SCL"]["SCL"]
    if scl_raw.shape != (h, w):
        sat["SCL"]["SCL"] = cv2.resize(scl_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    return sat


def _stack_bands(sat: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    layers = [sat[res][b] for res, bs in SELECTED_BANDS.items() for b in bs]
    return np.dstack(layers).astype(np.uint16)


def _pad_tile_reflect_or_edge(tile: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    h, w = tile.shape[1], tile.shape[2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile

    mode = "reflect" if (h > 1 and w > 1) else "edge"
    return np.pad(tile, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], mode=mode)


@dataclass(frozen=True)
class RunSpec:
    year: int
    available_month: int
    model_month: int
    start_day: str
    end_day: str
    max_doy: int


def _infer_meta_patch(
    *,
    meta_patch: str,
    raw_root: Path,
    out_dir: Path,
    config_path: str,
    checkpoint_path: str,
    device_ids: list[int],
    batch_size: int,
    tile_size: int,
    start_day: str,
    end_day: str,
    max_doy: int,
    force: bool,
):
    from models import get_model
    from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
    from utils.config_files_utils import read_yaml
    from utils.torch_utils import get_device, load_from_checkpoint
    import torch

    out_fp = out_dir / f"{meta_patch}.tif"
    if out_fp.exists() and not force:
        return

    meta_dir = raw_root / meta_patch
    if not meta_dir.exists():
        return

    all_dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
    if not all_dates:
        return

    selected_dates = _resample_dates(all_dates)
    if not selected_dates:
        return

    stacks: list[np.ndarray] = []
    doys: list[int] = []
    used_dates: list[str] = []
    for d in selected_dates:
        sat = _read_satellite_image(meta_dir, d)
        if sat is None:
            continue
        stacks.append(_stack_bands(sat))
        doys.append(d.timetuple().tm_yday)
        used_dates.append(d.strftime("%Y-%m-%d"))

    if not stacks:
        return

    # Enforce max_doy (month model) at inference time.
    keep_idx = [i for i, doy in enumerate(doys) if doy <= max_doy]
    stacks = [stacks[i] for i in keep_idx]
    doys = [doys[i] for i in keep_idx]
    used_dates = [used_dates[i] for i in keep_idx]
    if not stacks:
        return

    h, w, _ = stacks[0].shape

    # Use first used date for georeferencing template
    template_fp = meta_dir / used_dates[0] / f"B2_{used_dates[0]}.tif"
    if not template_fp.exists():
        return

    cfg = read_yaml(config_path)
    device = get_device(device_ids, allow_cpu=False)
    cfg["local_device_ids"] = device_ids
    net = get_model(cfg, device)
    if checkpoint_path:
        load_from_checkpoint(net, checkpoint_path, partial_restore=False, device=device)
    net.to(device)
    net.eval()

    transform = PASTIS_segmentation_transform(cfg["MODEL"], is_training=False)
    doy_arr = np.asarray(doys, dtype=np.int16)
    dummy_labels = np.zeros((1, tile_size, tile_size), dtype=np.uint8)

    pred_mosaic = np.zeros((h, w), dtype=np.uint8)

    batch_inputs: list[torch.Tensor] = []
    batch_pos: list[tuple[int, int]] = []

    def flush():
        if not batch_inputs:
            return
        inputs = torch.stack(batch_inputs, 0).to(device)
        with torch.no_grad():
            logits = net(inputs)
            preds = logits.argmax(1).cpu().numpy().astype(np.uint8)  # (B,H,W)
        for i, (iy, ix) in enumerate(batch_pos):
            vh = min(tile_size, h - iy)
            vw = min(tile_size, w - ix)
            pred_mosaic[iy : iy + vh, ix : ix + vw] = preds[i, :vh, :vw]
        batch_inputs.clear()
        batch_pos.clear()

    y_positions = list(range(0, h, tile_size))
    x_positions = list(range(0, w, tile_size))

    for iy in y_positions:
        for ix in x_positions:
            # Build (T, Ht, Wt, 11) then pad to (T,24,24,11)
            tiles_t = [img[iy : iy + tile_size, ix : ix + tile_size, :] for img in stacks]
            tile = np.stack(tiles_t, 0)
            tile = _pad_tile_reflect_or_edge(tile, target_h=tile_size, target_w=tile_size)

            # Drop SCL channel (last) -> 10 channels, match training normalization.
            if tile.shape[-1] == 11:
                tile = tile[..., :-1]

            # (T, H, W, C) -> (T, C, H, W)
            img_np = np.transpose(tile.astype(np.float32), (0, 3, 1, 2))
            sample = {"img": img_np, "labels": dummy_labels, "doy": doy_arr}
            sample_t = transform(sample)

            batch_inputs.append(sample_t["inputs"])
            batch_pos.append((iy, ix))
            if len(batch_inputs) >= batch_size:
                flush()

    flush()

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_fp) as src:
        profile = src.profile
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        nodata=0,
        compress="DEFLATE",
    )
    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(pred_mosaic, 1)


def cmd_predict(args) -> int:
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"raw_root not found: {raw_root}")

    available_month = int(args.available_month)
    model_month = _clamp_model_month(available_month)
    year = int(args.year)

    start_day = f"{year}-01-01"
    end_day = _month_range(year, available_month)[1]
    max_doy = _max_doy_for_month(year, model_month)

    if available_month < 6:
        print(f"available_month={available_month} < 6: skipping predictions (no supported model yet).")
        return 0

    config_path = args.config or _default_model_paths(model_month)[0]
    checkpoint_path = args.checkpoint or _default_model_paths(model_month)[1]
    if not Path(config_path).exists():
        raise SystemExit(f"config not found: {config_path}")
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device_ids = [int(d) for d in args.device.split(",")] if args.device else [0]
    out_dir = Path(args.pred_root) / f"{model_month:02d}mo"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_patches = args.meta_patches or _list_meta_patches(raw_root)
    print(
        f"Predicting year={year} available_month={available_month} model_month={model_month} "
        f"date_range={start_day}..{end_day} max_doy={max_doy} patches={len(meta_patches)}"
    )

    for mp in meta_patches:
        _infer_meta_patch(
            meta_patch=mp,
            raw_root=raw_root,
            out_dir=out_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device_ids=device_ids,
            batch_size=int(args.batch_size),
            tile_size=int(args.tile_size),
            start_day=start_day,
            end_day=end_day,
            max_doy=max_doy,
            force=bool(args.force),
        )

    # Write a small run metadata file for traceability.
    meta_fp = Path(args.pred_root) / f"{model_month:02d}mo" / "_run.txt"
    meta_fp.write_text(
        f"year={year}\n"
        f"available_month={available_month}\n"
        f"model_month={model_month}\n"
        f"start_day={start_day}\n"
        f"end_day={end_day}\n"
        f"max_doy={max_doy}\n"
        f"config={config_path}\n"
        f"checkpoint={checkpoint_path}\n"
    )
    return 0


def cmd_fastdiffsr(args) -> int:
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"raw_root not found: {raw_root}")

    year = int(args.year)
    month = int(args.month)
    if month < 1 or month > 12:
        raise SystemExit(f"month must be 1..12, got {month}")

    start_day, end_day = _month_range(year, month)
    device_ids = [int(d) for d in str(args.device).split(",")] if args.device else [0]

    config_path = str(args.fastdiffsr_config or FASTDIFFSR_DEFAULT_CONFIG)
    if not Path(config_path).exists():
        raise SystemExit(f"FastDiffSR config not found: {config_path}")
    resume_state = _resolve_fastdiffsr_resume_state(args.fastdiffsr_checkpoint)

    out_dir = Path(args.sr_root) / f"{month:02d}mo"
    out_dir.mkdir(parents=True, exist_ok=True)

    diffusion = _build_fastdiffsr_model(
        config_path=config_path,
        resume_state=resume_state,
        device_ids=device_ids,
    )

    meta_patches = args.meta_patches or _list_meta_patches(raw_root)
    date_policy = str(args.date_policy)
    scale = int(args.scale)
    lr_tile_size = int(args.lr_tile_size)
    lr_overlap = int(args.lr_overlap)
    if scale <= 0:
        raise SystemExit(f"scale must be > 0, got {scale}")
    if lr_tile_size <= 0:
        raise SystemExit(f"lr_tile_size must be > 0, got {lr_tile_size}")
    if lr_overlap < 0:
        raise SystemExit(f"lr_overlap must be >= 0, got {lr_overlap}")
    if lr_overlap >= lr_tile_size:
        raise SystemExit(f"lr_overlap must be < lr_tile_size ({lr_tile_size}), got {lr_overlap}")

    print(
        f"FastDiffSR inference year={year} month={month} date_range={start_day}..{end_day} "
        f"patches={len(meta_patches)} date_policy={date_policy} scale={scale} "
        f"lr_tile_size={lr_tile_size} lr_overlap={lr_overlap}"
    )

    planned_dates_by_mp: dict[str, date] = {}
    statewide_anchor_summary: dict | None = None
    if date_policy == "statewide_anchor":
        # Anchor date should stay state-wide even if caller requests a subset of meta-patches.
        statewide_meta_patches = _list_meta_patches(raw_root)
        planned_dates_by_mp, statewide_anchor_summary = _plan_statewide_anchor_dates(
            raw_root=raw_root,
            meta_patches=statewide_meta_patches,
            start_day=start_day,
            end_day=end_day,
            cloud_frac_max=float(args.cloud_thresh_fastdiffsr),
        )
        print(
            "Statewide anchor plan:"
            f" anchor_date={statewide_anchor_summary.get('anchor_date')}"
            f" coverage={statewide_anchor_summary.get('anchor_coverage_tiles')}"
            f" planned_tiles={statewide_anchor_summary.get('planned_tiles')}"
            f" anchor_assigned={statewide_anchor_summary.get('anchor_assigned_tiles')}"
            f" fallback_assigned={statewide_anchor_summary.get('fallback_assigned_tiles')}"
        )

    wrote = 0
    meta_patch_iter = _maybe_tqdm(
        meta_patches,
        enabled=bool(getattr(args, "progress", True)),
        desc="FastDiffSR meta-patches",
        total=len(meta_patches),
    )
    for mp in meta_patch_iter:
        meta_dir = raw_root / mp
        if not meta_dir.exists():
            continue

        selected: list[tuple[date, np.ndarray, Path]] = []
        if date_policy == "statewide_anchor":
            planned_date = planned_dates_by_mp.get(str(mp))
            if planned_date is None:
                continue
            got = _read_rgb10m_image(meta_dir, planned_date, cloud_frac_max=float(args.cloud_thresh_fastdiffsr))
            if got is not None:
                rgb_lr, template_fp = got
                selected.append((planned_date, rgb_lr, template_fp))
        else:
            dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
            if not dates:
                continue

            if date_policy == "latest":
                dates = sorted(dates, reverse=True)
            elif date_policy == "earliest":
                dates = sorted(dates)
            elif date_policy == "all":
                dates = sorted(dates)
            else:
                raise SystemExit(f"Unsupported date_policy: {date_policy}")

            for d in dates:
                got = _read_rgb10m_image(meta_dir, d, cloud_frac_max=float(args.cloud_thresh_fastdiffsr))
                if got is None:
                    continue
                rgb_lr, template_fp = got
                selected.append((d, rgb_lr, template_fp))
                if date_policy != "all":
                    break

        for d, rgb_lr, template_fp in selected:
            suffix = f"_{d.strftime('%Y%m%d')}" if date_policy == "all" else ""
            out_fp = out_dir / f"{mp}{suffix}.tif"
            if out_fp.exists() and not args.force:
                continue

            sr_rgb = _infer_fastdiffsr_rgb(
                diffusion,
                rgb_lr,
                scale=scale,
                lr_tile_size=lr_tile_size,
                lr_overlap=lr_overlap,
            )
            _write_rgb_geotiff(out_fp=out_fp, rgb=sr_rgb, template_fp=template_fp, scale=scale)
            wrote += 1

    run_meta = out_dir / "_run_fastdiffsr.txt"
    run_meta.write_text(
        f"year={year}\n"
        f"month={month}\n"
        f"start_day={start_day}\n"
        f"end_day={end_day}\n"
        f"date_policy={date_policy}\n"
        f"scale={scale}\n"
        f"lr_tile_size={lr_tile_size}\n"
        f"lr_overlap={lr_overlap}\n"
        f"raw_root={raw_root}\n"
        f"sr_root={args.sr_root}\n"
        f"config={config_path}\n"
        f"resume_state={resume_state}\n"
        f"cloud_thresh_fastdiffsr={float(args.cloud_thresh_fastdiffsr)}\n"
        f"wrote={wrote}\n"
        + (
            f"statewide_anchor_summary={json.dumps(statewide_anchor_summary, sort_keys=True)}\n"
            if statewide_anchor_summary is not None
            else ""
        )
    )
    print(f"FastDiffSR finished. Wrote {wrote} GeoTIFF(s) under: {out_dir}")
    return 0


def cmd_fastdiffsr_rerun_incomplete(args) -> int:
    """
    Scan an existing FastDiffSR month directory for suspiciously small/missing outputs and
    rerun only those meta-patches with force overwrite.
    """
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"raw_root not found: {raw_root}")

    year = int(args.year)
    month = int(args.month)
    if month < 1 or month > 12:
        raise SystemExit(f"month must be 1..12, got {month}")

    date_policy = str(args.date_policy)
    if date_policy == "all":
        raise SystemExit(
            "fastdiffsr_rerun_incomplete currently supports only single-output policies "
            "('latest', 'earliest', or 'statewide_anchor'). date_policy='all' is not supported."
        )

    out_dir = Path(args.sr_root) / f"{month:02d}mo"
    out_dir.mkdir(parents=True, exist_ok=True)
    min_size_mb = float(args.min_size_mb)
    if min_size_mb < 0:
        raise SystemExit(f"--min-size-mb must be >= 0, got {min_size_mb}")
    min_size_bytes = int(round(min_size_mb * 1024 * 1024))

    meta_patches = args.meta_patches or _list_meta_patches(raw_root)
    rerun_meta_patches: list[str] = []
    report_rows: list[str] = []
    kept = 0
    small = 0
    missing = 0

    scan_iter = _maybe_tqdm(
        meta_patches,
        enabled=bool(getattr(args, "progress", True)),
        desc="Scan incomplete SR tiles",
        total=len(meta_patches),
    )
    for mp in scan_iter:
        fp = out_dir / f"{mp}.tif"
        if not fp.exists():
            missing += 1
            report_rows.append(f"{mp}\tmissing\t0\t{fp}")
            if bool(args.include_missing):
                rerun_meta_patches.append(mp)
            continue

        sz = int(fp.stat().st_size)
        if sz < min_size_bytes:
            small += 1
            report_rows.append(f"{mp}\tsmall\t{sz}\t{fp}")
            rerun_meta_patches.append(mp)
        else:
            kept += 1

    # Preserve order, remove duplicates.
    seen = set()
    rerun_meta_patches = [mp for mp in rerun_meta_patches if not (mp in seen or seen.add(mp))]

    report_fp = out_dir / "_rerun_incomplete_candidates.txt"
    report_fp.write_text(
        "meta_patch\treason\tbytes\tpath\n" + ("\n".join(report_rows) + ("\n" if report_rows else "")),
        encoding="utf-8",
    )

    print(
        f"FastDiffSR incomplete scan year={year} month={month} out_dir={out_dir} "
        f"threshold={min_size_mb:.2f}MB total={len(meta_patches)} kept={kept} "
        f"small={small} missing={missing} rerun={len(rerun_meta_patches)} "
        f"include_missing={bool(args.include_missing)}"
    )
    print(f"Candidate report: {report_fp}")

    if not rerun_meta_patches:
        print("No incomplete tiles detected. Nothing to rerun.")
        return 0

    if bool(args.dry_run):
        show = int(max(1, args.show_first))
        print(f"Dry-run only. First {show} candidates: {rerun_meta_patches[:show]}")
        return 0

    rerun_args = argparse.Namespace(**vars(args))
    rerun_args.force = True
    rerun_args.meta_patches = rerun_meta_patches

    print(
        f"Re-running FastDiffSR for {len(rerun_meta_patches)} incomplete meta-patches "
        f"(force overwrite enabled)."
    )
    return cmd_fastdiffsr(rerun_args)


def _bounds_intersect(*, a, b) -> bool:
    return (
        float(a[0]) <= float(b[2])
        and float(a[2]) >= float(b[0])
        and float(a[1]) <= float(b[3])
        and float(a[3]) >= float(b[1])
    )


def _arkansas_grid_cell_bounds_wgs84(i: int, j: int, *, grid_n: int = AR_GRID_N) -> tuple[float, float, float, float]:
    lon_edges = np.linspace(AR_LON_MIN, AR_LON_MAX, int(grid_n) + 1)
    lat_edges = np.linspace(AR_LAT_MIN, AR_LAT_MAX, int(grid_n) + 1)
    return (
        float(lon_edges[int(i)]),
        float(lat_edges[int(j)]),
        float(lon_edges[int(i) + 1]),
        float(lat_edges[int(j) + 1]),
    )


def _month_tag(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _pseudo_month_date(year: int, month: int, month_tag_day: int) -> str:
    max_day = calendar.monthrange(int(year), int(month))[1]
    day = max(1, min(int(month_tag_day), max_day))
    return f"{int(year):04d}-{int(month):02d}-{day:02d}"


def _read_window_for_bounds(ds: rasterio.DatasetReader, bounds_wgs84: tuple[float, float, float, float]) -> tuple[np.ndarray, rasterio.Affine]:
    cell_crs = transform_bounds("EPSG:4326", str(ds.crs), *bounds_wgs84, densify_pts=21)
    win = from_bounds(*cell_crs, transform=ds.transform)
    full = Window(col_off=0, row_off=0, width=ds.width, height=ds.height)
    win = win.intersection(full).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        raise ValueError("window does not intersect source raster")
    data = ds.read(window=win, boundless=False)
    transform = window_transform(win, ds.transform)
    return data, transform


def cmd_statewide_retile(args) -> int:
    in_dir = Path(args.in_dir)
    out_root = Path(args.out_root)
    year = int(args.year)
    month = int(args.month)
    grid_n = int(args.grid_n)
    month_tag_day = int(args.month_tag_day)
    if month < 1 or month > 12:
        raise SystemExit(f"--month must be 1..12, got {month}")
    if grid_n <= 0:
        raise SystemExit(f"--grid-n must be > 0, got {grid_n}")
    if month_tag_day < 1 or month_tag_day > 31:
        raise SystemExit(f"--month-tag-day must be in 1..31, got {month_tag_day}")
    if not in_dir.exists():
        raise SystemExit(f"input dir not found: {in_dir}")

    month_tag = _month_tag(year, month)
    pseudo_date = _pseudo_month_date(year, month, month_tag_day)
    band_files = {band: in_dir / f"{band}_{month_tag}.tif" for band in STATEWIDE_REQUIRED_BANDS}
    missing_inputs = [str(fp) for fp in band_files.values() if not fp.exists()]
    if missing_inputs:
        raise SystemExit(
            "Missing statewide monthly band files:\n" + "\n".join(missing_inputs)
        )

    out_root.mkdir(parents=True, exist_ok=True)
    wrote = 0
    skipped = 0
    failed = 0
    band_write_counts = {band: 0 for band in STATEWIDE_REQUIRED_BANDS}

    datasets: dict[str, rasterio.DatasetReader] = {}
    try:
        for band, fp in band_files.items():
            datasets[band] = rasterio.open(fp)

        print(
            f"Retiling statewide month dir {in_dir} -> {out_root} "
            f"as Arkansas {grid_n}x{grid_n} raw grid with pseudo_date={pseudo_date}"
        )

        for i in range(grid_n):
            for j in range(grid_n):
                cell_wgs84 = _arkansas_grid_cell_bounds_wgs84(i, j, grid_n=grid_n)
                tile_dir = out_root / f"{i}_{j}" / pseudo_date
                tile_dir.mkdir(parents=True, exist_ok=True)

                for band, ds in datasets.items():
                    out_fp = tile_dir / f"{band}_{pseudo_date}.tif"
                    if out_fp.exists() and not bool(args.overwrite):
                        skipped += 1
                        continue

                    try:
                        data, out_transform = _read_window_for_bounds(ds, cell_wgs84)
                        profile = dict(ds.profile)
                        profile.update(
                            driver="GTiff",
                            height=int(data.shape[1]),
                            width=int(data.shape[2]),
                            transform=out_transform,
                            compress="DEFLATE",
                        )
                        with rasterio.open(out_fp, "w", **profile) as dst:
                            dst.write(data)
                        wrote += 1
                        band_write_counts[band] += 1
                    except Exception as exc:
                        failed += 1
                        print(f"Failed statewide retile {i}_{j} {band}: {exc}")

        summary_lines = [
            f"in_dir={in_dir}",
            f"out_root={out_root}",
            f"year={year}",
            f"month={month}",
            f"month_tag={month_tag}",
            f"pseudo_date={pseudo_date}",
            f"grid_n={grid_n}",
            f"band_count={len(STATEWIDE_REQUIRED_BANDS)}",
            f"wrote={wrote}",
            f"skipped_existing={skipped}",
            f"failed={failed}",
        ]
        for band in STATEWIDE_REQUIRED_BANDS:
            summary_lines.append(f"wrote_{band}={band_write_counts[band]}")
        meta_fp = out_root / f"_statewide_retile_{month_tag}.txt"
        meta_fp.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(
            f"Statewide retile finished. wrote={wrote}, skipped_existing={skipped}, "
            f"failed={failed}. Metadata: {meta_fp}"
        )
    finally:
        for ds in datasets.values():
            try:
                ds.close()
            except Exception:
                pass

    return 0 if failed == 0 else 1


def cmd_fastdiffsr_retile(args) -> int:
    """
    Post-process an existing FastDiffSR month directory:
    mosaic by actual georeferencing, then retile to the canonical Arkansas 20x20 grid.

    This does not re-run FastDiffSR inference and does not modify source tiles unless
    the caller points --out-dir to the same directory and sets --overwrite.
    """
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    grid_n = int(args.grid_n)
    if grid_n <= 0:
        raise SystemExit(f"--grid-n must be > 0, got {grid_n}")
    if not in_dir.exists():
        raise SystemExit(f"input dir not found: {in_dir}")

    src_files = [fp for fp in sorted(in_dir.glob("*.tif")) if re.fullmatch(r"\d+_\d+\.tif", fp.name)]
    if not src_files:
        raise SystemExit(
            f"No single-date FastDiffSR tiles found in {in_dir}. "
            "Expected files like '<i>_<j>.tif'. (date_policy=all retile is not implemented yet.)"
        )

    tile_meta: list[dict] = []
    crs_set: set[str] = set()
    resxs: list[float] = []
    resys: list[float] = []
    ref_profile: dict | None = None

    for fp in src_files:
        try:
            with rasterio.open(fp) as ds:
                if ds.crs is None:
                    continue
                crs_key = str(ds.crs)
                crs_set.add(crs_key)
                b = ds.bounds
                tile_meta.append(
                    {
                        "path": fp,
                        "name": fp.stem,
                        "bounds": (float(b.left), float(b.bottom), float(b.right), float(b.top)),
                        "crs": crs_key,
                        "count": int(ds.count),
                        "dtype": str(ds.dtypes[0]) if ds.dtypes else "uint8",
                        "res": (float(abs(ds.res[0])), float(abs(ds.res[1]))),
                    }
                )
                resxs.append(float(abs(ds.res[0])))
                resys.append(float(abs(ds.res[1])))
                if ref_profile is None:
                    ref_profile = dict(ds.profile)
        except Exception as exc:
            print(f"Warning: skipping unreadable tile {fp}: {exc}")
            continue

    if not tile_meta:
        raise SystemExit(f"No readable georeferenced tiles found in: {in_dir}")
    if len(crs_set) != 1:
        raise SystemExit(f"Source tiles have mixed CRS values: {sorted(crs_set)}")
    if ref_profile is None:
        raise SystemExit("Unable to read source tile profile.")

    target_crs = next(iter(crs_set))
    res_x = float(np.median(np.array(resxs, dtype=np.float64)))
    res_y = float(np.median(np.array(resys, dtype=np.float64)))
    target_res = (res_x, res_y)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Retiling FastDiffSR month dir {in_dir} -> {out_dir} "
        f"to Arkansas {grid_n}x{grid_n} grid | src_tiles={len(tile_meta)} crs={target_crs} res={target_res}"
    )

    wrote = 0
    skipped = 0
    empty = 0
    failed = 0
    written_bounds_wgs84: list[tuple[float, float, float, float]] = []

    for i in range(grid_n):
        for j in range(grid_n):
            tile_id = f"{i}_{j}"
            out_fp = out_dir / f"{tile_id}.tif"
            if out_fp.exists() and not bool(args.overwrite):
                skipped += 1
                continue

            cell_wgs84 = _arkansas_grid_cell_bounds_wgs84(i, j, grid_n=grid_n)
            try:
                cell_crs = transform_bounds("EPSG:4326", target_crs, *cell_wgs84, densify_pts=21)
            except Exception as exc:
                failed += 1
                print(f"Failed bounds transform for {tile_id}: {exc}")
                continue

            candidates = [m for m in tile_meta if _bounds_intersect(a=m["bounds"], b=cell_crs)]
            if not candidates:
                empty += 1
                if bool(args.write_empty):
                    print(f"{tile_id}: no source overlap -> empty tile writing is not implemented yet; skipped")
                continue

            datasets = []
            try:
                for m in candidates:
                    datasets.append(rasterio.open(m["path"]))

                merged, out_transform = merge(
                    datasets,
                    bounds=cell_crs,
                    res=target_res,
                    nodata=0,
                )
                if merged.shape[0] < 3:
                    empty += 1
                    continue

                rgb = merged[:3].astype(np.uint8, copy=False)
                if (not bool(args.write_empty)) and int(np.count_nonzero(rgb)) == 0:
                    empty += 1
                    continue

                profile = dict(ref_profile)
                profile.update(
                    driver="GTiff",
                    count=3,
                    dtype=rasterio.uint8,
                    height=int(rgb.shape[1]),
                    width=int(rgb.shape[2]),
                    transform=out_transform,
                    crs=datasets[0].crs,
                    nodata=0,
                    compress="DEFLATE",
                )
                with rasterio.open(out_fp, "w", **profile) as dst:
                    dst.write(rgb)

                wrote += 1
                try:
                    wb = transform_bounds(str(datasets[0].crs), "EPSG:4326", *cell_crs, densify_pts=21)
                    written_bounds_wgs84.append(tuple(float(x) for x in wb))
                except Exception:
                    pass
            except Exception as exc:
                failed += 1
                print(f"Failed retile {tile_id}: {exc}")
            finally:
                for ds in datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass

    summary_lines = [
        f"in_dir={in_dir}",
        f"out_dir={out_dir}",
        f"grid_n={grid_n}",
        f"source_tile_count={len(tile_meta)}",
        f"target_crs={target_crs}",
        f"target_res={target_res}",
        f"wrote={wrote}",
        f"skipped_existing={skipped}",
        f"empty_targets={empty}",
        f"failed={failed}",
    ]

    if written_bounds_wgs84:
        xs1 = [b[0] for b in written_bounds_wgs84]
        ys1 = [b[1] for b in written_bounds_wgs84]
        xs2 = [b[2] for b in written_bounds_wgs84]
        ys2 = [b[3] for b in written_bounds_wgs84]
        summary_lines.append(
            "written_union_bounds_wgs84="
            f"({min(xs1):.6f},{min(ys1):.6f},{max(xs2):.6f},{max(ys2):.6f})"
        )

    meta_fp = out_dir / "_retile_fastdiffsr.txt"
    meta_fp.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(
        f"FastDiffSR retile finished. wrote={wrote}, skipped_existing={skipped}, "
        f"empty_targets={empty}, failed={failed}. Metadata: {meta_fp}"
    )
    return 0 if failed == 0 else 1


def cmd_fastdiffsr_mosaic(args) -> int:
    """
    Merge an existing FastDiffSR month directory into one georeferenced GeoTIFF.

    Intended for precomputed single-date monthly outputs like '<i>_<j>.tif'.
    """
    in_dir = Path(args.in_dir)
    out_fp = Path(args.out_file)
    if not in_dir.exists():
        raise SystemExit(f"input dir not found: {in_dir}")
    if out_fp.exists() and not bool(args.overwrite):
        raise SystemExit(f"output already exists: {out_fp} (pass --overwrite to replace it)")

    src_files = [fp for fp in sorted(in_dir.glob("*.tif")) if re.fullmatch(r"\d+_\d+\.tif", fp.name)]
    if not src_files:
        raise SystemExit(
            f"No single-date FastDiffSR tiles found in {in_dir}. "
            "Expected files like '<i>_<j>.tif'. (date_policy=all mosaic is not implemented yet.)"
        )

    datasets = []
    tile_count = 0
    failed = 0
    crs_set: set[str] = set()
    resxs: list[float] = []
    resys: list[float] = []
    lefts: list[float] = []
    bottoms: list[float] = []
    rights: list[float] = []
    tops: list[float] = []
    ref_profile: dict | None = None
    try:
        for fp in src_files:
            try:
                ds = rasterio.open(fp)
            except Exception as exc:
                failed += 1
                print(f"Warning: skipping unreadable tile {fp}: {exc}")
                continue
            if ds.crs is None:
                failed += 1
                print(f"Warning: skipping tile without CRS {fp}")
                ds.close()
                continue
            crs_set.add(str(ds.crs))
            resxs.append(float(abs(ds.res[0])))
            resys.append(float(abs(ds.res[1])))
            lefts.append(float(ds.bounds.left))
            bottoms.append(float(ds.bounds.bottom))
            rights.append(float(ds.bounds.right))
            tops.append(float(ds.bounds.top))
            if ref_profile is None:
                ref_profile = dict(ds.profile)
            datasets.append(ds)
            tile_count += 1

        if not datasets:
            raise SystemExit(f"No readable georeferenced FastDiffSR tiles found in: {in_dir}")
        if len(crs_set) != 1:
            raise SystemExit(f"Source tiles have mixed CRS values: {sorted(crs_set)}")
        if ref_profile is None:
            raise SystemExit("Unable to read source tile profile.")

        target_crs = next(iter(crs_set))
        target_res = (
            float(np.median(np.array(resxs, dtype=np.float64))),
            float(np.median(np.array(resys, dtype=np.float64))),
        )
        est_width = int(np.ceil((max(rights) - min(lefts)) / max(target_res[0], 1.0e-12)))
        est_height = int(np.ceil((max(tops) - min(bottoms)) / max(target_res[1], 1.0e-12)))
        est_bytes = est_width * est_height * int(ref_profile.get("count", 3)) * np.dtype(ref_profile.get("dtype", "uint8")).itemsize
        est_gib = float(est_bytes) / float(1024 ** 3)
        print(
            f"Mosaicking FastDiffSR month dir {in_dir} -> {out_fp} "
            f"| src_tiles={tile_count} crs={target_crs} res={target_res} "
            f"estimated_size={est_width}x{est_height} (~{est_gib:.2f} GiB in memory)"
        )
        if est_gib >= 8.0:
            print("Warning: large in-memory merge expected. If this runs out of RAM, mosaic a smaller region or use a larger machine.")

        merged, out_transform = merge(
            datasets,
            res=target_res,
            nodata=int(args.nodata),
        )
        if merged.shape[0] < 3:
            raise SystemExit(f"Mosaic has fewer than 3 bands: shape={merged.shape}")

        out_fp.parent.mkdir(parents=True, exist_ok=True)
        profile = dict(ref_profile)
        profile.update(
            driver="GTiff",
            count=int(merged.shape[0]),
            dtype=merged.dtype,
            height=int(merged.shape[1]),
            width=int(merged.shape[2]),
            transform=out_transform,
            crs=datasets[0].crs,
            nodata=int(args.nodata),
            compress="DEFLATE",
            tiled=True,
            blockxsize=_choose_gtiff_blocksize(int(merged.shape[2])),
            blockysize=_choose_gtiff_blocksize(int(merged.shape[1])),
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(merged)
            if bool(args.build_overviews):
                factors = [f for f in (2, 4, 8, 16, 32) if max(dst.width, dst.height) // f >= 256]
                if factors:
                    dst.build_overviews(factors, Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")

        summary_lines = [
            f"in_dir={in_dir}",
            f"out_file={out_fp}",
            f"source_tile_count={tile_count}",
            f"skipped_or_failed_sources={failed}",
            f"target_crs={target_crs}",
            f"target_res={target_res}",
            f"estimated_width={est_width}",
            f"estimated_height={est_height}",
            f"estimated_uncompressed_gib={est_gib:.4f}",
            f"height={int(merged.shape[1])}",
            f"width={int(merged.shape[2])}",
            f"band_count={int(merged.shape[0])}",
            f"dtype={merged.dtype}",
            f"nodata={int(args.nodata)}",
            f"build_overviews={bool(args.build_overviews)}",
        ]
        meta_fp = out_fp.with_name(f"{out_fp.stem}_mosaic.txt")
        meta_fp.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"FastDiffSR mosaic finished. Output: {out_fp}. Metadata: {meta_fp}")
        return 0
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Arkansas prediction and FastDiffSR utility CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("predict", help="Run batch inference and save GeoTIFF preds for one year/month")
    pr.add_argument("--year", type=int, required=True)
    pr.add_argument(
        "--available-month",
        type=int,
        required=True,
        help="How many months of data are available (1-12). Selects model=min(max(m,6),11).",
    )
    pr.add_argument("--raw-root", required=True, help="Path to the shared <year>_AR Sentinel-2 directory")
    pr.add_argument("--pred-root", required=True, help="Output dir (will write <MM>mo/<meta_patch>.tif)")
    pr.add_argument("--config", default=None)
    pr.add_argument("--checkpoint", default=None)
    pr.add_argument("--device", default="0", help="GPU ids, e.g. '0' or '0,1'")
    pr.add_argument("--batch-size", type=int, default=64)
    pr.add_argument("--tile-size", type=int, default=24)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--meta-patches", nargs="*", default=None, help="Optional subset like '0_0 0_1 ...'")
    pr.set_defaults(func=cmd_predict)

    fd = sub.add_parser("fastdiffsr", help="Run FastDiffSR super-resolution on selected meta-patches and save GeoTIFFs")
    fd.add_argument("--year", type=int, required=True)
    fd.add_argument("--month", type=int, required=True, choices=range(1, 13))
    fd.add_argument("--raw-root", required=True, help="Path to the shared <year>_AR Sentinel-2 directory")
    fd.add_argument("--sr-root", required=True, help="Output root (writes <MM>mo/<meta_patch>.tif)")
    fd.add_argument("--fastdiffsr-config", default=FASTDIFFSR_DEFAULT_CONFIG)
    fd.add_argument(
        "--fastdiffsr-checkpoint",
        required=True,
        help="Checkpoint dir, *_gen.pth/_opt.pth file, or resume-state prefix path.",
    )
    fd.add_argument("--device", default="0", help="GPU ids, e.g. '0' or '0,1'")
    fd.add_argument("--scale", type=int, default=4, help="SR upscale factor (default: 4)")
    fd.add_argument("--lr-tile-size", type=int, default=64, help="LR tile size for strict tiled inference (default: 64)")
    fd.add_argument("--lr-overlap", type=int, default=16, help="LR overlap between adjacent inference tiles in pixels (default: 16)")
    fd.add_argument("--date-policy", choices=["latest", "earliest", "all", "statewide_anchor"], default="latest")
    fd.add_argument("--cloud-thresh-fastdiffsr", type=float, default=0.10, help="Max cloud fraction by SCL mask (0..1).")
    fd.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for meta-patch inference (default: True).",
    )
    fd.add_argument("--force", action="store_true")
    fd.add_argument("--meta-patches", nargs="*", default=None, help="Optional subset like '0_0 0_1 ...'")
    fd.set_defaults(func=cmd_fastdiffsr)

    fi = sub.add_parser(
        "fastdiffsr_rerun_incomplete",
        help="Scan a month output dir for small/missing FastDiffSR tiles and rerun only those meta-patches",
    )
    fi.add_argument("--year", type=int, required=True)
    fi.add_argument("--month", type=int, required=True, choices=range(1, 13))
    fi.add_argument("--raw-root", required=True, help="Path to the shared <year>_AR Sentinel-2 directory")
    fi.add_argument("--sr-root", required=True, help="Output root (expects <MM>mo/<meta_patch>.tif)")
    fi.add_argument("--fastdiffsr-config", default=FASTDIFFSR_DEFAULT_CONFIG)
    fi.add_argument(
        "--fastdiffsr-checkpoint",
        required=True,
        help="Checkpoint dir, *_gen.pth/_opt.pth file, or resume-state prefix path.",
    )
    fi.add_argument("--device", default="0", help="GPU ids, e.g. '0' or '0,1'")
    fi.add_argument("--scale", type=int, default=4, help="SR upscale factor (default: 4)")
    fi.add_argument("--lr-tile-size", type=int, default=64, help="LR tile size for strict tiled inference (default: 64)")
    fi.add_argument("--lr-overlap", type=int, default=16, help="LR overlap between adjacent inference tiles in pixels (default: 16)")
    fi.add_argument("--date-policy", choices=["latest", "earliest", "all", "statewide_anchor"], default="latest")
    fi.add_argument("--cloud-thresh-fastdiffsr", type=float, default=0.10, help="Max cloud fraction by SCL mask (0..1).")
    fi.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars while scanning/rerunning tiles (default: True).",
    )
    fi.add_argument(
        "--min-size-mb",
        type=float,
        default=120.0,
        help="Mark tile as incomplete when file size is smaller than this threshold (default: 120MB).",
    )
    fi.add_argument(
        "--include-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include missing output tiles in rerun candidates (default: True).",
    )
    fi.add_argument("--dry-run", action="store_true", help="Only print candidates; do not rerun inference.")
    fi.add_argument("--show-first", type=int, default=20, help="How many candidate IDs to print in dry-run (default: 20).")
    fi.add_argument("--meta-patches", nargs="*", default=None, help="Optional subset like '0_0 0_1 ...'")
    fi.set_defaults(func=cmd_fastdiffsr_rerun_incomplete)

    fr = sub.add_parser(
        "fastdiffsr_retile",
        help="Post-process existing FastDiffSR month tiles: mosaic by georef and retile to canonical Arkansas grid",
    )
    fr.add_argument("--in-dir", required=True, help="Input month dir containing FastDiffSR tiles like <i>_<j>.tif")
    fr.add_argument("--out-dir", required=True, help="Output month dir for Arkansas-grid re-tiled GeoTIFFs")
    fr.add_argument("--grid-n", type=int, default=AR_GRID_N, help="Arkansas grid size per axis (default: 20)")
    fr.add_argument("--overwrite", action="store_true", help="Overwrite existing output tiles")
    fr.add_argument(
        "--write-empty",
        action="store_true",
        help="Keep empty target tiles (currently skipped if no source overlap or all-zero merge)",
    )
    fr.set_defaults(func=cmd_fastdiffsr_retile)

    fm = sub.add_parser(
        "fastdiffsr_mosaic",
        help="Merge an existing FastDiffSR month directory into one compressed GeoTIFF mosaic",
    )
    fm.add_argument("--in-dir", required=True, help="Input month dir containing FastDiffSR tiles like <i>_<j>.tif")
    fm.add_argument("--out-file", required=True, help="Output mosaic GeoTIFF path")
    fm.add_argument("--nodata", type=int, default=0, help="Nodata value for the merged mosaic (default: 0)")
    fm.add_argument(
        "--build-overviews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build internal overviews for faster windowed reads (default: True)",
    )
    fm.add_argument("--overwrite", action="store_true", help="Overwrite an existing output mosaic")
    fm.set_defaults(func=cmd_fastdiffsr_mosaic)

    sr = sub.add_parser(
        "statewide_retile",
        help="Retile a statewide monthly Arkansas mosaic back into the canonical raw 20x20 grid layout",
    )
    sr.add_argument("--in-dir", required=True, help="Input statewide month dir containing band files like B2_2025-06.tif")
    sr.add_argument("--out-root", required=True, help="Output shared <year>_AR Sentinel-2 directory")
    sr.add_argument("--year", type=int, required=True)
    sr.add_argument("--month", type=int, required=True, choices=range(1, 13))
    sr.add_argument("--grid-n", type=int, default=AR_GRID_N, help="Arkansas grid size per axis (default: 20)")
    sr.add_argument("--month-tag-day", type=int, default=15, help="Pseudo day in output filenames/directories (default: 15)")
    sr.add_argument("--overwrite", action="store_true", help="Overwrite existing output band files")
    sr.set_defaults(func=cmd_statewide_retile)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
