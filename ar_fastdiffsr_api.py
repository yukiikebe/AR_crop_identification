from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

# Make relative paths work whether launched from repo root or DeepSatModels/
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

FASTDIFFSR_DEFAULT_CONFIG = "FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_planet.json"
FASTDIFFSR_DATE_POLICIES = ("latest", "earliest", "all", "statewide_anchor")
FASTDIFFSR_DEFAULT_RAW_ROOT_TEMPLATE = "../../AR_sentinel2/{year}_AR"
FASTDIFFSR_DEFAULT_EE_PROJECT = "satelite-430703"
FASTDIFFSR_DEFAULT_CHECKPOINT = (
    "/home/yuki/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/"
    "sr_fastdiffsr_train_64_256_Planet_260220_163421/checkpoint/I283712_E757"
)
PS_SCENE_DEFAULT_ROOT = "/home/yuki/PSScene"

# Must match Arkansas bounds used by ar_pred_api.py/app_AR_deploy.py
AR_LON_MIN = -94.7610
AR_LON_MAX = -89.5522
AR_LAT_MIN = 32.8376
AR_LAT_MAX = 36.6652
GRID_N = 20

JOB_TERMINAL_STATES = {"completed", "failed", "interrupted"}
JOB_ACTIVE_STATES = {"queued", "running"}


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _read_jsonc(fp: str) -> dict:
    text = ""
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            text += line.split("//")[0] + "\n"
    return json.loads(text)


def _resolve_path(path_like: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path_like).strip())))
    if p.is_absolute():
        return p
    return (APP_DIR / p).resolve()


def _resolve_ps_scene_root() -> Path:
    root = os.environ.get("DEEPSAT_PS_SCENE_ROOT", "").strip() or PS_SCENE_DEFAULT_ROOT
    return _resolve_path(root)


@lru_cache(maxsize=1)
def _load_ps_scene_manifest() -> tuple[dict, ...]:
    root = _resolve_ps_scene_root()
    if not root.exists():
        return tuple()

    out: list[dict] = []
    for meta_fp in sorted(root.glob("*_metadata.json")):
        try:
            obj = json.loads(meta_fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        scene_id = str(obj.get("id") or meta_fp.stem.replace("_metadata", "")).strip()
        tif_fp = root / f"{scene_id}_3B_AnalyticMS_SR_8b_clip.tif"
        if not tif_fp.exists():
            continue

        geom = obj.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if str(geom.get("type")) != "Polygon" or not coords or not isinstance(coords[0], list):
            continue

        ring: list[tuple[float, float]] = []
        for pt in coords[0]:
            if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                ring = []
                break
            ring.append((float(pt[0]), float(pt[1])))
        if len(ring) < 4:
            continue

        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        props = obj.get("properties") or {}
        acquired = str(props.get("acquired") or "").strip()
        out.append(
            {
                "id": scene_id,
                "acquired": acquired,
                "acquired_day": acquired[:10] if len(acquired) >= 10 else None,
                "geometry_lonlat": tuple(ring),
                "lon_min": float(min(lons)),
                "lat_min": float(min(lats)),
                "lon_max": float(max(lons)),
                "lat_max": float(max(lats)),
                "metadata_path": str(meta_fp),
                "raster_path": str(tif_fp),
            }
        )
    return tuple(out)


def _bbox_intersects(
    *,
    a_lon_min: float,
    a_lat_min: float,
    a_lon_max: float,
    a_lat_max: float,
    b_lon_min: float,
    b_lat_min: float,
    b_lon_max: float,
    b_lat_max: float,
) -> bool:
    return not (
        float(a_lon_max) < float(b_lon_min)
        or float(a_lon_min) > float(b_lon_max)
        or float(a_lat_max) < float(b_lat_min)
        or float(a_lat_min) > float(b_lat_max)
    )


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return str(default).strip()


def _require_env(name: str, default: str) -> str:
    value = _env_or_default(name, default)
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing required backend env: {name}")
    return value


def _parse_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=500, detail=f"Invalid boolean env {name}={raw!r}")


def _parse_env_int(name: str, default: int, *, min_v: int, max_v: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        value = int(default)
    else:
        try:
            value = int(raw.strip())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Invalid integer env {name}={raw!r}") from exc
    if value < int(min_v) or value > int(max_v):
        raise HTTPException(
            status_code=500,
            detail=f"Env {name}={value} out of range [{min_v}, {max_v}]",
        )
    return int(value)


def _parse_env_float(name: str, default: float, *, min_v: float, max_v: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        value = float(default)
    else:
        try:
            value = float(raw.strip())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Invalid float env {name}={raw!r}") from exc
    if value < float(min_v) or value > float(max_v):
        raise HTTPException(
            status_code=500,
            detail=f"Env {name}={value} out of range [{min_v}, {max_v}]",
        )
    return float(value)


def _resolve_raw_root_from_template(template: str, *, year: int, month: int) -> Path:
    try:
        rendered = template.format(
            year=int(year),
            month=int(month),
            mm=f"{int(month):02d}",
            month2=f"{int(month):02d}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE. Supported placeholders: "
                "{year}, {month}, {mm}/{month2}."
            ),
        ) from exc
    return _resolve_path(rendered)


def _normalize_date_policy(value: str) -> str:
    v = str(value).strip().lower()
    aliases = {
        "anchor_latest": "statewide_anchor",
        "statewide_anchor_latest": "statewide_anchor",
        "similar_date": "statewide_anchor",
    }
    v = aliases.get(v, v)
    if v not in FASTDIFFSR_DATE_POLICIES:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid DEEPSAT_FASTDIFFSR_DATE_POLICY={value!r}. "
                f"Supported: {list(FASTDIFFSR_DATE_POLICIES)}"
            ),
        )
    return v


def _required_env_status() -> tuple[dict[str, bool], list[str]]:
    required = {
        "DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE": FASTDIFFSR_DEFAULT_RAW_ROOT_TEMPLATE,
        "DEEPSAT_FASTDIFFSR_CHECKPOINT": FASTDIFFSR_DEFAULT_CHECKPOINT,
        "DEEPSAT_FASTDIFFSR_CONFIG": FASTDIFFSR_DEFAULT_CONFIG,
    }
    status = {k: bool(_env_or_default(k, d)) for k, d in required.items()}
    missing = [k for k, ok in status.items() if not ok]
    return status, missing


def _backend_settings_for_info() -> dict:
    required_status, missing = _required_env_status()

    # Keep /info robust even when env values are malformed.
    try:
        scale = _parse_env_int("DEEPSAT_FASTDIFFSR_SCALE", 4, min_v=1, max_v=16)
    except Exception:
        scale = 4
    try:
        lr_tile_size = _parse_env_int("DEEPSAT_FASTDIFFSR_LR_TILE_SIZE", 64, min_v=1, max_v=1024)
    except Exception:
        lr_tile_size = 64
    try:
        cloud = _parse_env_float("DEEPSAT_FASTDIFFSR_CLOUD_THRESH", 0.10, min_v=0.0, max_v=1.0)
    except Exception:
        cloud = 0.10
    try:
        blank_frac_max = _parse_env_float("DEEPSAT_FASTDIFFSR_BLANK_FRAC_MAX", 0.98, min_v=0.0, max_v=1.0)
    except Exception:
        blank_frac_max = 0.98
    try:
        date_policy = _normalize_date_policy(os.environ.get("DEEPSAT_FASTDIFFSR_DATE_POLICY", "latest"))
    except Exception:
        date_policy = "latest"
    try:
        dl_workers = _parse_env_int("DEEPSAT_FASTDIFFSR_DOWNLOAD_WORKERS", 4, min_v=1, max_v=128)
    except Exception:
        dl_workers = 4
    try:
        dl_cloud = _parse_env_float("DEEPSAT_FASTDIFFSR_DOWNLOAD_CLOUD_THRESH", 20.0, min_v=0.0, max_v=100.0)
    except Exception:
        dl_cloud = 20.0
    try:
        dl_verify = _parse_env_bool("DEEPSAT_FASTDIFFSR_DOWNLOAD_VERIFY", True)
    except Exception:
        dl_verify = True
    try:
        dl_strict = _parse_env_bool("DEEPSAT_FASTDIFFSR_DOWNLOAD_STRICT", True)
    except Exception:
        dl_strict = True
    try:
        dl_fast_skip = _parse_env_bool("DEEPSAT_FASTDIFFSR_DOWNLOAD_FAST_SKIP", True)
    except Exception:
        dl_fast_skip = True

    return {
        "ready": len(missing) == 0,
        "required_env": required_status,
        "missing_required_env": missing,
        "request_schema": {
            "strict": True,
            "required": ["year", "month", "bbox"],
            "bbox": ["lon_min", "lat_min", "lon_max", "lat_max"],
        },
        "fixed_defaults": {
            "raw_root_template": _env_or_default("DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE", FASTDIFFSR_DEFAULT_RAW_ROOT_TEMPLATE),
            "output_root": os.environ.get("DEEPSAT_FASTDIFFSR_OUT_ROOT", "./fastdiffsr_preds"),
            "mosaic_root": os.environ.get("DEEPSAT_FASTDIFFSR_MOSAIC_ROOT", ""),
            "ps_scene_root": str(_resolve_ps_scene_root()),
            "checkpoint_path": _env_or_default("DEEPSAT_FASTDIFFSR_CHECKPOINT", FASTDIFFSR_DEFAULT_CHECKPOINT),
            "config_path": _env_or_default("DEEPSAT_FASTDIFFSR_CONFIG", FASTDIFFSR_DEFAULT_CONFIG),
            "device": os.environ.get("DEEPSAT_FASTDIFFSR_DEVICE", "0"),
            "scale": scale,
            "lr_tile_size": lr_tile_size,
            "date_policy": date_policy,
            "cloud_thresh_fastdiffsr": cloud,
            "blank_frac_max": blank_frac_max,
            "force": str(os.environ.get("DEEPSAT_FASTDIFFSR_FORCE", "false")).strip().lower() in {"1", "true", "yes", "on"},
            "preview_max_side": 4096,
            "preview_stretch": str(os.environ.get("DEEPSAT_FASTDIFFSR_PREVIEW_STRETCH", "true")).strip().lower() in {"1", "true", "yes", "on"},
            "preview_swap_rb": str(os.environ.get("DEEPSAT_FASTDIFFSR_PREVIEW_SWAP_RB", "false")).strip().lower() in {"1", "true", "yes", "on"},
            "output_layout": "<SR_ROOT>/AR_{year}_fastdiffsr/<MM>mo/<meta_patch>.tif",
            "mosaic_layout": "<MOSAIC_ROOT>/AR_{year}_fastdiffsr/{year}_{month:02d}.tif",
            "jobs_layout": "<SR_ROOT>/_jobs/<job_id>.json",
        },
        "auto_download": {
            "enabled": True,
            "ee_project": os.environ.get("DEEPSAT_FASTDIFFSR_EE_PROJECT", FASTDIFFSR_DEFAULT_EE_PROJECT),
            "download_data_root": os.environ.get("DEEPSAT_FASTDIFFSR_DOWNLOAD_DATA_ROOT", ""),
            "download_workers": dl_workers,
            "download_cloud_thresh": dl_cloud,
            "download_verify": dl_verify,
            "download_strict": dl_strict,
            "download_fast_skip": dl_fast_skip,
        },
        "capabilities": {
            "merged_preview_png": True,
            "preview_single_date_only": True,
            "preview_display_stretch": True,
            "preview_optional_swap_rb": True,
            "preview_month_mosaic_preferred": True,
            "planet_reference_preview": True,
        },
        # Backward-compatible keys used by older client checks.
        "default_config": _env_or_default("DEEPSAT_FASTDIFFSR_CONFIG", FASTDIFFSR_DEFAULT_CONFIG),
        "date_policy": list(FASTDIFFSR_DATE_POLICIES),
    }


def _load_backend_settings(*, year: int, month: int) -> dict:
    raw_root_template = _require_env("DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE", FASTDIFFSR_DEFAULT_RAW_ROOT_TEMPLATE)
    checkpoint_env = _require_env("DEEPSAT_FASTDIFFSR_CHECKPOINT", FASTDIFFSR_DEFAULT_CHECKPOINT)
    config_env = _require_env("DEEPSAT_FASTDIFFSR_CONFIG", FASTDIFFSR_DEFAULT_CONFIG)

    out_root = _resolve_path(os.environ.get("DEEPSAT_FASTDIFFSR_OUT_ROOT", "./fastdiffsr_preds"))
    out_dir = out_root / f"AR_{int(year)}_fastdiffsr" / f"{int(month):02d}mo"
    mosaic_root_env = str(os.environ.get("DEEPSAT_FASTDIFFSR_MOSAIC_ROOT", "")).strip()
    mosaic_root = _resolve_path(mosaic_root_env) if mosaic_root_env else None
    mosaic_path = None
    if mosaic_root is not None:
        mosaic_path = mosaic_root / f"AR_{int(year)}_fastdiffsr" / f"{int(year)}_{int(month):02d}.tif"

    settings = {
        "raw_root": _resolve_raw_root_from_template(raw_root_template, year=int(year), month=int(month)),
        "raw_root_template": raw_root_template,
        "out_root": out_root,
        "out_dir": out_dir,
        "mosaic_root": mosaic_root,
        "mosaic_path": mosaic_path,
        "checkpoint_path": _resolve_path(checkpoint_env),
        "config_path": _resolve_path(config_env),
        "device": _normalize_device_key(os.environ.get("DEEPSAT_FASTDIFFSR_DEVICE", "0")),
        "scale": _parse_env_int("DEEPSAT_FASTDIFFSR_SCALE", 4, min_v=1, max_v=16),
        "lr_tile_size": _parse_env_int("DEEPSAT_FASTDIFFSR_LR_TILE_SIZE", 64, min_v=1, max_v=1024),
        "date_policy": _normalize_date_policy(os.environ.get("DEEPSAT_FASTDIFFSR_DATE_POLICY", "latest")),
        "cloud_thresh_fastdiffsr": _parse_env_float("DEEPSAT_FASTDIFFSR_CLOUD_THRESH", 0.10, min_v=0.0, max_v=1.0),
        "blank_frac_max": _parse_env_float("DEEPSAT_FASTDIFFSR_BLANK_FRAC_MAX", 0.98, min_v=0.0, max_v=1.0),
        "force": _parse_env_bool("DEEPSAT_FASTDIFFSR_FORCE", False),
        "ee_project": str(os.environ.get("DEEPSAT_FASTDIFFSR_EE_PROJECT", FASTDIFFSR_DEFAULT_EE_PROJECT)).strip() or FASTDIFFSR_DEFAULT_EE_PROJECT,
        "download_data_root": None,
        "download_workers": _parse_env_int("DEEPSAT_FASTDIFFSR_DOWNLOAD_WORKERS", 4, min_v=1, max_v=128),
        "download_cloud_thresh": _parse_env_float("DEEPSAT_FASTDIFFSR_DOWNLOAD_CLOUD_THRESH", 20.0, min_v=0.0, max_v=100.0),
        "download_verify": _parse_env_bool("DEEPSAT_FASTDIFFSR_DOWNLOAD_VERIFY", True),
        "download_strict": _parse_env_bool("DEEPSAT_FASTDIFFSR_DOWNLOAD_STRICT", True),
        "download_fast_skip": _parse_env_bool("DEEPSAT_FASTDIFFSR_DOWNLOAD_FAST_SKIP", True),
    }
    dl_root_env = str(os.environ.get("DEEPSAT_FASTDIFFSR_DOWNLOAD_DATA_ROOT", "")).strip()
    if dl_root_env:
        settings["download_data_root"] = _resolve_path(dl_root_env)
    else:
        settings["download_data_root"] = Path(settings["raw_root"]).parent
    return settings


def _resolve_fastdiffsr_resume_state(checkpoint_path: str) -> str:
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


def _build_fastdiffsr_model(config_path: str, resume_state: str, device_ids: list) -> object:
    fastdiffsr_root = APP_DIR / "FastDiffSR" / "FastDiffSR"
    if not fastdiffsr_root.exists():
        raise SystemExit(f"FastDiffSR code root not found: {fastdiffsr_root}")

    if str(fastdiffsr_root) not in sys.path:
        sys.path.insert(0, str(fastdiffsr_root))

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


def _read_rgb10m_image(
    meta_dir: Path,
    day: date,
    cloud_frac_max: float = 0.10,
    blank_frac_max: float | None = None,
):
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

    if blank_frac_max is not None:
        zero_mask = np.logical_and(np.logical_and(b2 == 0, b3 == 0), b4 == 0)
        blank_frac = float(zero_mask.sum()) / float(zero_mask.size)
        if blank_frac > float(blank_frac_max):
            return None

    rgb = np.dstack([b4, b3, b2]).astype(np.float32)
    return rgb, b2_fp


def _stretch_to_uint8(rgb: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
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


def _pad_rgb_reflect_or_edge(tile: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = tile.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile
    mode = "reflect" if (h > 1 and w > 1) else "edge"
    return np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)], mode=mode)


def _infer_fastdiffsr_rgb(diffusion, rgb_lr: np.ndarray, scale: int, lr_tile_size: int = 64) -> np.ndarray:
    if lr_tile_size <= 0:
        raise SystemExit(f"lr_tile_size must be > 0, got {lr_tile_size}")
    if scale <= 0:
        raise SystemExit(f"scale must be > 0, got {scale}")

    h, w, _ = rgb_lr.shape
    rgb8 = _stretch_to_uint8(rgb_lr)
    sr_mosaic = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    sr_tile_size = lr_tile_size * scale

    y_positions = list(range(0, h, lr_tile_size))
    x_positions = list(range(0, w, lr_tile_size))

    for iy in y_positions:
        for ix in x_positions:
            tile_lr = rgb8[iy : iy + lr_tile_size, ix : ix + lr_tile_size, :]
            vh, vw = tile_lr.shape[:2]
            tile_lr = _pad_rgb_reflect_or_edge(tile_lr, target_h=lr_tile_size, target_w=lr_tile_size)
            tile_sr_in = cv2.resize(tile_lr, (sr_tile_size, sr_tile_size), interpolation=cv2.INTER_CUBIC)

            sr_t = torch.from_numpy(np.transpose(tile_sr_in, (2, 0, 1))).float() / 255.0
            sr_t = sr_t * 2.0 - 1.0
            sr_t = sr_t.unsqueeze(0)

            batch = {"SR": sr_t, "HR": sr_t.clone()}
            diffusion.feed_data(batch)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals(need_LR=False)

            sr_tile = visuals["SR"].squeeze(0).clamp(-1, 1).permute(1, 2, 0).numpy()
            sr_tile = np.clip(((sr_tile + 1.0) * 127.5).round(), 0, 255).astype(np.uint8)
            sr_tile = sr_tile[: vh * scale, : vw * scale, :]

            oy = iy * scale
            ox = ix * scale
            sr_mosaic[oy : oy + (vh * scale), ox : ox + (vw * scale), :] = sr_tile

    return sr_mosaic


def _write_rgb_geotiff(out_fp: Path, rgb: np.ndarray, template_fp: Path, scale: int):
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


def _sr_tile_meta_fp(out_fp: Path) -> Path:
    return out_fp.with_suffix(".meta.json")


def _write_sr_tile_metadata(*, out_fp: Path, meta_patch: str, used_date: date, template_fp: Path) -> None:
    payload = {
        "meta_patch": str(meta_patch),
        "date": used_date.strftime("%Y-%m-%d"),
        "template_path": str(template_fp),
    }
    _write_json_atomic(_sr_tile_meta_fp(out_fp), payload)


def _read_sr_tile_metadata(out_fp: Path) -> Optional[dict]:
    fp = _sr_tile_meta_fp(out_fp)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rgb_png_base64(rgb: np.ndarray) -> str:
    rgb8 = np.asarray(rgb, dtype=np.uint8)
    ok, enc = cv2.imencode(".png", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode SR preview PNG.")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def _downsample_preview_rgb(rgb: np.ndarray, *, max_side: int = 4096) -> np.ndarray:
    """
    Resize large preview images for web display only.
    Does not affect saved GeoTIFF outputs.
    """
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= int(max_side):
        return rgb
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _stretch_preview_rgb_for_web(
    rgb: np.ndarray,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
    ignore_zeros: bool = True,
) -> np.ndarray:
    """
    Display-only contrast stretch for web previews.
    Keeps saved GeoTIFF values unchanged.
    """
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr

    out = arr.copy()
    if bool(ignore_zeros):
        valid_mask = np.any(arr[..., :3] > 0, axis=2)
    else:
        valid_mask = np.ones(arr.shape[:2], dtype=bool)

    if int(valid_mask.sum()) < 16:
        return out

    for c in range(3):
        ch = arr[..., c].astype(np.float32)
        vals = ch[valid_mask]
        if vals.size < 16:
            continue
        lo = float(np.percentile(vals, p_low))
        hi = float(np.percentile(vals, p_high))
        if hi <= lo + 1e-6:
            continue
        stretched = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        ch_out = (stretched * 255.0).astype(np.uint8)
        if bool(ignore_zeros):
            ch_out = np.where(valid_mask, ch_out, 0)
        out[..., c] = ch_out
    return out


def _prepare_preview_rgb_for_web(rgb: np.ndarray) -> np.ndarray:
    """
    Web-preview-only post-processing. Does not affect saved GeoTIFFs.
    """
    arr = np.asarray(rgb, dtype=np.uint8)

    # Optional channel swap to test/handle model outputs that appear BGR-like in preview.
    if _parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_SWAP_RB", False):
        arr = arr[..., ::-1].copy()

    # Enable display stretch by default so very dark but valid tiles are visible in the UI.
    if _parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_STRETCH", True):
        arr = _stretch_preview_rgb_for_web(arr, p_low=2.0, p_high=98.0, ignore_zeros=True)

    return arr


def _preview_rgb_debug_stats(rgb: np.ndarray) -> dict:
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "error": "expected HxWxC array",
        }

    stats = {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "pixel_count": int(arr.shape[0] * arr.shape[1]),
    }

    try:
        valid_any = np.any(arr[..., : min(3, arr.shape[2])] > 0, axis=2)
        stats["nonzero_any_rgb_pixels"] = int(valid_any.sum())
        stats["nonzero_any_rgb_fraction"] = float(valid_any.mean())
    except Exception:
        pass

    ch_stats = []
    for c in range(min(3, arr.shape[2])):
        ch = arr[..., c]
        ch_stats.append(
            {
                "channel": int(c),
                "min": int(np.min(ch)),
                "max": int(np.max(ch)),
                "mean": float(np.mean(ch)),
                "nonzero_pixels": int(np.count_nonzero(ch)),
            }
        )
    stats["channels"] = ch_stats
    return stats


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _month_range(year: int, month: int) -> Tuple[str, str]:
    start = date(year, month, 1)
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    end = next_month.toordinal() - 1
    end_d = date.fromordinal(end)
    return start.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")


def _meta_patches_intersecting_bbox(*, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> List[str]:
    lon_range = np.linspace(AR_LON_MIN, AR_LON_MAX, GRID_N + 1)
    lat_range = np.linspace(AR_LAT_MIN, AR_LAT_MAX, GRID_N + 1)
    out: List[str] = []
    for i in range(GRID_N):
        cell_lon1, cell_lon2 = lon_range[i], lon_range[i + 1]
        for j in range(GRID_N):
            cell_lat1, cell_lat2 = lat_range[j], lat_range[j + 1]
            intersects = (
                (cell_lon1 <= lon_max)
                and (cell_lon2 >= lon_min)
                and (cell_lat1 <= lat_max)
                and (cell_lat2 >= lat_min)
            )
            if intersects:
                out.append(f"{i}_{j}")
    return out


def _all_meta_patches() -> List[str]:
    return [f"{i}_{j}" for i in range(GRID_N) for j in range(GRID_N)]


def _list_dates(meta_dir: Path, *, start_day: str, end_day: str) -> List[date]:
    start = _parse_ymd(start_day)
    end = _parse_ymd(end_day)
    out: List[date] = []
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
    blank_frac_max: float | None = None,
) -> List[date]:
    valid: List[date] = []
    for d in _list_dates(meta_dir, start_day=start_day, end_day=end_day):
        try:
            got = _read_rgb10m_image(
                meta_dir,
                d,
                cloud_frac_max=float(cloud_frac_max),
                blank_frac_max=blank_frac_max,
            )
        except Exception:
            got = None
        if got is not None:
            valid.append(d)
    return valid


def _plan_statewide_anchor_dates(
    *,
    raw_root: Path,
    meta_patches: List[str],
    start_day: str,
    end_day: str,
    cloud_frac_max: float,
    blank_frac_max: float | None = None,
) -> tuple[Dict[str, date], dict]:
    """
    Pick one month-wide anchor date (latest date with maximum valid tile coverage), then
    assign each tile the anchor date if available, otherwise the nearest valid in-month date.
    """
    valid_dates_by_mp: Dict[str, List[date]] = {}
    date_counts: Dict[date, int] = {}
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
            blank_frac_max=blank_frac_max,
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
    plan: Dict[str, date] = {}
    anchor_assigned = 0
    fallback_assigned = 0
    gap_hist: Dict[str, int] = {}

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


def _normalize_bbox(*, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> tuple[float, float, float, float]:
    lo_min = min(float(lon_min), float(lon_max))
    lo_max = max(float(lon_min), float(lon_max))
    la_min = min(float(lat_min), float(lat_max))
    la_max = max(float(lat_min), float(lat_max))
    return lo_min, la_min, lo_max, la_max


def _existing_outputs_for_meta_patch(*, out_dir: Path, meta_patch: str, date_policy: str) -> List[Path]:
    if str(date_policy) == "all":
        return sorted(out_dir.glob(f"{meta_patch}_*.tif"))
    fp = out_dir / f"{meta_patch}.tif"
    return [fp] if fp.exists() else []


def _requested_output_status(*, out_dir: Path, meta_patches: List[str], date_policy: str) -> tuple[List[str], List[str]]:
    ready: List[str] = []
    missing: List[str] = []
    for mp in meta_patches:
        existing = _existing_outputs_for_meta_patch(out_dir=out_dir, meta_patch=mp, date_policy=date_policy)
        if existing:
            ready.append(mp)
        else:
            missing.append(mp)
    return ready, missing


def _bounds_intersect_xyxy(
    *,
    left: float,
    bottom: float,
    right: float,
    top: float,
    q_left: float,
    q_bottom: float,
    q_right: float,
    q_top: float,
) -> bool:
    return (left <= q_right) and (right >= q_left) and (bottom <= q_top) and (top >= q_bottom)


def _existing_output_tiles_intersecting_bbox(
    *,
    out_dir: Path,
    bbox_wgs84: tuple[float, float, float, float],
    date_policy: str,
) -> List[str]:
    """
    Select existing SR tiles by actual GeoTIFF bounds instead of filename grid convention.
    This makes preview selection robust even if meta-patch naming order differs from the
    theoretical 20x20 bbox->grid mapping used elsewhere.
    """
    if str(date_policy) == "all":
        return []
    if not out_dir.exists():
        return []

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    bbox_by_crs: dict[str, tuple[float, float, float, float]] = {}
    selected: List[str] = []

    for fp in sorted(out_dir.glob("*.tif")):
        # For single-date policies we expect names like "<i>_<j>.tif"; ignore any other tif.
        if not re.fullmatch(r"\d+_\d+\.tif", fp.name):
            continue
        try:
            with rasterio.open(fp) as ds:
                if ds.crs is None:
                    continue
                crs_key = str(ds.crs)
                if crs_key not in bbox_by_crs:
                    bbox_by_crs[crs_key] = transform_bounds(
                        "EPSG:4326",
                        ds.crs,
                        float(lon_min),
                        float(lat_min),
                        float(lon_max),
                        float(lat_max),
                        densify_pts=21,
                    )
                q_left, q_bottom, q_right, q_top = bbox_by_crs[crs_key]
                b = ds.bounds
                if _bounds_intersect_xyxy(
                    left=float(b.left),
                    bottom=float(b.bottom),
                    right=float(b.right),
                    top=float(b.top),
                    q_left=float(q_left),
                    q_bottom=float(q_bottom),
                    q_right=float(q_right),
                    q_top=float(q_top),
                ):
                    selected.append(fp.stem)
        except Exception:
            continue

    return selected


def _date_from_all_policy_filename(fp: Path) -> Optional[str]:
    m = re.match(r".+_(\d{8})$", fp.stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return None


def _job_settings_signature(settings: dict) -> dict:
    return {
        "raw_root_template": str(settings["raw_root_template"]),
        "out_root": str(settings["out_root"]),
        "config_path": str(settings["config_path"]),
        "checkpoint_path": str(settings["checkpoint_path"]),
        "device": str(settings["device"]),
        "scale": int(settings["scale"]),
        "lr_tile_size": int(settings["lr_tile_size"]),
        "date_policy": str(settings["date_policy"]),
        "cloud_thresh_fastdiffsr": float(settings["cloud_thresh_fastdiffsr"]),
        "blank_frac_max": float(settings.get("blank_frac_max", 0.98)),
        "force": bool(settings["force"]),
        "ee_project": str(settings.get("ee_project", FASTDIFFSR_DEFAULT_EE_PROJECT)),
        "download_data_root": str(settings.get("download_data_root", "")),
        "download_workers": int(settings.get("download_workers", 4)),
        "download_cloud_thresh": float(settings.get("download_cloud_thresh", 20.0)),
        "download_verify": bool(settings.get("download_verify", True)),
        "download_strict": bool(settings.get("download_strict", True)),
        "download_fast_skip": bool(settings.get("download_fast_skip", True)),
    }


def _job_key(*, year: int, month: int, meta_patches: List[str], settings: dict) -> str:
    payload = {
        "year": int(year),
        "month": int(month),
        "meta_patches": sorted(meta_patches),
        "settings": _job_settings_signature(settings),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _month_seed_job_key(*, year: int, month: int, settings: dict) -> str:
    payload = {
        "year": int(year),
        "month": int(month),
        "job_kind": "month_seed",
        "job_scope": "full_month",
        "settings": _job_settings_signature(settings),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _jobs_dir(out_root: Path) -> Path:
    return out_root / "_jobs"


def _job_fp(jobs_dir: Path, job_id: str) -> Path:
    return jobs_dir / f"{job_id}.json"


def _write_json_atomic(fp: Path, payload: dict) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(fp)


def _read_job_file(jobs_dir: Path, job_id: str) -> dict:
    fp = _job_fp(jobs_dir, job_id)
    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"job_id not found: {job_id}")
    return json.loads(fp.read_text(encoding="utf-8"))


def _status_url(job_id: str) -> str:
    return f"/jobs/{job_id}"


def _set_job_phase(job: dict, *, phase: str, message: Optional[str] = None) -> None:
    job["phase"] = str(phase)
    if message is not None:
        job["phase_message"] = str(message)
    _update_job(job)


def _find_latest_job_by_key(*, out_root: Path, job_key: str) -> Optional[dict]:
    jobs_dir = _jobs_dir(out_root)
    if not jobs_dir.exists():
        return None

    latest: Optional[dict] = None
    latest_ts = ""
    for fp in sorted(jobs_dir.glob("*.json")):
        try:
            job = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(job.get("job_key")) != str(job_key):
            continue
        ts = str(job.get("updated_at") or job.get("created_at") or "")
        if latest is None or ts >= latest_ts:
            latest = job
            latest_ts = ts
    return latest


def _collect_existing_outputs(*, out_dir: Path, meta_patches: List[str], date_policy: str) -> List[FastDiffSROutput]:
    outputs: List[FastDiffSROutput] = []
    for mp in meta_patches:
        existing = _existing_outputs_for_meta_patch(out_dir=out_dir, meta_patch=mp, date_policy=date_policy)
        for fp in existing:
            meta = _read_sr_tile_metadata(fp)
            used_date = None
            if str(date_policy) == "all":
                used_date = _date_from_all_policy_filename(fp)
            elif isinstance(meta, dict):
                used_date = str(meta.get("date") or "") or None
            outputs.append(
                FastDiffSROutput(
                    meta_patch=mp,
                    date=used_date,
                    status="skipped_existing",
                    path=str(fp),
                )
            )
    return outputs


def _month_mosaic_intersects_bbox(*, mosaic_fp: Path, bbox_wgs84: tuple[float, float, float, float]) -> bool:
    if not mosaic_fp.exists():
        return False
    try:
        with rasterio.open(mosaic_fp) as ds:
            if ds.crs is None:
                return False
            q_left, q_bottom, q_right, q_top = transform_bounds(
                "EPSG:4326",
                ds.crs,
                float(bbox_wgs84[0]),
                float(bbox_wgs84[1]),
                float(bbox_wgs84[2]),
                float(bbox_wgs84[3]),
                densify_pts=21,
            )
            return not (
                float(ds.bounds.right) < q_left
                or float(ds.bounds.left) > q_right
                or float(ds.bounds.top) < q_bottom
                or float(ds.bounds.bottom) > q_top
            )
    except Exception:
        return False


def _finalize_fastdiffsr_preview(
    *,
    rgb: np.ndarray,
    out_transform,
    crs,
    bbox: tuple[float, float, float, float],
    meta_patches: List[str],
    date_policy: str,
    preview_tile_count: int,
    selected_tile_bounds: list[dict],
    source: str,
    source_path: Optional[Path] = None,
    merge_bounds_output_crs: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    rgb = np.asarray(rgb, dtype=np.uint8)
    rgb_full = rgb
    out_transform_full = out_transform
    valid_mask = np.any(rgb_full > 0, axis=2)
    valid_crop = None
    if bool(valid_mask.any()):
        ys, xs = np.where(valid_mask)
        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1
        if r1 > r0 and c1 > c0:
            rgb = rgb_full[r0:r1, c0:c1, :]
            out_transform = out_transform_full * rasterio.Affine.translation(c0, r0)
            valid_crop = {"row0": r0, "row1": r1, "col0": c0, "col1": c1}

    lon_min, lat_min, lon_max, lat_max = bbox
    preview_swap_rb = _parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_SWAP_RB", False)
    preview_stretch = _parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_STRETCH", True)
    debug = {
        "source": str(source),
        "source_path": str(source_path) if source_path is not None else None,
        "date_policy": str(date_policy),
        "tile_count": int(preview_tile_count),
        "meta_patches": [str(x) for x in meta_patches],
        "bbox_wgs84": {
            "lon_min": float(lon_min),
            "lat_min": float(lat_min),
            "lon_max": float(lon_max),
            "lat_max": float(lat_max),
        },
        "selected_tile_bounds": selected_tile_bounds,
        "cropped_to_valid_sr_extent": bool(valid_crop is not None),
        "valid_crop_pixels": valid_crop,
        "preview_max_side": 4096,
        "preview_stretch": bool(preview_stretch),
        "preview_swap_rb": bool(preview_swap_rb),
        "merged_rgb_stats_before_downsample": _preview_rgb_debug_stats(rgb),
    }
    debug["merged_rgb_stats_before_downsample_full_bbox"] = _preview_rgb_debug_stats(rgb_full)
    sr_native_h = int(rgb.shape[0])
    sr_native_w = int(rgb.shape[1])
    rgb_preview = _downsample_preview_rgb(rgb, max_side=4096)
    debug["preview_rgb_stats_after_downsample"] = _preview_rgb_debug_stats(rgb_preview)
    rgb_preview = _prepare_preview_rgb_for_web(rgb_preview)
    debug["preview_rgb_stats_final"] = _preview_rgb_debug_stats(rgb_preview)
    debug["preview_native_size"] = {"width": sr_native_w, "height": sr_native_h}
    debug["preview_display_size"] = {"width": int(rgb_preview.shape[1]), "height": int(rgb_preview.shape[0])}
    debug["preview_was_downsampled"] = bool(
        int(rgb_preview.shape[0]) != sr_native_h or int(rgb_preview.shape[1]) != sr_native_w
    )
    compare_bounds = array_bounds(int(rgb.shape[0]), int(rgb.shape[1]), out_transform)
    debug["merge_bounds_output_crs"] = [
        float(x) for x in (merge_bounds_output_crs if merge_bounds_output_crs is not None else compare_bounds)
    ]
    return {
        "preview_png_base64": _rgb_png_base64(rgb_preview),
        "preview_width": int(rgb_preview.shape[1]),
        "preview_height": int(rgb_preview.shape[0]),
        "preview_native_width": sr_native_w,
        "preview_native_height": sr_native_h,
        "preview_was_downsampled": bool(
            int(rgb_preview.shape[0]) != sr_native_h or int(rgb_preview.shape[1]) != sr_native_w
        ),
        "preview_crs": str(crs),
        "preview_transform_gdal": tuple(float(x) for x in out_transform.to_gdal()),
        "preview_tile_count": int(preview_tile_count),
        "preview_debug": debug,
        "_compare_grid": {
            "crs": str(crs),
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "transform_gdal": tuple(float(x) for x in out_transform.to_gdal()),
            "bounds_output_crs": [float(x) for x in compare_bounds],
        },
        "_preview_rgb_native": rgb,
        "_preview_out_transform": out_transform,
        "_preview_crs_obj": crs,
    }


def _build_fastdiffsr_preview_from_mosaic(
    *,
    mosaic_fp: Path,
    bbox: tuple[float, float, float, float],
    meta_patches: List[str],
    date_policy: str,
) -> dict:
    with rasterio.open(mosaic_fp) as ds:
        crs = ds.crs
        if crs is None:
            raise HTTPException(status_code=500, detail="FastDiffSR mosaic is missing CRS.")
        if int(ds.count) < 3:
            raise HTTPException(status_code=500, detail="FastDiffSR mosaic must have at least 3 bands.")

        lon_min, lat_min, lon_max, lat_max = bbox
        bounds_crs = transform_bounds("EPSG:4326", crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21)
        window = window_from_bounds(*bounds_crs, transform=ds.transform)
        data = ds.read(indexes=[1, 2, 3], window=window, boundless=True, fill_value=0)
        if int(data.shape[1]) <= 0 or int(data.shape[2]) <= 0:
            raise HTTPException(status_code=404, detail="No FastDiffSR mosaic pixels found for the requested bbox.")
        out_transform = ds.window_transform(window)
        rgb = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        return _finalize_fastdiffsr_preview(
            rgb=rgb,
            out_transform=out_transform,
            crs=crs,
            bbox=bbox,
            meta_patches=meta_patches,
            date_policy=date_policy,
            preview_tile_count=1,
            selected_tile_bounds=[
                {
                    "meta_patch": "month_mosaic",
                    "date": None,
                    "bounds": [
                        float(ds.bounds.left),
                        float(ds.bounds.bottom),
                        float(ds.bounds.right),
                        float(ds.bounds.top),
                    ],
                    "crs": str(crs),
                    "path": str(mosaic_fp),
                }
            ],
            source="month_mosaic",
            source_path=mosaic_fp,
            merge_bounds_output_crs=tuple(float(x) for x in bounds_crs),
        )


def _build_fastdiffsr_preview_for_bbox(
    *,
    out_dir: Path,
    bbox: tuple[float, float, float, float],
    meta_patches: List[str],
    date_policy: str,
    mosaic_fp: Optional[Path] = None,
) -> dict:
    if str(date_policy) == "all":
        return {
            "preview_png_base64": None,
            "preview_width": None,
            "preview_height": None,
            "preview_crs": None,
            "preview_transform_gdal": None,
            "preview_tile_count": 0,
            "preview_debug": {
                "reason": "date_policy=all preview disabled",
                "date_policy": str(date_policy),
            },
        }

    if mosaic_fp is not None and mosaic_fp.exists():
        return _build_fastdiffsr_preview_from_mosaic(
            mosaic_fp=mosaic_fp,
            bbox=bbox,
            meta_patches=meta_patches,
            date_policy=date_policy,
        )

    datasets = []
    opened = []
    selected_tile_bounds = []
    try:
        for mp in meta_patches:
            fp = out_dir / f"{mp}.tif"
            if not fp.exists():
                continue
            ds = rasterio.open(fp)
            opened.append(ds)
            datasets.append(ds)
            meta = _read_sr_tile_metadata(fp)
            selected_tile_bounds.append(
                {
                    "meta_patch": str(mp),
                    "date": str(meta.get("date")) if isinstance(meta, dict) and meta.get("date") else None,
                    "bounds": [
                        float(ds.bounds.left),
                        float(ds.bounds.bottom),
                        float(ds.bounds.right),
                        float(ds.bounds.top),
                    ],
                    "crs": str(ds.crs) if ds.crs is not None else None,
                }
            )

        if not datasets:
            raise HTTPException(status_code=404, detail="No FastDiffSR tiles found for the requested bbox.")

        crs = datasets[0].crs
        if crs is None:
            raise HTTPException(status_code=500, detail="FastDiffSR outputs missing CRS; re-generate with georeferencing.")

        crs_set = {str(ds.crs) for ds in datasets}
        if len(crs_set) != 1:
            raise HTTPException(status_code=500, detail=f"FastDiffSR output tiles have mixed CRS values: {sorted(crs_set)}")

        lon_min, lat_min, lon_max, lat_max = bbox
        bounds_crs = transform_bounds("EPSG:4326", crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21)
        merged, out_transform = merge(datasets, bounds=bounds_crs)
        rgb = np.transpose(merged[:3], (1, 2, 0)).astype(np.uint8)
        return _finalize_fastdiffsr_preview(
            rgb=rgb,
            out_transform=out_transform,
            crs=crs,
            bbox=bbox,
            meta_patches=meta_patches,
            date_policy=date_policy,
            preview_tile_count=len(datasets),
            selected_tile_bounds=selected_tile_bounds,
            source="month_tiles",
            merge_bounds_output_crs=tuple(float(x) for x in bounds_crs),
        )
    finally:
        for ds in opened:
            try:
                ds.close()
            except Exception:
                pass


def _select_raw_preview_tile_sources(
    *,
    raw_root: Path,
    meta_patches: List[str],
    start_day: str,
    end_day: str,
    date_policy: str,
    cloud_thresh_fastdiffsr: float,
    blank_frac_max: float,
    exact_dates_by_meta_patch: Optional[Dict[str, str]] = None,
) -> tuple[list[tuple[str, date, np.ndarray, Path]], dict]:
    """
    Best-effort raw RGB source selection for visual comparison with SR preview.
    Uses the current date policy semantics on the requested meta-patches.
    """
    if str(date_policy) == "all":
        return [], {"reason": "date_policy=all raw preview disabled", "date_policy": str(date_policy)}

    selected: list[tuple[str, date, np.ndarray, Path]] = []
    missing_reasons: dict[str, str] = {}
    planned_dates: Dict[str, date] = {}
    fallback_dates_used: Dict[str, str] = {}

    if str(date_policy) == "statewide_anchor":
        planned_dates, plan_summary = _plan_statewide_anchor_dates(
            raw_root=raw_root,
            meta_patches=[str(x) for x in meta_patches],
            start_day=start_day,
            end_day=end_day,
            cloud_frac_max=float(cloud_thresh_fastdiffsr),
            blank_frac_max=float(blank_frac_max),
        )
    else:
        plan_summary = None

    for mp in [str(x) for x in meta_patches]:
        meta_dir = raw_root / mp
        if not meta_dir.exists():
            missing_reasons[mp] = "meta_patch_missing"
            continue

        exact_date_str = None
        if isinstance(exact_dates_by_meta_patch, dict):
            exact_date_str = str(exact_dates_by_meta_patch.get(mp) or "").strip() or None
        if exact_date_str:
            try:
                exact_d = _parse_ymd(exact_date_str)
            except Exception:
                exact_d = None
            if exact_d is not None:
                got = _read_rgb10m_image(meta_dir, exact_d, cloud_frac_max=1.0, blank_frac_max=None)
                if got is not None:
                    rgb_lr, template_fp = got
                    selected.append((mp, exact_d, rgb_lr, template_fp))
                    continue
                missing_reasons[mp] = "exact_date_missing_or_invalid"
                continue

        if str(date_policy) == "statewide_anchor":
            d = planned_dates.get(mp)
            if d is None:
                # Distinguish no dates vs no valid observations
                any_dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
                missing_reasons[mp] = "no_dates_in_month" if not any_dates else "no_valid_observation"
                continue
            got = _read_rgb10m_image(
                meta_dir,
                d,
                cloud_frac_max=float(cloud_thresh_fastdiffsr),
                blank_frac_max=float(blank_frac_max),
            )
            if got is not None:
                rgb_lr, template_fp = got
                selected.append((mp, d, rgb_lr, template_fp))
                continue

            # Fallback: nearest usable date in the same month.
            month_dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
            fallback_dates = sorted(
                [x for x in month_dates if x != d],
                key=lambda x: (abs((x - d).days), -x.toordinal()),
            )
            chosen = None
            for fd in fallback_dates:
                got2 = _read_rgb10m_image(
                    meta_dir,
                    fd,
                    cloud_frac_max=float(cloud_thresh_fastdiffsr),
                    blank_frac_max=float(blank_frac_max),
                )
                if got2 is None:
                    continue
                rgb_lr, template_fp = got2
                chosen = (mp, fd, rgb_lr, template_fp)
                fallback_dates_used[mp] = f"{d.strftime('%Y-%m-%d')}->{fd.strftime('%Y-%m-%d')}"
                break
            if chosen is None:
                missing_reasons[mp] = "planned_date_not_valid"
                continue
            selected.append(chosen)
            continue

        dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
        if not dates:
            missing_reasons[mp] = "no_dates_in_month"
            continue
        if str(date_policy) == "latest":
            dates = sorted(dates, reverse=True)
        else:  # earliest
            dates = sorted(dates)
        chosen = None
        for d in dates:
            got = _read_rgb10m_image(
                meta_dir,
                d,
                cloud_frac_max=float(cloud_thresh_fastdiffsr),
                blank_frac_max=float(blank_frac_max),
            )
            if got is None:
                continue
            rgb_lr, template_fp = got
            chosen = (mp, d, rgb_lr, template_fp)
            break
        if chosen is None:
            missing_reasons[mp] = "no_valid_observation"
            continue
        selected.append(chosen)

    debug = {
        "date_policy": str(date_policy),
        "tile_count": int(len(selected)),
        "meta_patches": [str(mp) for mp, _, _, _ in selected],
        "missing_reasons": {str(k): str(v) for k, v in missing_reasons.items()},
        "fallback_dates_used": {str(k): str(v) for k, v in fallback_dates_used.items()},
        "blank_frac_max": float(blank_frac_max),
        "exact_dates_enforced": bool(isinstance(exact_dates_by_meta_patch, dict) and len(exact_dates_by_meta_patch) > 0),
    }
    if isinstance(plan_summary, dict):
        debug["date_plan_summary_bbox"] = plan_summary
    return selected, debug


def _build_raw_input_preview_for_bbox(
    *,
    raw_root: Path,
    bbox: tuple[float, float, float, float],
    meta_patches: List[str],
    start_day: str,
    end_day: str,
    date_policy: str,
    cloud_thresh_fastdiffsr: float,
    blank_frac_max: float,
    exact_dates_by_meta_patch: Optional[Dict[str, str]] = None,
    target_grid: Optional[dict] = None,
) -> dict:
    if str(date_policy) == "all":
        return {
            "input_preview_png_base64": None,
            "input_preview_width": None,
            "input_preview_height": None,
            "input_preview_crs": None,
            "input_preview_transform_gdal": None,
            "input_preview_tile_count": 0,
            "input_preview_debug": {"reason": "date_policy=all raw preview disabled", "date_policy": str(date_policy)},
        }

    selected_sources, raw_debug = _select_raw_preview_tile_sources(
        raw_root=raw_root,
        meta_patches=meta_patches,
        start_day=start_day,
        end_day=end_day,
        date_policy=date_policy,
        cloud_thresh_fastdiffsr=float(cloud_thresh_fastdiffsr),
        blank_frac_max=float(blank_frac_max),
        exact_dates_by_meta_patch=exact_dates_by_meta_patch,
    )
    if not selected_sources:
        return {
            "input_preview_png_base64": None,
            "input_preview_width": None,
            "input_preview_height": None,
            "input_preview_crs": None,
            "input_preview_transform_gdal": None,
            "input_preview_tile_count": 0,
            "input_preview_debug": raw_debug,
        }

    memfiles: list[MemoryFile] = []
    datasets = []
    selected_tile_bounds = []
    try:
        for mp, d, rgb_lr, template_fp in selected_sources:
            rgb8 = _stretch_to_uint8(rgb_lr)
            with rasterio.open(template_fp) as src:
                profile = src.profile.copy()
                transform = src.transform
                crs = src.crs
                width = int(src.width)
                height = int(src.height)
            profile.update(
                count=3,
                dtype=rasterio.uint8,
                width=width,
                height=height,
                nodata=None,
                transform=transform,
                compress="DEFLATE",
            )
            mf = MemoryFile()
            ds = mf.open(**profile)
            ds.write(np.transpose(rgb8, (2, 0, 1)))
            memfiles.append(mf)
            datasets.append(ds)
            selected_tile_bounds.append(
                {
                    "meta_patch": str(mp),
                    "date": d.strftime("%Y-%m-%d"),
                    "bounds": [
                        float(ds.bounds.left),
                        float(ds.bounds.bottom),
                        float(ds.bounds.right),
                        float(ds.bounds.top),
                    ],
                    "crs": str(ds.crs) if ds.crs is not None else None,
                }
            )

        crs = datasets[0].crs
        if crs is None:
            return {
                "input_preview_png_base64": None,
                "input_preview_width": None,
                "input_preview_height": None,
                "input_preview_crs": None,
                "input_preview_transform_gdal": None,
                "input_preview_tile_count": 0,
                "input_preview_debug": {**raw_debug, "reason": "raw_input_tiles_missing_crs"},
            }

        crs_set = {str(ds.crs) for ds in datasets}
        if len(crs_set) != 1:
            return {
                "input_preview_png_base64": None,
                "input_preview_width": None,
                "input_preview_height": None,
                "input_preview_crs": None,
                "input_preview_transform_gdal": None,
                "input_preview_tile_count": 0,
                "input_preview_debug": {**raw_debug, "reason": "mixed_crs", "crs_values": sorted(crs_set)},
            }

        lon_min, lat_min, lon_max, lat_max = bbox
        if isinstance(target_grid, dict) and target_grid.get("crs") and target_grid.get("bounds_output_crs"):
            tg = dict(target_grid)
            target_crs = str(tg.get("crs"))
            tb = tg.get("bounds_output_crs")
            if not (isinstance(tb, (list, tuple)) and len(tb) == 4):
                raise ValueError("target_grid.bounds_output_crs missing/invalid")
            target_bounds = tuple(float(x) for x in tb)
            if target_crs != str(crs):
                # Fallback for mixed CRS (unlikely here): transform SR compare bounds into raw CRS.
                q_left, q_bottom, q_right, q_top = transform_bounds(
                    target_crs,
                    crs,
                    float(target_bounds[0]),
                    float(target_bounds[1]),
                    float(target_bounds[2]),
                    float(target_bounds[3]),
                    densify_pts=21,
                )
                bounds_crs = (q_left, q_bottom, q_right, q_top)
            else:
                bounds_crs = target_bounds
            merged, out_transform = merge(datasets, bounds=bounds_crs)
            rgb = np.transpose(merged[:3], (1, 2, 0)).astype(np.uint8)
        else:
            bounds_crs = transform_bounds("EPSG:4326", crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21)
            merged, out_transform = merge(datasets, bounds=bounds_crs)
            rgb = np.transpose(merged[:3], (1, 2, 0)).astype(np.uint8)
        rgb_full = rgb
        out_transform_full = out_transform
        valid_mask = np.any(rgb_full > 0, axis=2)
        valid_crop = None
        if bool(valid_mask.any()):
            ys, xs = np.where(valid_mask)
            r0, r1 = int(ys.min()), int(ys.max()) + 1
            c0, c1 = int(xs.min()), int(xs.max()) + 1
            if r1 > r0 and c1 > c0:
                rgb = rgb_full[r0:r1, c0:c1, :]
                out_transform = out_transform_full * rasterio.Affine.translation(c0, r0)
                valid_crop = {"row0": r0, "row1": r1, "col0": c0, "col1": c1}

        raw_native_h = int(rgb.shape[0])
        raw_native_w = int(rgb.shape[1])
        preview_swap_rb = _parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_SWAP_RB", False)
        preview_stretch = _parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_STRETCH", True)
        rgb_preview_pre_web = _downsample_preview_rgb(rgb, max_side=4096)
        rgb_preview = _prepare_preview_rgb_for_web(rgb_preview_pre_web)
        debug = {
            **raw_debug,
            "bbox_wgs84": {
                "lon_min": float(lon_min),
                "lat_min": float(lat_min),
                "lon_max": float(lon_max),
                "lat_max": float(lat_max),
            },
            "merge_bounds_output_crs": [float(x) for x in bounds_crs],
            "selected_tile_bounds": selected_tile_bounds,
            "aligned_to_sr_footprint": bool(isinstance(target_grid, dict)),
            "cropped_to_valid_raw_extent": bool(valid_crop is not None),
            "valid_crop_pixels": valid_crop,
            "preview_max_side": 4096,
            "preview_stretch": bool(preview_stretch),
            "preview_swap_rb": bool(preview_swap_rb),
            "merged_rgb_stats_before_downsample": _preview_rgb_debug_stats(rgb),
            "merged_rgb_stats_before_downsample_full_bbox": _preview_rgb_debug_stats(rgb_full),
            "preview_native_size": {"width": raw_native_w, "height": raw_native_h},
        }
        debug["preview_rgb_stats_after_downsample"] = _preview_rgb_debug_stats(rgb_preview_pre_web)
        debug["preview_rgb_stats_final"] = _preview_rgb_debug_stats(rgb_preview)
        debug["preview_display_size"] = {"width": int(rgb_preview.shape[1]), "height": int(rgb_preview.shape[0])}
        debug["preview_was_downsampled"] = bool(
            int(rgb_preview.shape[0]) != raw_native_h or int(rgb_preview.shape[1]) != raw_native_w
        )
        return {
            "input_preview_png_base64": _rgb_png_base64(rgb_preview),
            "input_preview_width": int(rgb_preview.shape[1]),
            "input_preview_height": int(rgb_preview.shape[0]),
            "input_preview_native_width": raw_native_w,
            "input_preview_native_height": raw_native_h,
            "input_preview_was_downsampled": bool(
                int(rgb_preview.shape[0]) != raw_native_h or int(rgb_preview.shape[1]) != raw_native_w
            ),
            "input_preview_crs": str(crs),
            "input_preview_transform_gdal": tuple(float(x) for x in out_transform.to_gdal()),
            "input_preview_tile_count": int(len(datasets)),
            "input_preview_debug": debug,
            "_valid_mask_native": np.any(rgb > 0, axis=2),
        }
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        for mf in memfiles:
            try:
                mf.close()
            except Exception:
                pass


def _mask_sr_preview_to_raw_valid(*, preview: dict, raw_preview: dict) -> dict:
    sr_rgb = preview.get("_preview_rgb_native")
    raw_valid_mask = raw_preview.get("_valid_mask_native")
    out_transform = preview.get("_preview_out_transform")
    crs = preview.get("_preview_crs_obj")
    if sr_rgb is None or raw_valid_mask is None or out_transform is None or crs is None:
        return preview

    sr_rgb = np.asarray(sr_rgb)
    raw_valid_mask = np.asarray(raw_valid_mask, dtype=bool)
    if sr_rgb.ndim != 3 or raw_valid_mask.ndim != 2:
        return preview
    if sr_rgb.shape[:2] != raw_valid_mask.shape:
        return preview

    masked_rgb = np.where(raw_valid_mask[..., None], sr_rgb, 0).astype(np.uint8)
    masked_transform = out_transform
    masked_crop = None
    if bool(raw_valid_mask.any()):
        ys, xs = np.where(raw_valid_mask)
        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1
        if r1 > r0 and c1 > c0:
            masked_rgb = masked_rgb[r0:r1, c0:c1, :]
            masked_transform = out_transform * rasterio.Affine.translation(c0, r0)
            masked_crop = {"row0": r0, "row1": r1, "col0": c0, "col1": c1}

    debug = dict(preview.get("preview_debug") or {})
    debug["masked_to_raw_valid_footprint"] = True
    debug["raw_valid_mask_fraction_on_sr_grid"] = float(raw_valid_mask.mean())
    debug["raw_valid_mask_nonzero_pixels_on_sr_grid"] = int(raw_valid_mask.sum())
    debug["sr_mask_crop_pixels"] = masked_crop
    debug["merged_rgb_stats_after_raw_mask_before_downsample"] = _preview_rgb_debug_stats(masked_rgb)

    native_h = int(masked_rgb.shape[0])
    native_w = int(masked_rgb.shape[1])
    rgb_preview = _downsample_preview_rgb(masked_rgb, max_side=4096)
    debug["preview_rgb_stats_after_downsample"] = _preview_rgb_debug_stats(rgb_preview)
    rgb_preview = _prepare_preview_rgb_for_web(rgb_preview)
    debug["preview_rgb_stats_final"] = _preview_rgb_debug_stats(rgb_preview)
    debug["preview_native_size"] = {"width": native_w, "height": native_h}
    debug["preview_display_size"] = {"width": int(rgb_preview.shape[1]), "height": int(rgb_preview.shape[0])}
    debug["preview_was_downsampled"] = bool(
        int(rgb_preview.shape[0]) != native_h or int(rgb_preview.shape[1]) != native_w
    )
    compare_bounds = array_bounds(int(masked_rgb.shape[0]), int(masked_rgb.shape[1]), masked_transform)

    updated = dict(preview)
    updated["preview_png_base64"] = _rgb_png_base64(rgb_preview)
    updated["preview_width"] = int(rgb_preview.shape[1])
    updated["preview_height"] = int(rgb_preview.shape[0])
    updated["preview_native_width"] = native_w
    updated["preview_native_height"] = native_h
    updated["preview_was_downsampled"] = bool(
        int(rgb_preview.shape[0]) != native_h or int(rgb_preview.shape[1]) != native_w
    )
    updated["preview_crs"] = str(crs)
    updated["preview_transform_gdal"] = tuple(float(x) for x in masked_transform.to_gdal())
    updated["preview_debug"] = debug
    updated["_compare_grid"] = {
        "crs": str(crs),
        "height": int(masked_rgb.shape[0]),
        "width": int(masked_rgb.shape[1]),
        "transform_gdal": tuple(float(x) for x in masked_transform.to_gdal()),
        "bounds_output_crs": [float(x) for x in compare_bounds],
    }
    updated["_preview_rgb_native"] = masked_rgb
    updated["_preview_out_transform"] = masked_transform
    updated["_preview_crs_obj"] = crs
    return updated


def _empty_planet_preview(*, reason: str, extra_debug: Optional[dict] = None) -> dict:
    debug = {"reason": str(reason)}
    if isinstance(extra_debug, dict):
        debug.update(extra_debug)
    return {
        "planet_preview_png_base64": None,
        "planet_preview_width": None,
        "planet_preview_height": None,
        "planet_preview_native_width": None,
        "planet_preview_native_height": None,
        "planet_preview_was_downsampled": False,
        "planet_preview_crs": None,
        "planet_preview_transform_gdal": None,
        "planet_preview_tile_count": 0,
        "planet_preview_debug": debug,
    }


def _planet_rgb_band_indexes(ds) -> tuple[int, int, int]:
    by_name: dict[str, int] = {}
    for idx, desc in enumerate(ds.descriptions, start=1):
        key = str(desc or "").strip().lower()
        if key:
            by_name[key] = idx

    red = by_name.get("red")
    green = by_name.get("green")
    blue = by_name.get("blue")
    if red and green and blue:
        return red, green, blue

    red_ci = None
    green_ci = None
    blue_ci = None
    for idx, ci in enumerate(ds.colorinterp, start=1):
        if ci == ColorInterp.red:
            red_ci = idx
        elif ci == ColorInterp.green:
            green_ci = idx
        elif ci == ColorInterp.blue:
            blue_ci = idx
    if red_ci and green_ci and blue_ci:
        return red_ci, green_ci, blue_ci

    if int(ds.count) >= 6:
        return 6, 4, 2
    if int(ds.count) >= 3:
        return 1, 2, 3
    raise HTTPException(status_code=500, detail="Planet PSScene raster does not contain readable RGB bands.")


def _build_planet_preview_for_bbox(*, bbox: tuple[float, float, float, float]) -> dict:
    root = _resolve_ps_scene_root()
    if not root.exists():
        return _empty_planet_preview(
            reason="psscene_root_missing",
            extra_debug={"ps_scene_root": str(root)},
        )

    manifest = _load_ps_scene_manifest()
    if not manifest:
        return _empty_planet_preview(
            reason="psscene_manifest_empty",
            extra_debug={"ps_scene_root": str(root)},
        )

    lon_min, lat_min, lon_max, lat_max = bbox
    selected = [
        scene
        for scene in manifest
        if _bbox_intersects(
            a_lon_min=lon_min,
            a_lat_min=lat_min,
            a_lon_max=lon_max,
            a_lat_max=lat_max,
            b_lon_min=float(scene["lon_min"]),
            b_lat_min=float(scene["lat_min"]),
            b_lon_max=float(scene["lon_max"]),
            b_lat_max=float(scene["lat_max"]),
        )
    ]
    if not selected:
        return _empty_planet_preview(
            reason="no_psscene_overlap",
            extra_debug={
                "ps_scene_root": str(root),
                "bbox_wgs84": {
                    "lon_min": float(lon_min),
                    "lat_min": float(lat_min),
                    "lon_max": float(lon_max),
                    "lat_max": float(lat_max),
                },
            },
        )

    target_crs = "EPSG:3857"
    target_bounds = transform_bounds("EPSG:4326", target_crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21)
    left, bottom, right, top = (float(x) for x in target_bounds)
    if not (math.isfinite(left) and math.isfinite(bottom) and math.isfinite(right) and math.isfinite(top)):
        return _empty_planet_preview(reason="invalid_target_bounds")
    if right <= left or top <= bottom:
        return _empty_planet_preview(reason="empty_target_bounds")

    opened = []
    selected_tile_bounds = []
    skipped_scenes: list[dict] = []
    try:
        for scene in selected:
            try:
                ds = rasterio.open(scene["raster_path"])
            except Exception as exc:
                skipped_scenes.append(
                    {
                        "id": str(scene["id"]),
                        "path": str(scene["raster_path"]),
                        "reason": f"open_failed:{exc}",
                    }
                )
                continue
            if ds.crs is None:
                skipped_scenes.append(
                    {
                        "id": str(scene["id"]),
                        "path": str(scene["raster_path"]),
                        "reason": "missing_crs",
                    }
                )
                ds.close()
                continue
            opened.append((scene, ds))
            selected_tile_bounds.append(
                {
                    "scene_id": str(scene["id"]),
                    "date": scene.get("acquired_day"),
                    "bounds_wgs84": [
                        float(scene["lon_min"]),
                        float(scene["lat_min"]),
                        float(scene["lon_max"]),
                        float(scene["lat_max"]),
                    ],
                    "crs": str(ds.crs),
                    "path": str(scene["raster_path"]),
                }
            )

        if not opened:
            return _empty_planet_preview(
                reason="no_readable_psscene_rasters",
                extra_debug={"skipped_scenes": skipped_scenes, "ps_scene_root": str(root)},
            )

        res_candidates = []
        for _, ds in opened:
            try:
                res_x = abs(float(ds.res[0]))
                res_y = abs(float(ds.res[1]))
            except Exception:
                continue
            if res_x > 0:
                res_candidates.append(res_x)
            if res_y > 0:
                res_candidates.append(res_y)
        target_res = min(res_candidates) if res_candidates else 3.0
        target_width = max(1, int(math.ceil((right - left) / target_res)))
        target_height = max(1, int(math.ceil((top - bottom) / target_res)))

        preview_max_side = 4096
        longest = max(target_width, target_height)
        if longest > preview_max_side:
            scale = float(longest) / float(preview_max_side)
            target_width = max(1, int(math.ceil(target_width / scale)))
            target_height = max(1, int(math.ceil(target_height / scale)))

        target_transform = from_bounds(left, bottom, right, top, target_width, target_height)
        composite = np.zeros((3, target_height, target_width), dtype=np.uint16)
        valid_mask = np.zeros((target_height, target_width), dtype=bool)

        for scene, ds in opened:
            q_left, q_bottom, q_right, q_top = transform_bounds(
                "EPSG:4326",
                ds.crs,
                lon_min,
                lat_min,
                lon_max,
                lat_max,
                densify_pts=21,
            )
            src_left = max(float(ds.bounds.left), float(q_left))
            src_bottom = max(float(ds.bounds.bottom), float(q_bottom))
            src_right = min(float(ds.bounds.right), float(q_right))
            src_top = min(float(ds.bounds.top), float(q_top))
            if src_right <= src_left or src_top <= src_bottom:
                continue

            window = window_from_bounds(src_left, src_bottom, src_right, src_top, transform=ds.transform)
            try:
                band_indexes = _planet_rgb_band_indexes(ds)
                src_rgb = ds.read(indexes=list(band_indexes), window=window)
            except Exception as exc:
                skipped_scenes.append(
                    {
                        "id": str(scene["id"]),
                        "path": str(scene["raster_path"]),
                        "reason": f"read_failed:{exc}",
                    }
                )
                continue
            if src_rgb.size == 0:
                continue

            src_transform = ds.window_transform(window)
            src_valid = np.any(src_rgb > 0, axis=0).astype(np.uint8)
            if not bool(src_valid.any()):
                continue

            tmp_rgb = np.zeros((3, target_height, target_width), dtype=np.uint16)
            tmp_valid = np.zeros((target_height, target_width), dtype=np.uint8)
            for band_idx in range(3):
                reproject(
                    source=src_rgb[band_idx],
                    destination=tmp_rgb[band_idx],
                    src_transform=src_transform,
                    src_crs=ds.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    src_nodata=0,
                    dst_nodata=0,
                    resampling=Resampling.bilinear,
                )
            reproject(
                source=src_valid,
                destination=tmp_valid,
                src_transform=src_transform,
                src_crs=ds.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            use_mask = tmp_valid > 0
            if not bool(use_mask.any()):
                continue
            composite[:, use_mask] = tmp_rgb[:, use_mask]
            valid_mask[use_mask] = True

        if not bool(valid_mask.any()):
            return _empty_planet_preview(
                reason="no_valid_planet_pixels_for_bbox",
                extra_debug={
                    "ps_scene_root": str(root),
                    "selected_tile_bounds": selected_tile_bounds,
                    "skipped_scenes": skipped_scenes,
                },
            )

        rgb_full = np.transpose(composite, (1, 2, 0))
        rgb_full = _stretch_to_uint8(rgb_full)
        out_transform = target_transform
        cropped = None
        if bool(valid_mask.any()):
            ys, xs = np.where(valid_mask)
            r0, r1 = int(ys.min()), int(ys.max()) + 1
            c0, c1 = int(xs.min()), int(xs.max()) + 1
            if r1 > r0 and c1 > c0:
                rgb = rgb_full[r0:r1, c0:c1, :]
                valid_mask_cropped = valid_mask[r0:r1, c0:c1]
                out_transform = target_transform * rasterio.Affine.translation(c0, r0)
                cropped = {"row0": r0, "row1": r1, "col0": c0, "col1": c1}
            else:
                rgb = rgb_full
                valid_mask_cropped = valid_mask
        else:
            rgb = rgb_full
            valid_mask_cropped = valid_mask

        rgb = np.where(valid_mask_cropped[..., None], rgb, 0).astype(np.uint8)
        native_h = int(rgb.shape[0])
        native_w = int(rgb.shape[1])
        rgb_preview_pre_web = _downsample_preview_rgb(rgb, max_side=4096)
        rgb_preview = _prepare_preview_rgb_for_web(rgb_preview_pre_web)
        partial_coverage = float(valid_mask.mean()) < 0.999999
        debug = {
            "source": "planet_psscene",
            "ps_scene_root": str(root),
            "fixed_reference_date": "2024-08-13",
            "bbox_wgs84": {
                "lon_min": float(lon_min),
                "lat_min": float(lat_min),
                "lon_max": float(lon_max),
                "lat_max": float(lat_max),
            },
            "selected_tile_bounds": selected_tile_bounds,
            "skipped_scenes": skipped_scenes,
            "partial_coverage": bool(partial_coverage),
            "cropped_to_valid_planet_extent": bool(cropped is not None),
            "valid_crop_pixels": cropped,
            "preview_max_side": 4096,
            "preview_stretch": bool(_parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_STRETCH", True)),
            "preview_swap_rb": bool(_parse_env_bool("DEEPSAT_FASTDIFFSR_PREVIEW_SWAP_RB", False)),
            "preview_target_crs": target_crs,
            "preview_target_size_before_crop": {"width": int(target_width), "height": int(target_height)},
            "preview_native_size": {"width": native_w, "height": native_h},
            "valid_fraction_on_requested_bbox_grid": float(valid_mask.mean()),
            "merged_rgb_stats_before_downsample": _preview_rgb_debug_stats(rgb),
            "merged_rgb_stats_before_downsample_full_bbox": _preview_rgb_debug_stats(rgb_full),
        }
        debug["preview_rgb_stats_after_downsample"] = _preview_rgb_debug_stats(rgb_preview_pre_web)
        debug["preview_rgb_stats_final"] = _preview_rgb_debug_stats(rgb_preview)
        debug["preview_display_size"] = {"width": int(rgb_preview.shape[1]), "height": int(rgb_preview.shape[0])}
        debug["preview_was_downsampled"] = bool(
            int(rgb_preview.shape[0]) != native_h or int(rgb_preview.shape[1]) != native_w
        )
        debug["merge_bounds_output_crs"] = [float(x) for x in array_bounds(native_h, native_w, out_transform)]

        return {
            "planet_preview_png_base64": _rgb_png_base64(rgb_preview),
            "planet_preview_width": int(rgb_preview.shape[1]),
            "planet_preview_height": int(rgb_preview.shape[0]),
            "planet_preview_native_width": native_w,
            "planet_preview_native_height": native_h,
            "planet_preview_was_downsampled": bool(
                int(rgb_preview.shape[0]) != native_h or int(rgb_preview.shape[1]) != native_w
            ),
            "planet_preview_crs": target_crs,
            "planet_preview_transform_gdal": tuple(float(x) for x in out_transform.to_gdal()),
            "planet_preview_tile_count": int(len(opened)),
            "planet_preview_debug": debug,
        }
    finally:
        for _, ds in opened:
            try:
                ds.close()
            except Exception:
                pass


def _download_cmd_from_job(job: dict, settings: dict) -> list[str]:
    data_root = Path(str(settings["download_data_root"]))
    cmd = [
        sys.executable,
        str(APP_DIR / "ar_deploy.py"),
        "download",
        "--project",
        str(settings["ee_project"]),
        "--year",
        str(job["year"]),
        "--month",
        str(job["month"]),
        "--data-root",
        str(data_root),
        "--download-workers",
        str(int(settings["download_workers"])),
        "--cloud-thresh",
        str(float(settings["download_cloud_thresh"])),
        "--download-retries",
        "3",
        "--download-retry-sleep-s",
        "1.0",
        "--verify" if bool(settings["download_verify"]) else "--no-verify",
        "--strict" if bool(settings["download_strict"]) else "--no-strict",
        "--fast-skip" if bool(settings["download_fast_skip"]) else "--no-fast-skip",
    ]
    return cmd


def _ensure_raw_month_downloaded(*, job: dict, settings: dict, raw_root: Path) -> None:
    if raw_root.exists():
        job["phase"] = "generating_sr"
        job["phase_message"] = f"Raw data already exists: {raw_root}"
        _update_job(job)
        return

    _set_job_phase(job, phase="downloading_raw", message=f"Downloading Arkansas raw data for {job['year']}-{int(job['month']):02d}...")
    cmd = _download_cmd_from_job(job, settings)
    proc = subprocess.run(
        cmd,
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"download subprocess exit code {proc.returncode}"
        job.setdefault("errors", []).append(f"download_failed: {msg}")
        job["phase_message"] = "Raw data download failed."
        _finish_job(job, state="failed")
        raise RuntimeError(msg)

    if not raw_root.exists():
        msg = f"raw_root not found after download: {raw_root}"
        job.setdefault("errors", []).append(msg)
        job["phase_message"] = "Raw data download finished but expected directory was not created."
        _finish_job(job, state="failed")
        raise RuntimeError(msg)

    _set_job_phase(job, phase="generating_sr", message="Raw data download completed. Starting SR generation.")


@lru_cache(maxsize=4)
def _load_diffusion_cached(config_path: str, resume_state: str, device_key: str):
    device_ids = [int(x) for x in device_key.split(",") if x.strip()]
    if not device_ids:
        device_ids = [0]
    return _build_fastdiffsr_model(
        config_path=config_path,
        resume_state=resume_state,
        device_ids=device_ids,
    )


def _normalize_device_key(device: str) -> str:
    ids = [str(int(x.strip())) for x in str(device).split(",") if x.strip()]
    return ",".join(ids) if ids else "0"


class BBox(BaseModel):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    model_config = ConfigDict(extra="forbid")


class FastDiffSROutput(BaseModel):
    meta_patch: str
    date: Optional[str] = None
    status: Literal[
        "written",
        "skipped_existing",
        "meta_patch_missing",
        "no_dates_in_month",
        "no_valid_observation",
        "error",
    ]
    path: Optional[str] = None


class FastDiffSRPredictRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100)
    month: int = Field(..., ge=1, le=12)
    bbox: BBox

    # Strict contract: reject legacy request fields.
    model_config = ConfigDict(extra="forbid")


class FastDiffSRPredictResponse(BaseModel):
    year: int
    month: int
    start_day: str
    end_day: str
    raw_root: str
    output_dir: str

    config_path: str
    resume_state: str
    device: str
    scale: int
    lr_tile_size: int
    date_policy: str
    cloud_thresh_fastdiffsr: float

    meta_patches: List[str]
    total_meta_patches: int
    wrote: int
    skipped_existing: int
    preview_png_base64: Optional[str] = None
    preview_width: Optional[int] = None
    preview_height: Optional[int] = None
    preview_native_width: Optional[int] = None
    preview_native_height: Optional[int] = None
    preview_was_downsampled: bool = False
    preview_crs: Optional[str] = None
    preview_transform_gdal: Optional[Tuple[float, float, float, float, float, float]] = None
    preview_tile_count: int = 0
    preview_debug: Optional[dict] = None
    input_preview_png_base64: Optional[str] = None
    input_preview_width: Optional[int] = None
    input_preview_height: Optional[int] = None
    input_preview_native_width: Optional[int] = None
    input_preview_native_height: Optional[int] = None
    input_preview_was_downsampled: bool = False
    input_preview_crs: Optional[str] = None
    input_preview_transform_gdal: Optional[Tuple[float, float, float, float, float, float]] = None
    input_preview_tile_count: int = 0
    input_preview_debug: Optional[dict] = None
    planet_preview_png_base64: Optional[str] = None
    planet_preview_width: Optional[int] = None
    planet_preview_height: Optional[int] = None
    planet_preview_native_width: Optional[int] = None
    planet_preview_native_height: Optional[int] = None
    planet_preview_was_downsampled: bool = False
    planet_preview_crs: Optional[str] = None
    planet_preview_transform_gdal: Optional[Tuple[float, float, float, float, float, float]] = None
    planet_preview_tile_count: int = 0
    planet_preview_debug: Optional[dict] = None
    outputs: List[FastDiffSROutput]


class FastDiffSRJobAcceptedResponse(BaseModel):
    state: Literal["queued", "running"]
    job_kind: Literal["month_seed"] = "month_seed"
    job_scope: Literal["full_month"] = "full_month"
    phase: Literal["queued", "downloading_raw", "generating_sr"]
    job_id: str
    status_url: str
    year: int
    month: int
    total_meta_patches: int
    missing_meta_patches: int
    missing_meta_patch_ids: List[str]
    requested_meta_patches: int
    requested_meta_patch_ids: List[str]
    output_dir: str
    message: str


class FastDiffSRJobStatusResponse(BaseModel):
    job_id: str
    job_key: str
    state: Literal["queued", "running", "completed", "failed", "interrupted"]
    job_kind: Literal["month_seed"] = "month_seed"
    job_scope: Literal["full_month"] = "full_month"
    phase: Literal["queued", "downloading_raw", "generating_sr", "completed", "failed", "interrupted"]
    phase_message: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    year: int
    month: int
    bbox: BBox
    output_dir: str
    status_url: str
    progress: Dict[str, int]
    date_plan_summary: Optional[dict] = None
    missing_meta_patches: List[str]
    outputs: List[FastDiffSROutput]
    errors: List[str]


app = FastAPI(title="Arkansas FastDiffSR API", version="0.3.0")
_FASTDIFFSR_LOCK = threading.Lock()

_JOB_LOCK = threading.Lock()
_JOB_COND = threading.Condition(_JOB_LOCK)
_JOB_QUEUE: List[str] = []
_JOB_ACTIVE_BY_KEY: Dict[str, str] = {}
_WORKER_THREAD: Optional[threading.Thread] = None


def _mark_inflight_jobs_interrupted() -> None:
    out_root = _resolve_path(os.environ.get("DEEPSAT_FASTDIFFSR_OUT_ROOT", "./fastdiffsr_preds"))
    jobs_dir = _jobs_dir(out_root)
    if not jobs_dir.exists():
        return
    for fp in sorted(jobs_dir.glob("*.json")):
        try:
            job = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("state") in JOB_ACTIVE_STATES:
            now = _utc_now_iso()
            job["state"] = "interrupted"
            job["phase"] = "interrupted"
            job["phase_message"] = "API restarted while job was in progress."
            job["updated_at"] = now
            job["finished_at"] = now
            errs = list(job.get("errors") or [])
            errs.append("API restarted while job was in progress.")
            job["errors"] = errs
            _write_json_atomic(fp, job)


@app.on_event("startup")
def _on_startup() -> None:
    _mark_inflight_jobs_interrupted()


def _enqueue_job(job: dict) -> None:
    global _WORKER_THREAD
    out_root = _resolve_path(str(job["settings"]["out_root"]))
    jobs_dir = _jobs_dir(out_root)
    _write_json_atomic(_job_fp(jobs_dir, str(job["job_id"])), job)

    with _JOB_COND:
        _JOB_ACTIVE_BY_KEY[str(job["job_key"])] = str(job["job_id"])
        _JOB_QUEUE.append(str(job["job_id"]))
        if _WORKER_THREAD is None or not _WORKER_THREAD.is_alive():
            _WORKER_THREAD = threading.Thread(target=_job_worker_loop, name="fastdiffsr-job-worker", daemon=True)
            _WORKER_THREAD.start()
        _JOB_COND.notify_all()


def _update_job(job: dict) -> None:
    out_root = _resolve_path(str(job["settings"]["out_root"]))
    jobs_dir = _jobs_dir(out_root)
    job["updated_at"] = _utc_now_iso()
    _write_json_atomic(_job_fp(jobs_dir, str(job["job_id"])), job)


def _finish_job(job: dict, *, state: str) -> None:
    job["state"] = state
    if str(state) in JOB_TERMINAL_STATES:
        job["phase"] = str(state)
    job["updated_at"] = _utc_now_iso()
    job["finished_at"] = _utc_now_iso()
    _update_job(job)
    with _JOB_LOCK:
        cur = _JOB_ACTIVE_BY_KEY.get(str(job["job_key"]))
        if cur == str(job["job_id"]):
            del _JOB_ACTIVE_BY_KEY[str(job["job_key"])]


def _job_worker_loop() -> None:
    while True:
        with _JOB_COND:
            while not _JOB_QUEUE:
                _JOB_COND.wait(timeout=10.0)
                if not _JOB_QUEUE:
                    return
            job_id = _JOB_QUEUE.pop(0)

        try:
            _run_job(job_id)
        except Exception:
            # Best effort: if processing fails before job load, ignore and continue.
            pass


def _run_job(job_id: str) -> None:
    out_root = _resolve_path(os.environ.get("DEEPSAT_FASTDIFFSR_OUT_ROOT", "./fastdiffsr_preds"))
    jobs_dir = _jobs_dir(out_root)
    job = _read_job_file(jobs_dir, job_id)

    if job.get("state") not in JOB_ACTIVE_STATES:
        return

    job["state"] = "running"
    job["started_at"] = _utc_now_iso()
    job.setdefault("phase", "queued")
    _update_job(job)

    settings = dict(job.get("settings") or {})
    raw_root = Path(str(settings["raw_root"]))
    out_dir = Path(str(settings["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(str(settings["config_path"]))
    checkpoint_path = Path(str(settings["checkpoint_path"]))

    # Validate runtime resources at job execution time.
    if not config_path.exists():
        errs = list(job.get("errors") or [])
        errs.append(f"FastDiffSR config not found: {config_path}")
        job["errors"] = errs
        _finish_job(job, state="failed")
        return
    if not checkpoint_path.exists() and not Path(f"{checkpoint_path}_gen.pth").exists():
        errs = list(job.get("errors") or [])
        errs.append(f"FastDiffSR checkpoint not found: {checkpoint_path}")
        job["errors"] = errs
        _finish_job(job, state="failed")
        return

    try:
        resume_state = _resolve_fastdiffsr_resume_state(str(checkpoint_path))
    except SystemExit as exc:
        errs = list(job.get("errors") or [])
        errs.append(f"Invalid DEEPSAT_FASTDIFFSR_CHECKPOINT: {exc}")
        job["errors"] = errs
        _finish_job(job, state="failed")
        return

    job["resume_state"] = resume_state
    _update_job(job)

    start_day = str(job["start_day"])
    end_day = str(job["end_day"])
    date_policy = str(settings["date_policy"])
    blank_frac_max = float(settings.get("blank_frac_max", 0.98))
    planned_dates_by_meta_patch: Dict[str, date] = {}

    try:
        _ensure_raw_month_downloaded(job=job, settings=settings, raw_root=raw_root)
        if date_policy == "statewide_anchor":
            _set_job_phase(
                job,
                phase="generating_sr",
                message="Planning a month-wide anchor date to maximize same-date coverage across Arkansas tiles.",
            )
            planned_dates_by_meta_patch, plan_summary = _plan_statewide_anchor_dates(
                raw_root=raw_root,
                meta_patches=[str(x) for x in list(job.get("missing_meta_patches") or [])],
                start_day=start_day,
                end_day=end_day,
                cloud_frac_max=float(settings["cloud_thresh_fastdiffsr"]),
                blank_frac_max=blank_frac_max,
            )
            job["date_plan_summary"] = plan_summary
            _update_job(job)

        with _FASTDIFFSR_LOCK:
            _set_job_phase(job, phase="generating_sr", message="Loading FastDiffSR model and generating outputs.")
            diffusion = _load_diffusion_cached(str(config_path), resume_state, str(settings["device"]))

            for mp in list(job.get("missing_meta_patches") or []):
                # Progress is per meta-patch.
                try:
                    existing = _existing_outputs_for_meta_patch(out_dir=out_dir, meta_patch=str(mp), date_policy=date_policy)
                    if existing and not bool(settings["force"]):
                        for fp in existing:
                            entry = {
                                "meta_patch": str(mp),
                                "date": _date_from_all_policy_filename(fp) if date_policy == "all" else None,
                                "status": "skipped_existing",
                                "path": str(fp),
                            }
                            job.setdefault("outputs", []).append(entry)
                            job["progress"]["skipped"] = int(job["progress"].get("skipped", 0)) + 1
                        job["progress"]["done"] = int(job["progress"].get("done", 0)) + 1
                        _update_job(job)
                        continue

                    meta_dir = raw_root / str(mp)
                    if not meta_dir.exists():
                        job.setdefault("outputs", []).append({"meta_patch": str(mp), "status": "meta_patch_missing", "path": None, "date": None})
                        job["progress"]["done"] = int(job["progress"].get("done", 0)) + 1
                        _update_job(job)
                        continue

                    selected: List[Tuple[date, object, Path]] = []
                    if date_policy == "statewide_anchor":
                        planned_d = planned_dates_by_meta_patch.get(str(mp))
                        if planned_d is not None:
                            got = _read_rgb10m_image(
                                meta_dir,
                                planned_d,
                                cloud_frac_max=float(settings["cloud_thresh_fastdiffsr"]),
                                blank_frac_max=blank_frac_max,
                            )
                            if got is not None:
                                rgb_lr, template_fp = got
                                selected.append((planned_d, rgb_lr, template_fp))
                    else:
                        dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
                        if not dates:
                            job.setdefault("outputs", []).append({"meta_patch": str(mp), "status": "no_dates_in_month", "path": None, "date": None})
                            job["progress"]["done"] = int(job["progress"].get("done", 0)) + 1
                            _update_job(job)
                            continue

                        if date_policy == "latest":
                            dates = sorted(dates, reverse=True)
                        elif date_policy in ("earliest", "all"):
                            dates = sorted(dates)

                        for d in dates:
                            got = _read_rgb10m_image(
                                meta_dir,
                                d,
                                cloud_frac_max=float(settings["cloud_thresh_fastdiffsr"]),
                                blank_frac_max=blank_frac_max,
                            )
                            if got is None:
                                continue
                            rgb_lr, template_fp = got
                            selected.append((d, rgb_lr, template_fp))
                            if date_policy != "all":
                                break

                    if not selected:
                        # Distinguish tiles with no dates at all when using statewide anchor planning.
                        if date_policy == "statewide_anchor":
                            dates_any = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
                            status = "no_dates_in_month" if not dates_any else "no_valid_observation"
                        else:
                            status = "no_valid_observation"
                        job.setdefault("outputs", []).append({"meta_patch": str(mp), "status": status, "path": None, "date": None})
                        job["progress"]["done"] = int(job["progress"].get("done", 0)) + 1
                        _update_job(job)
                        continue

                    for d, rgb_lr, template_fp in selected:
                        suffix = f"_{d.strftime('%Y%m%d')}" if date_policy == "all" else ""
                        out_fp = out_dir / f"{mp}{suffix}.tif"
                        if out_fp.exists() and not bool(settings["force"]):
                            job.setdefault("outputs", []).append(
                                {
                                    "meta_patch": str(mp),
                                    "date": d.strftime("%Y-%m-%d"),
                                    "status": "skipped_existing",
                                    "path": str(out_fp),
                                }
                            )
                            job["progress"]["skipped"] = int(job["progress"].get("skipped", 0)) + 1
                            continue

                        sr_rgb = _infer_fastdiffsr_rgb(
                            diffusion,
                            rgb_lr,
                            scale=int(settings["scale"]),
                            lr_tile_size=int(settings["lr_tile_size"]),
                        )
                        _write_rgb_geotiff(
                            out_fp=out_fp,
                            rgb=sr_rgb,
                            template_fp=template_fp,
                            scale=int(settings["scale"]),
                        )
                        _write_sr_tile_metadata(
                            out_fp=out_fp,
                            meta_patch=str(mp),
                            used_date=d,
                            template_fp=template_fp,
                        )
                        job.setdefault("outputs", []).append(
                            {
                                "meta_patch": str(mp),
                                "date": d.strftime("%Y-%m-%d"),
                                "status": "written",
                                "path": str(out_fp),
                            }
                        )
                        job["progress"]["written"] = int(job["progress"].get("written", 0)) + 1

                    job["progress"]["done"] = int(job["progress"].get("done", 0)) + 1
                    _update_job(job)

                except Exception as exc:
                    job.setdefault("errors", []).append(f"meta_patch={mp}: {exc}")
                    job.setdefault("outputs", []).append(
                        {
                            "meta_patch": str(mp),
                            "status": "error",
                            "path": None,
                            "date": None,
                        }
                    )
                    job["progress"]["failed"] = int(job["progress"].get("failed", 0)) + 1
                    job["progress"]["done"] = int(job["progress"].get("done", 0)) + 1
                    _update_job(job)

    except Exception as exc:
        if str(job.get("state")) in JOB_TERMINAL_STATES:
            return
        job.setdefault("errors", []).append(f"job_failed: {exc}")
        job["phase_message"] = str(exc)
        _finish_job(job, state="failed")
        return

    _finish_job(job, state="completed")


def _build_existing_response(
    *,
    req: FastDiffSRPredictRequest,
    settings: dict,
    meta_patches: List[str],
    bbox: tuple[float, float, float, float],
    resume_state: str,
) -> FastDiffSRPredictResponse:
    out_dir = Path(settings["out_dir"])
    outputs = _collect_existing_outputs(
        out_dir=out_dir,
        meta_patches=meta_patches,
        date_policy=str(settings["date_policy"]),
    )
    exact_dates_by_meta_patch = {str(o.meta_patch): str(o.date) for o in outputs if getattr(o, "date", None)}
    preview = _build_fastdiffsr_preview_for_bbox(
        out_dir=out_dir,
        bbox=bbox,
        meta_patches=meta_patches,
        date_policy=str(settings["date_policy"]),
        mosaic_fp=settings.get("mosaic_path"),
    )
    start_day, end_day = _month_range(int(req.year), int(req.month))
    raw_preview = _build_raw_input_preview_for_bbox(
        raw_root=Path(settings["raw_root"]),
        bbox=bbox,
        meta_patches=meta_patches,
        start_day=start_day,
        end_day=end_day,
        date_policy=str(settings["date_policy"]),
        cloud_thresh_fastdiffsr=float(settings["cloud_thresh_fastdiffsr"]),
        blank_frac_max=float(settings.get("blank_frac_max", 0.98)),
        exact_dates_by_meta_patch=exact_dates_by_meta_patch,
        target_grid=preview.get("_compare_grid") if isinstance(preview, dict) else None,
    )
    preview = _mask_sr_preview_to_raw_valid(preview=preview, raw_preview=raw_preview)
    planet_preview = _build_planet_preview_for_bbox(bbox=bbox)
    return FastDiffSRPredictResponse(
        year=int(req.year),
        month=int(req.month),
        start_day=start_day,
        end_day=end_day,
        raw_root=str(settings["raw_root"]),
        output_dir=str(out_dir),
        config_path=str(settings["config_path"]),
        resume_state=resume_state,
        device=str(settings["device"]),
        scale=int(settings["scale"]),
        lr_tile_size=int(settings["lr_tile_size"]),
        date_policy=str(settings["date_policy"]),
        cloud_thresh_fastdiffsr=float(settings["cloud_thresh_fastdiffsr"]),
        meta_patches=[str(x) for x in meta_patches],
        total_meta_patches=len(meta_patches),
        wrote=0,
        skipped_existing=len(outputs),
        preview_png_base64=preview.get("preview_png_base64"),
        preview_width=preview.get("preview_width"),
        preview_height=preview.get("preview_height"),
        preview_native_width=preview.get("preview_native_width"),
        preview_native_height=preview.get("preview_native_height"),
        preview_was_downsampled=bool(preview.get("preview_was_downsampled", False)),
        preview_crs=preview.get("preview_crs"),
        preview_transform_gdal=preview.get("preview_transform_gdal"),
        preview_tile_count=int(preview.get("preview_tile_count", 0)),
        preview_debug=preview.get("preview_debug"),
        input_preview_png_base64=raw_preview.get("input_preview_png_base64"),
        input_preview_width=raw_preview.get("input_preview_width"),
        input_preview_height=raw_preview.get("input_preview_height"),
        input_preview_native_width=raw_preview.get("input_preview_native_width"),
        input_preview_native_height=raw_preview.get("input_preview_native_height"),
        input_preview_was_downsampled=bool(raw_preview.get("input_preview_was_downsampled", False)),
        input_preview_crs=raw_preview.get("input_preview_crs"),
        input_preview_transform_gdal=raw_preview.get("input_preview_transform_gdal"),
        input_preview_tile_count=int(raw_preview.get("input_preview_tile_count", 0)),
        input_preview_debug=raw_preview.get("input_preview_debug"),
        planet_preview_png_base64=planet_preview.get("planet_preview_png_base64"),
        planet_preview_width=planet_preview.get("planet_preview_width"),
        planet_preview_height=planet_preview.get("planet_preview_height"),
        planet_preview_native_width=planet_preview.get("planet_preview_native_width"),
        planet_preview_native_height=planet_preview.get("planet_preview_native_height"),
        planet_preview_was_downsampled=bool(planet_preview.get("planet_preview_was_downsampled", False)),
        planet_preview_crs=planet_preview.get("planet_preview_crs"),
        planet_preview_transform_gdal=planet_preview.get("planet_preview_transform_gdal"),
        planet_preview_tile_count=int(planet_preview.get("planet_preview_tile_count", 0)),
        planet_preview_debug=planet_preview.get("planet_preview_debug"),
        outputs=outputs,
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/info")
def info():
    return _backend_settings_for_info()


@app.get("/meta_patches")
def meta_patches(lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    lo_min, la_min, lo_max, la_max = _normalize_bbox(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)

    if lo_max < AR_LON_MIN or lo_min > AR_LON_MAX or la_max < AR_LAT_MIN or la_min > AR_LAT_MAX:
        raise HTTPException(status_code=400, detail="Requested bbox does not intersect the supported Arkansas region.")

    return {
        "bbox": {
            "lon_min": lo_min,
            "lat_min": la_min,
            "lon_max": lo_max,
            "lat_max": la_max,
        },
        "meta_patches": _meta_patches_intersecting_bbox(
            lon_min=lo_min,
            lat_min=la_min,
            lon_max=lo_max,
            lat_max=la_max,
        ),
    }


@app.get("/jobs/{job_id}", response_model=FastDiffSRJobStatusResponse)
def job_status(job_id: str):
    out_root = _resolve_path(os.environ.get("DEEPSAT_FASTDIFFSR_OUT_ROOT", "./fastdiffsr_preds"))
    jobs_dir = _jobs_dir(out_root)
    job = _read_job_file(jobs_dir, job_id)

    bbox_raw = dict(job.get("bbox") or {})
    bbox_obj = BBox(
        lon_min=float(bbox_raw.get("lon_min")),
        lat_min=float(bbox_raw.get("lat_min")),
        lon_max=float(bbox_raw.get("lon_max")),
        lat_max=float(bbox_raw.get("lat_max")),
    )

    outputs = [FastDiffSROutput(**o) for o in list(job.get("outputs") or [])]
    progress = dict(job.get("progress") or {})
    phase = str(job.get("phase", job.get("state", "queued")))
    if phase == "running":
        phase = "generating_sr"
    if phase not in {"queued", "downloading_raw", "generating_sr", "completed", "failed", "interrupted"}:
        phase = str(job.get("state", "queued"))
        if phase == "running":
            phase = "generating_sr"

    return FastDiffSRJobStatusResponse(
        job_id=str(job["job_id"]),
        job_key=str(job["job_key"]),
        state=str(job["state"]),
        job_kind=str(job.get("job_kind", "month_seed")),
        job_scope=str(job.get("job_scope", "full_month")),
        phase=phase,
        phase_message=(str(job.get("phase_message")) if job.get("phase_message") is not None else None),
        created_at=str(job["created_at"]),
        updated_at=str(job["updated_at"]),
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        year=int(job["year"]),
        month=int(job["month"]),
        bbox=bbox_obj,
        output_dir=str(job["output_dir"]),
        status_url=_status_url(str(job["job_id"])),
        progress={
            "total": int(progress.get("total", 0)),
            "done": int(progress.get("done", 0)),
            "written": int(progress.get("written", 0)),
            "skipped": int(progress.get("skipped", 0)),
            "failed": int(progress.get("failed", 0)),
        },
        date_plan_summary=(dict(job.get("date_plan_summary")) if isinstance(job.get("date_plan_summary"), dict) else None),
        missing_meta_patches=[str(x) for x in list(job.get("missing_meta_patches") or [])],
        outputs=outputs,
        errors=[str(x) for x in list(job.get("errors") or [])],
    )


@app.post(
    "/predict",
    response_model=Union[FastDiffSRPredictResponse, FastDiffSRJobAcceptedResponse],
    responses={202: {"model": FastDiffSRJobAcceptedResponse}},
)
def predict(req: FastDiffSRPredictRequest):
    lon_min, lat_min, lon_max, lat_max = _normalize_bbox(
        lon_min=req.bbox.lon_min,
        lat_min=req.bbox.lat_min,
        lon_max=req.bbox.lon_max,
        lat_max=req.bbox.lat_max,
    )

    if lon_max < AR_LON_MIN or lon_min > AR_LON_MAX or lat_max < AR_LAT_MIN or lat_min > AR_LAT_MAX:
        raise HTTPException(status_code=400, detail="Requested bbox does not intersect the supported Arkansas region.")

    settings = _load_backend_settings(year=int(req.year), month=int(req.month))

    config_path = Path(settings["config_path"])
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"FastDiffSR config not found: {config_path}")

    checkpoint_path = Path(settings["checkpoint_path"])
    if not checkpoint_path.exists() and not Path(f"{checkpoint_path}_gen.pth").exists():
        raise HTTPException(status_code=404, detail=f"FastDiffSR checkpoint not found: {checkpoint_path}")

    try:
        resume_state = _resolve_fastdiffsr_resume_state(str(checkpoint_path))
    except SystemExit as exc:
        raise HTTPException(status_code=500, detail=f"Invalid DEEPSAT_FASTDIFFSR_CHECKPOINT: {exc}") from exc

    out_dir = Path(settings["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_patches = _meta_patches_intersecting_bbox(
        lon_min=float(lon_min),
        lat_min=float(lat_min),
        lon_max=float(lon_max),
        lat_max=float(lat_max),
    )
    mosaic_fp = settings.get("mosaic_path")
    mosaic_ready = False
    if (
        str(settings["date_policy"]) != "all"
        and isinstance(mosaic_fp, Path)
        and _month_mosaic_intersects_bbox(mosaic_fp=mosaic_fp, bbox_wgs84=(lon_min, lat_min, lon_max, lat_max))
    ):
        mosaic_ready = True

    # Expected count is derived from the canonical 20x20 Arkansas grid.
    # For existing output preview selection, prefer actual GeoTIFF bounds because
    # filename/grid naming may not match the canonical grid indexing orientation.
    ready_meta_patches_actual: List[str] = []
    if mosaic_ready:
        ready_meta_patches_actual = [str(x) for x in meta_patches]
    elif str(settings["date_policy"]) != "all":
        ready_meta_patches_actual = _existing_output_tiles_intersecting_bbox(
            out_dir=out_dir,
            bbox_wgs84=(lon_min, lat_min, lon_max, lat_max),
            date_policy=str(settings["date_policy"]),
        )

    canonical_missing_meta_patches: List[str] = []
    for mp in meta_patches:
        existing = _existing_outputs_for_meta_patch(out_dir=out_dir, meta_patch=mp, date_policy=str(settings["date_policy"]))
        if not existing:
            canonical_missing_meta_patches.append(mp)

    missing_meta_patches: List[str] = []
    if str(settings["date_policy"]) != "all":
        # Prefer month mosaic readiness if configured, otherwise fall back to per-tile GeoTIFF checks.
        if mosaic_ready or (ready_meta_patches_actual and len(ready_meta_patches_actual) >= len(meta_patches)):
            missing_meta_patches = []
        else:
            missing_meta_patches = canonical_missing_meta_patches or [str(x) for x in meta_patches]
    else:
        missing_meta_patches = canonical_missing_meta_patches

    if not missing_meta_patches:
        return _build_existing_response(
            req=req,
            settings=settings,
            meta_patches=ready_meta_patches_actual or meta_patches,
            bbox=(lon_min, lat_min, lon_max, lat_max),
            resume_state=resume_state,
        )

    settings_for_job = {
        "raw_root": str(settings["raw_root"]),
        "raw_root_template": str(settings["raw_root_template"]),
        "out_root": str(settings["out_root"]),
        "out_dir": str(settings["out_dir"]),
        "config_path": str(settings["config_path"]),
        "checkpoint_path": str(settings["checkpoint_path"]),
        "device": str(settings["device"]),
        "scale": int(settings["scale"]),
        "lr_tile_size": int(settings["lr_tile_size"]),
        "date_policy": str(settings["date_policy"]),
        "cloud_thresh_fastdiffsr": float(settings["cloud_thresh_fastdiffsr"]),
        "blank_frac_max": float(settings.get("blank_frac_max", 0.98)),
        "force": bool(settings["force"]),
        "ee_project": str(settings["ee_project"]),
        "download_data_root": str(settings["download_data_root"]),
        "download_workers": int(settings["download_workers"]),
        "download_cloud_thresh": float(settings["download_cloud_thresh"]),
        "download_verify": bool(settings["download_verify"]),
        "download_strict": bool(settings["download_strict"]),
        "download_fast_skip": bool(settings["download_fast_skip"]),
    }

    all_meta_patches = _all_meta_patches()
    key = _month_seed_job_key(year=int(req.year), month=int(req.month), settings=settings)

    with _JOB_LOCK:
        existing_job_id = _JOB_ACTIVE_BY_KEY.get(key)

    if existing_job_id:
        out_root = _resolve_path(str(settings["out_root"]))
        jobs_dir = _jobs_dir(out_root)
        try:
            job = _read_job_file(jobs_dir, existing_job_id)
            state = str(job.get("state", "queued"))
            if state in JOB_ACTIVE_STATES:
                phase = str(job.get("phase", "queued"))
                if phase not in {"queued", "downloading_raw", "generating_sr"}:
                    phase = "generating_sr" if state == "running" else "queued"
                body = FastDiffSRJobAcceptedResponse(
                    state="running" if state == "running" else "queued",
                    phase=phase,
                    job_id=str(existing_job_id),
                    status_url=_status_url(str(existing_job_id)),
                    year=int(req.year),
                    month=int(req.month),
                    total_meta_patches=len(all_meta_patches),
                    missing_meta_patches=len(list(job.get("missing_meta_patches") or all_meta_patches)),
                    missing_meta_patch_ids=[str(x) for x in list(job.get("missing_meta_patches") or all_meta_patches)],
                    requested_meta_patches=len(meta_patches),
                    requested_meta_patch_ids=[str(x) for x in meta_patches],
                    output_dir=str(out_dir),
                    message="Arkansas month SR is being prepared in backend. Please poll job status.",
                )
                return JSONResponse(status_code=202, content=body.model_dump())
        except HTTPException:
            pass

    latest_same_key = _find_latest_job_by_key(out_root=Path(settings["out_root"]), job_key=key)
    if latest_same_key and str(latest_same_key.get("state")) == "completed":
        raise HTTPException(
            status_code=404,
            detail=(
                "Requested bbox SR tiles are not available after the most recent month generation job. "
                f"Missing requested meta-patches: {missing_meta_patches}"
            ),
        )

    job_id = f"job_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(4)}"
    now = _utc_now_iso()
    start_day, end_day = _month_range(int(req.year), int(req.month))

    job = {
        "job_id": job_id,
        "job_key": key,
        "state": "queued",
        "job_kind": "month_seed",
        "job_scope": "full_month",
        "phase": "queued",
        "phase_message": "Queued month-seed job for Arkansas raw download/SR generation.",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "year": int(req.year),
        "month": int(req.month),
        "bbox": {
            "lon_min": float(lon_min),
            "lat_min": float(lat_min),
            "lon_max": float(lon_max),
            "lat_max": float(lat_max),
        },
        "requested_bbox": {
            "lon_min": float(lon_min),
            "lat_min": float(lat_min),
            "lon_max": float(lon_max),
            "lat_max": float(lat_max),
        },
        "requested_meta_patches": [str(x) for x in meta_patches],
        "start_day": start_day,
        "end_day": end_day,
        "output_dir": str(out_dir),
        "status_url": _status_url(job_id),
        "meta_patches": [str(x) for x in all_meta_patches],
        "missing_meta_patches": [str(x) for x in all_meta_patches],
        "progress": {
            "total": len(all_meta_patches),
            "done": 0,
            "written": 0,
            "skipped": 0,
            "failed": 0,
        },
        "outputs": [],
        "errors": [],
        "settings": settings_for_job,
    }

    _enqueue_job(job)

    body = FastDiffSRJobAcceptedResponse(
        state="queued",
        phase="queued",
        job_id=job_id,
        status_url=_status_url(job_id),
        year=int(req.year),
        month=int(req.month),
        total_meta_patches=len(all_meta_patches),
        missing_meta_patches=len(all_meta_patches),
        missing_meta_patch_ids=[str(x) for x in all_meta_patches],
        requested_meta_patches=len(meta_patches),
        requested_meta_patch_ids=[str(x) for x in meta_patches],
        output_dir=str(out_dir),
        message="Arkansas month SR job was queued. Backend may download raw data first, then generate SR.",
    )
    return JSONResponse(status_code=202, content=body.model_dump())
