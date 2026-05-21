#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds as window_from_bounds
from tqdm import tqdm


PLANET_DEFAULT_ROOT = Path("/home/thanyu/planet_data/analytic_8b_sr_udm2")
ORTHO_DEFAULT_ROOT = Path("./Orthophotos")
DEFAULT_DATE_MANIFEST = Path(__file__).with_name("orthophoto_date_fallbacks.json")
TIFF_EXTENSIONS = (".tif", ".tiff")


@dataclass(frozen=True)
class RasterInfo:
    path: Path
    stem: str
    date: date
    crs: Any
    bounds: tuple[float, float, float, float]
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FastDiffSR training data using Planet scenes as LR and Orthophotos "
            "as HR with spatial overlap and nearest-date matching."
        )
    )
    parser.add_argument("--planet-root", type=Path, default=PLANET_DEFAULT_ROOT)
    parser.add_argument("--ortho-root", type=Path, default=ORTHO_DEFAULT_ROOT)
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("FastDiffSR/FastDiffSR/dataset/train_planet_ortho_64_256"),
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path("FastDiffSR/FastDiffSR/dataset/val_planet_ortho_64_256"),
    )
    parser.add_argument("--planet-patch-size", type=int, default=64)
    parser.add_argument("--hr-patch-size", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-day-diff", type=int, default=30)
    parser.add_argument("--black-threshold", type=int, default=50)
    parser.add_argument("--max-black-ratio", type=float, default=0.5)
    parser.add_argument("--min-valid-hr-ratio", type=float, default=0.95)
    parser.add_argument("--ortho-date-manifest", type=Path, default=DEFAULT_DATE_MANIFEST)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--q-low", type=float, default=2.0)
    parser.add_argument("--q-high", type=float, default=98.0)
    parser.add_argument("--sample-max-dim", type=int, default=1024)
    parser.add_argument(
        "--ortho-stats-mode",
        choices=("local", "global"),
        default="local",
        help=(
            "Use per-patch local RGB percentiles for NOAA normalization, or global "
            "orthophoto percentiles. Local avoids reading huge merged orthophotos."
        ),
    )
    parser.add_argument("--save-debug-png", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")
    if args.planet_patch_size <= 0 or args.hr_patch_size <= 0:
        raise ValueError("Patch sizes must be positive.")
    if args.hr_patch_size % args.planet_patch_size != 0:
        raise ValueError("--hr-patch-size must be divisible by --planet-patch-size.")
    expected_scale = args.hr_patch_size // args.planet_patch_size
    if args.scale != expected_scale:
        raise ValueError(
            f"--scale={args.scale} does not match hr/lr ratio {args.hr_patch_size}/{args.planet_patch_size}={expected_scale}."
        )
    if not (0.0 <= args.patch_overlap < 1.0):
        raise ValueError("--patch-overlap must be in [0, 1).")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1).")
    if not (0.0 <= args.q_low < args.q_high <= 100.0):
        raise ValueError("--q-low and --q-high must satisfy 0 <= q_low < q_high <= 100.")
    if not (0.0 <= args.max_black_ratio <= 1.0):
        raise ValueError("--max-black-ratio must be in [0, 1].")
    if not (0.0 <= args.min_valid_hr_ratio <= 1.0):
        raise ValueError("--min-valid-hr-ratio must be in [0, 1].")
    if args.sample_max_dim < 32:
        raise ValueError("--sample-max-dim must be at least 32.")


def normalize_name_key(name: str) -> str:
    return " ".join(name.lower().strip().split())


def extract_date_from_text(value: str | None) -> date | None:
    if not value:
        return None
    value = str(value)

    for pattern in (
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
        r"(?<!\d)(20\d{2})[-_/:\.](\d{2})[-_/:\.](\d{2})(?!\d)",
    ):
        match = re.search(pattern, value)
        if not match:
            continue
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
    return None


def extract_year_from_text(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(value))
    if not match:
        return None
    return int(match.group(1))


def load_date_overrides(path: Path | None) -> dict[str, date]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Orthophoto date manifest not found: {path}")
    payload = json.loads(path.read_text())
    overrides: dict[str, date] = {}
    for key, value in payload.items():
        parsed = extract_date_from_text(str(value))
        if parsed is None:
            raise ValueError(f"Could not parse date {value!r} for {key!r} in {path}")
        overrides[normalize_name_key(str(key))] = parsed
    return overrides


def collect_tag_strings(ds: rasterio.DatasetReader) -> Iterable[str]:
    for key, value in ds.tags().items():
        yield f"{key}={value}"
    try:
        for namespace in ds.tag_namespaces():
            for key, value in ds.tags(ns=namespace).items():
                yield f"{namespace}:{key}={value}"
    except Exception:
        return


def resolve_planet_date(path: Path, ds: rasterio.DatasetReader) -> date | None:
    parsed = extract_date_from_text(path.stem)
    if parsed is not None:
        return parsed
    for tag_text in collect_tag_strings(ds):
        parsed = extract_date_from_text(tag_text)
        if parsed is not None:
            return parsed
    return None


def resolve_ortho_date(
    path: Path,
    ds: rasterio.DatasetReader,
    overrides: dict[str, date],
) -> date | None:
    parsed = extract_date_from_text(path.stem)
    if parsed is not None:
        return parsed
    parsed_year = extract_year_from_text(path.stem)
    if parsed_year is not None:
        return date(parsed_year, 7, 1)
    for tag_text in collect_tag_strings(ds):
        parsed = extract_date_from_text(tag_text)
        if parsed is not None:
            return parsed
        parsed_year = extract_year_from_text(tag_text)
        if parsed_year is not None:
            return date(parsed_year, 7, 1)
    return overrides.get(normalize_name_key(path.name))


def is_planet_scene_tiff(path: Path) -> bool:
    lower = path.name.lower()
    if path.suffix.lower() not in TIFF_EXTENSIONS:
        return False
    if "udm" in lower:
        return False
    return "analytic" in lower or "sr_8b" in lower or "psscene" in str(path).lower()


def resolve_rgb_band_indices(ds: rasterio.DatasetReader) -> tuple[int, int, int]:
    if ds.count < 3:
        raise ValueError(f"{ds.name} has only {ds.count} band(s); need at least 3.")

    descriptions = [desc.lower() if isinstance(desc, str) else "" for desc in ds.descriptions]
    desired_names = ("red", "green", "blue")
    band_indices: list[int | None] = []

    for desired_name in desired_names:
        matched_index = next(
            (idx + 1 for idx, desc in enumerate(descriptions) if desc == desired_name),
            None,
        )
        if matched_index is None:
            matched_index = next(
                (idx + 1 for idx, desc in enumerate(descriptions) if desired_name in desc),
                None,
            )
        band_indices.append(matched_index)

    if all(index is not None for index in band_indices):
        return tuple(int(index) for index in band_indices)

    if ds.count >= 6:
        return (6, 4, 2)
    return (1, 2, 3)


def compute_rgb_stats(
    ds: rasterio.DatasetReader,
    band_indices: tuple[int, int, int],
    *,
    q_low: float,
    q_high: float,
    sample_max_dim: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    height = ds.height
    width = ds.width
    scale = max(height / sample_max_dim, width / sample_max_dim, 1.0)
    out_height = max(1, int(round(height / scale)))
    out_width = max(1, int(round(width / scale)))
    sample = ds.read(
        list(band_indices),
        out_shape=(len(band_indices), out_height, out_width),
        resampling=Resampling.average,
    ).astype(np.float32, copy=False)

    stats: list[tuple[float, float]] = []
    for band_sample in sample:
        finite = band_sample[np.isfinite(band_sample)]
        if ds.nodata is not None:
            finite = finite[finite != float(ds.nodata)]
        if finite.size == 0:
            stats.append((0.0, 255.0))
            continue
        lo = float(np.percentile(finite, q_low))
        hi = float(np.percentile(finite, q_high))
        if hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
        stats.append((lo, hi))
    return tuple(stats)  # type: ignore[return-value]


def normalize_rgb_array(
    img: np.ndarray,
    stats: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    out = np.zeros(img.shape, dtype=np.uint8)
    for channel in range(min(img.shape[-1], 3)):
        lo, hi = stats[channel]
        if hi <= lo:
            continue
        band = img[:, :, channel].astype(np.float32, copy=False)
        scaled = (np.clip(band, lo, hi) - lo) / (hi - lo)
        out[:, :, channel] = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    return out


def compute_rgb_array_stats(
    img: np.ndarray,
    *,
    q_low: float,
    q_high: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    stats: list[tuple[float, float]] = []
    valid = valid_mask.astype(bool, copy=False) if valid_mask is not None else None
    for channel in range(min(img.shape[-1], 3)):
        band = img[:, :, channel].astype(np.float32, copy=False)
        if valid is not None:
            finite = band[np.isfinite(band) & valid]
        else:
            finite = band[np.isfinite(band)]
        finite = finite[finite != 0]
        if finite.size == 0:
            stats.append((0.0, 255.0))
            continue
        lo = float(np.percentile(finite, q_low))
        hi = float(np.percentile(finite, q_high))
        if hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
        stats.append((lo, hi))
    while len(stats) < 3:
        stats.append((0.0, 255.0))
    return tuple(stats)  # type: ignore[return-value]


def black_ratio(img_uint8: np.ndarray, threshold: int) -> float:
    return float(np.all(img_uint8 <= threshold, axis=-1).mean())


def bright_patch_stats(img_uint8: np.ndarray, threshold: int) -> tuple[float, float]:
    brightness = img_uint8.astype(np.float32, copy=False).mean(axis=-1)
    return float(brightness.mean()), float((brightness >= threshold).mean())


def white_dark_patch_stats(
    img_uint8: np.ndarray,
    *,
    white_brightness_threshold: int,
    white_saturation_threshold: float,
    dark_threshold: int,
) -> tuple[float, float]:
    arr = img_uint8.astype(np.float32, copy=False)
    brightness = arr.mean(axis=-1)
    max_rgb = arr.max(axis=-1)
    min_rgb = arr.min(axis=-1)
    saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1.0)
    white_ratio = (
        (brightness >= int(white_brightness_threshold))
        & (saturation <= float(white_saturation_threshold))
    ).mean()
    dark_ratio = (brightness <= int(dark_threshold)).mean()
    return float(white_ratio), float(dark_ratio)


def resolve_patch_step(patch_size: int, overlap: float) -> int:
    step = int(round(patch_size * (1.0 - overlap)))
    if step <= 0:
        raise ValueError("Patch step resolved to <= 0. Lower --patch-overlap.")
    return step


def iter_complete_windows(width: int, height: int, patch_size: int, step: int) -> Iterable[Window]:
    max_x = width - patch_size
    max_y = height - patch_size
    if max_x < 0 or max_y < 0:
        return
    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            yield Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
            x += step
        y += step


def bounds_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    return right > left and top > bottom


def union_bounds(
    bounds_list: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    bounds_seq = list(bounds_list)
    if not bounds_seq:
        return None
    return (
        min(bounds[0] for bounds in bounds_seq),
        min(bounds[1] for bounds in bounds_seq),
        max(bounds[2] for bounds in bounds_seq),
        max(bounds[3] for bounds in bounds_seq),
    )


def bounds_center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)


def bounds_center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    a_center = bounds_center(a)
    b_center = bounds_center(b)
    return float(np.hypot(a_center[0] - b_center[0], a_center[1] - b_center[1]))


def format_bounds(bounds: tuple[float, float, float, float]) -> str:
    return "(" + ", ".join(f"{value:.6f}" for value in bounds) + ")"


def transform_bounds_safe(
    bounds: tuple[float, float, float, float],
    src_crs: Any,
    dst_crs: Any,
) -> tuple[float, float, float, float] | None:
    if src_crs is None or dst_crs is None:
        return None
    if src_crs == dst_crs:
        return bounds
    try:
        transformed = transform_bounds(src_crs, dst_crs, *bounds, densify_pts=21)
    except Exception:
        return None
    return tuple(float(v) for v in transformed)


def collect_planet_scenes(root: Path) -> list[Path]:
    scenes: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and is_planet_scene_tiff(path):
            scenes.append(path)
    return sorted(set(scenes))


def collect_tiffs(root: Path) -> list[Path]:
    return sorted({path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS})


def describe_planet_scene(path: Path) -> RasterInfo | tuple[None, str]:
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None, "planet_missing_crs"
            parsed_date = resolve_planet_date(path, ds)
            if parsed_date is None:
                return None, "planet_missing_date"
            return RasterInfo(
                path=path,
                stem=path.stem,
                date=parsed_date,
                crs=ds.crs,
                bounds=tuple(float(v) for v in ds.bounds),
                width=ds.width,
                height=ds.height,
            )
    except Exception as exc:
        return None, f"planet_open_failed:{exc}"


def describe_ortho(path: Path, overrides: dict[str, date]) -> RasterInfo | tuple[None, str]:
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None, "ortho_missing_crs"
            parsed_date = resolve_ortho_date(path, ds, overrides)
            if parsed_date is None:
                return None, "ortho_missing_date"
            return RasterInfo(
                path=path,
                stem=path.stem,
                date=parsed_date,
                crs=ds.crs,
                bounds=tuple(float(v) for v in ds.bounds),
                width=ds.width,
                height=ds.height,
            )
    except Exception as exc:
        return None, f"ortho_open_failed:{exc}"


def choose_best_ortho(
    planet: RasterInfo,
    orthos: list[RasterInfo],
    *,
    max_day_diff: int,
) -> tuple[RasterInfo | None, int | None]:
    candidates: list[tuple[int, str, RasterInfo]] = []
    for ortho in orthos:
        ortho_bounds_in_planet = transform_bounds_safe(ortho.bounds, ortho.crs, planet.crs)
        if ortho_bounds_in_planet is None:
            continue
        if not bounds_intersect(planet.bounds, ortho_bounds_in_planet):
            continue
        day_diff = abs((planet.date - ortho.date).days)
        candidates.append((day_diff, str(ortho.path), ortho))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    day_diff, _, ortho = candidates[0]
    if day_diff > max_day_diff:
        return None, day_diff
    return ortho, day_diff


def build_no_pair_diagnostics(
    planets: list[RasterInfo],
    orthos: list[RasterInfo],
    *,
    max_examples: int = 3,
) -> str:
    ortho_wgs84: list[tuple[RasterInfo, tuple[float, float, float, float]]] = []
    for ortho in orthos:
        bounds_wgs84 = transform_bounds_safe(ortho.bounds, ortho.crs, "EPSG:4326")
        if bounds_wgs84 is not None:
            ortho_wgs84.append((ortho, bounds_wgs84))

    ortho_union = union_bounds(bounds for _, bounds in ortho_wgs84)
    lines = [
        f"usable_planets={len(planets)}",
        f"usable_orthophotos={len(orthos)}",
    ]

    if ortho_union is None:
        lines.append("Could not transform Orthophoto bounds to EPSG:4326 for diagnostics.")
        return "\n".join(lines)

    lines.append(f"orthophoto_union_epsg4326={format_bounds(ortho_union)}")

    overlapping_planets = 0
    nearest_planets: list[tuple[float, RasterInfo, tuple[float, float, float, float]]] = []
    for planet in planets:
        bounds_wgs84 = transform_bounds_safe(planet.bounds, planet.crs, "EPSG:4326")
        if bounds_wgs84 is None:
            continue
        if bounds_intersect(bounds_wgs84, ortho_union):
            overlapping_planets += 1
        nearest_planets.append((bounds_center_distance(bounds_wgs84, ortho_union), planet, bounds_wgs84))

    lines.append(f"planets_overlapping_orthophoto_union={overlapping_planets}")
    if overlapping_planets == 0:
        lines.append(
            "No usable Planet scene overlaps the Orthophoto area. "
            "This usually means --planet-root points to a different region."
        )

    for _, ortho, bounds_wgs84 in sorted(ortho_wgs84, key=lambda item: str(item[0].path))[:max_examples]:
        lines.append(
            "orthophoto_example="
            f"{ortho.path} date={ortho.date.isoformat()} bounds_epsg4326={format_bounds(bounds_wgs84)}"
        )

    nearest_planets.sort(key=lambda item: (item[0], str(item[1].path)))
    for distance, planet, bounds_wgs84 in nearest_planets[:max_examples]:
        lines.append(
            "nearest_planet="
            f"{planet.path} date={planet.date.isoformat()} "
            f"center_distance_deg={distance:.6f} bounds_epsg4326={format_bounds(bounds_wgs84)}"
        )
    return "\n".join(lines)


def save_rgb_tiff(path: Path, img_uint8: np.ndarray) -> None:
    Image.fromarray(img_uint8, mode="RGB").save(path, format="TIFF")


def save_rgb_png(path: Path, img_uint8: np.ndarray) -> None:
    Image.fromarray(img_uint8, mode="RGB").save(path, format="PNG")


def ensure_clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    if any(path.rglob("*")):
        existing_files = [p for p in path.rglob("*") if p.is_file()]
        if existing_files:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Please use a new directory or clear it first."
            )


def prepare_output_root(
    root: Path,
    *,
    l_res: int,
    r_res: int,
    save_debug_png: bool,
    allow_existing: bool = False,
) -> dict[str, Path]:
    if not allow_existing:
        ensure_clean_output_dir(root)
    dirs = {
        "root": root,
        "lr": root / f"lr_{l_res}",
        "hr": root / f"hr_{r_res}",
        "sr": root / f"sr_{l_res}_{r_res}",
    }
    if save_debug_png:
        dirs.update(
            {
                "debug_lr": root / "debug_png" / f"lr_{l_res}",
                "debug_hr": root / "debug_png" / f"hr_{r_res}",
                "debug_sr": root / "debug_png" / f"sr_{l_res}_{r_res}",
            }
        )
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def default_manifest_path(train_out: Path, val_out: Path) -> Path:
    try:
        common = Path(os.path.commonpath([str(train_out.resolve()), str(val_out.resolve())]))
    except ValueError:
        common = Path.cwd()
    return common / "pairs_manifest.jsonl"


def split_pairs_by_source(
    pair_keys: list[str],
    *,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    shuffled = sorted(pair_keys)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or val_ratio <= 0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)
    val_keys = set(shuffled[:val_count])
    return {key: ("val" if key in val_keys else "train") for key in shuffled}


def serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def write_manifest_line(manifest_fp, payload: dict[str, Any]) -> None:
    manifest_fp.write(json.dumps(payload, sort_keys=True) + "\n")


def read_rgb_patch(
    ds: rasterio.DatasetReader,
    band_indices: tuple[int, int, int],
    window: Window,
) -> np.ndarray:
    patch = ds.read(list(band_indices), window=window)
    return np.moveaxis(patch, 0, -1)


def build_valid_mask(
    ds: rasterio.DatasetReader,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    source_mask = ds.dataset_mask()
    dst_mask = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=source_mask,
        destination=dst_mask,
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst_mask


def render_ortho_patch(
    ortho_ds: rasterio.DatasetReader,
    band_indices: tuple[int, int, int],
    stats: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
    *,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    q_low: float | None = None,
    q_high: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dst_bounds = rasterio.transform.array_bounds(dst_shape[0], dst_shape[1], dst_transform)
    src_bounds = transform_bounds_safe(dst_bounds, dst_crs, ortho_ds.crs)
    if src_bounds is None:
        return np.zeros((dst_shape[0], dst_shape[1], 3), dtype=np.uint8), np.zeros(dst_shape, dtype=np.uint8)
    src_window = window_from_bounds(*src_bounds, transform=ortho_ds.transform)
    col_off = math.floor(src_window.col_off) - 2
    row_off = math.floor(src_window.row_off) - 2
    width = math.ceil(src_window.width + (src_window.col_off - math.floor(src_window.col_off))) + 4
    height = math.ceil(src_window.height + (src_window.row_off - math.floor(src_window.row_off))) + 4
    read_window = Window(col_off=col_off, row_off=row_off, width=max(1, width), height=max(1, height))
    src_transform = ortho_ds.window_transform(read_window)
    src_nodata = ortho_ds.nodata if ortho_ds.nodata is not None else 0
    dst = np.zeros((len(band_indices), dst_shape[0], dst_shape[1]), dtype=np.float32)
    for dst_idx, src_idx in enumerate(band_indices):
        source = ortho_ds.read(
            src_idx,
            window=read_window,
            boundless=True,
            fill_value=src_nodata,
        )
        reproject(
            source=source,
            destination=dst[dst_idx],
            src_transform=src_transform,
            src_crs=ortho_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=ortho_ds.nodata if ortho_ds.nodata is not None else 0,
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
    rgb = np.moveaxis(dst, 0, -1)
    valid_mask = np.any(np.isfinite(rgb) & (rgb != 0), axis=-1).astype(np.uint8) * 255
    if stats is None:
        stats = compute_rgb_array_stats(
            rgb,
            q_low=2.0 if q_low is None else float(q_low),
            q_high=98.0 if q_high is None else float(q_high),
            valid_mask=valid_mask,
        )
    return normalize_rgb_array(rgb, stats), valid_mask


def build_sr_patch(lr_patch_uint8: np.ndarray, hr_size: int) -> np.ndarray:
    lr_img = Image.fromarray(lr_patch_uint8, mode="RGB")
    sr_img = lr_img.resize((hr_size, hr_size), resample=Image.BICUBIC)
    return np.asarray(sr_img, dtype=np.uint8)


def parse_reconstruction_ortho_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FastDiffSR data from Planet-quality Sentinel-2 reconstructions as LR "
            "and NOAA/Lexington orthophotos as HR."
        )
    )
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--reconstruction-manifest", type=Path, required=True)
    parser.add_argument("--ortho-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--exclude-ortho-glob",
        action="append",
        default=["*Merged.tif", "*Merged.tiff"],
        help="Orthophoto filename glob to skip. Can be repeated. Defaults skip merged NOAA mosaics.",
    )
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--lr-patch-size", type=int, default=64)
    parser.add_argument("--hr-patch-size", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--reconstruction-crop-stride", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-ortho-day-diff",
        type=int,
        default=None,
        help=(
            "Only pair a reconstruction crop with orthophotos within this many days "
            "of the source Sentinel date. Omit to use spatial overlap only."
        ),
    )
    parser.add_argument("--black-threshold", type=int, default=30)
    parser.add_argument("--max-black-ratio", type=float, default=0.0)
    parser.add_argument("--min-valid-hr-ratio", type=float, default=0.95)
    parser.add_argument("--lr-bright-threshold", type=int, default=245)
    parser.add_argument("--max-lr-bright-ratio", type=float, default=0.50)
    parser.add_argument("--max-lr-mean-brightness", type=float, default=245.0)
    parser.add_argument("--min-lr-mean-brightness", type=float, default=0.0)
    parser.add_argument("--lr-white-brightness-threshold", type=int, default=170)
    parser.add_argument("--lr-white-saturation-threshold", type=float, default=0.25)
    parser.add_argument("--max-lr-white-ratio", type=float, default=1.0)
    parser.add_argument("--lr-dark-threshold", type=int, default=45)
    parser.add_argument("--max-lr-dark-ratio", type=float, default=1.0)
    parser.add_argument("--hr-bright-threshold", type=int, default=245)
    parser.add_argument("--max-hr-bright-ratio", type=float, default=0.50)
    parser.add_argument("--max-hr-mean-brightness", type=float, default=245.0)
    parser.add_argument("--min-hr-mean-brightness", type=float, default=0.0)
    parser.add_argument("--hr-white-brightness-threshold", type=int, default=170)
    parser.add_argument("--hr-white-saturation-threshold", type=float, default=0.25)
    parser.add_argument("--max-hr-white-ratio", type=float, default=1.0)
    parser.add_argument("--hr-dark-threshold", type=int, default=45)
    parser.add_argument("--max-hr-dark-ratio", type=float, default=1.0)
    parser.add_argument("--ortho-date-manifest", type=Path, default=DEFAULT_DATE_MANIFEST)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--q-low", type=float, default=2.0)
    parser.add_argument("--q-high", type=float, default=98.0)
    parser.add_argument("--sample-max-dim", type=int, default=1024)
    parser.add_argument(
        "--ortho-stats-mode",
        choices=("local", "global"),
        default="local",
        help=(
            "Use per-patch local RGB percentiles for NOAA normalization, or global "
            "orthophoto percentiles. Local avoids reading huge merged orthophotos."
        ),
    )
    parser.add_argument("--save-debug-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes to use by sharding reconstructions.",
    )
    parser.add_argument("--reconstruction-shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reconstruction-shard-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sample-id-offset", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--sample-id-stride", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def validate_reconstruction_ortho_args(args: argparse.Namespace) -> None:
    if args.lr_patch_size <= 0 or args.hr_patch_size <= 0:
        raise ValueError("Patch sizes must be positive.")
    if args.hr_patch_size % args.lr_patch_size != 0:
        raise ValueError("--hr-patch-size must be divisible by --lr-patch-size.")
    expected_scale = args.hr_patch_size // args.lr_patch_size
    if int(args.scale) != expected_scale:
        raise ValueError(
            f"--scale={args.scale} does not match {args.hr_patch_size}/{args.lr_patch_size}={expected_scale}."
        )
    if args.reconstruction_crop_stride <= 0:
        raise ValueError("--reconstruction-crop-stride must be positive.")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1).")
    if args.max_ortho_day_diff is not None and int(args.max_ortho_day_diff) < 0:
        raise ValueError("--max-ortho-day-diff must be non-negative.")
    if not (0.0 <= args.min_valid_hr_ratio <= 1.0):
        raise ValueError("--min-valid-hr-ratio must be in [0, 1].")
    if not (0.0 <= args.max_black_ratio <= 1.0):
        raise ValueError("--max-black-ratio must be in [0, 1].")
    for name in ("lr", "hr"):
        if not (0 <= int(getattr(args, f"{name}_bright_threshold")) <= 255):
            raise ValueError(f"--{name}-bright-threshold must be in [0, 255].")
        if not (0.0 <= float(getattr(args, f"max_{name}_bright_ratio")) <= 1.0):
            raise ValueError(f"--max-{name}-bright-ratio must be in [0, 1].")
        if not (0.0 <= float(getattr(args, f"max_{name}_mean_brightness")) <= 255.0):
            raise ValueError(f"--max-{name}-mean-brightness must be in [0, 255].")
        if not (0.0 <= float(getattr(args, f"min_{name}_mean_brightness")) <= 255.0):
            raise ValueError(f"--min-{name}-mean-brightness must be in [0, 255].")
        if float(getattr(args, f"min_{name}_mean_brightness")) > float(
            getattr(args, f"max_{name}_mean_brightness")
        ):
            raise ValueError(f"--min-{name}-mean-brightness must be <= --max-{name}-mean-brightness.")
        if not (0 <= int(getattr(args, f"{name}_white_brightness_threshold")) <= 255):
            raise ValueError(f"--{name}-white-brightness-threshold must be in [0, 255].")
        if not (0.0 <= float(getattr(args, f"{name}_white_saturation_threshold")) <= 1.0):
            raise ValueError(f"--{name}-white-saturation-threshold must be in [0, 1].")
        if not (0.0 <= float(getattr(args, f"max_{name}_white_ratio")) <= 1.0):
            raise ValueError(f"--max-{name}-white-ratio must be in [0, 1].")
        if not (0 <= int(getattr(args, f"{name}_dark_threshold")) <= 255):
            raise ValueError(f"--{name}-dark-threshold must be in [0, 255].")
        if not (0.0 <= float(getattr(args, f"max_{name}_dark_ratio")) <= 1.0):
            raise ValueError(f"--max-{name}-dark-ratio must be in [0, 1].")
    if int(args.num_workers) <= 0:
        raise ValueError("--num-workers must be positive.")
    if int(args.sample_id_stride) <= 0:
        raise ValueError("--sample-id-stride must be positive.")
    if int(args.sample_id_offset) < 0:
        raise ValueError("--sample-id-offset must be non-negative.")
    if args.reconstruction_shard_index is None and args.reconstruction_shard_count is not None:
        raise ValueError("--reconstruction-shard-count requires --reconstruction-shard-index.")
    if args.reconstruction_shard_index is not None:
        if args.reconstruction_shard_count is None:
            raise ValueError("--reconstruction-shard-index requires --reconstruction-shard-count.")
        if not (0 <= int(args.reconstruction_shard_index) < int(args.reconstruction_shard_count)):
            raise ValueError("--reconstruction-shard-index must satisfy 0 <= index < count.")


def load_reconstruction_records(manifest_path: Path, reconstruction_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") != "kept" or not payload.get("sample_id"):
                continue
            sample_id = str(payload["sample_id"])
            recon_path = reconstruction_root / f"{sample_id}.tif"
            if not recon_path.exists():
                continue
            item = dict(payload)
            item["reconstruction_path"] = str(recon_path)
            records.append(item)
    records.sort(key=lambda item: str(item["sample_id"]))
    return records


def filter_reconstruction_shard(
    records: list[dict[str, Any]],
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> list[dict[str, Any]]:
    if shard_index is None or shard_count is None or shard_count <= 1:
        return records
    return [record for idx, record in enumerate(records) if idx % shard_count == shard_index]


def worker_artifact_dir(train_out: Path, val_out: Path) -> Path:
    return default_manifest_path(train_out, val_out).parent / "_worker_runs_reconstruction_ortho"


def merge_worker_jsonl(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fp:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as in_fp:
                shutil.copyfileobj(in_fp, out_fp)


def collect_ortho_infos(
    roots: list[Path],
    date_overrides: dict[str, date],
    manifest_fp,
    *,
    exclude_globs: list[str] | None = None,
) -> list[RasterInfo]:
    orthos: list[RasterInfo] = []
    exclude_globs = exclude_globs or []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Orthophoto root not found: {root}")
        for path in collect_tiffs(root):
            matched_exclude = next((pattern for pattern in exclude_globs if path.match(pattern)), None)
            if matched_exclude is not None:
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "orthophoto",
                        "status": "skipped",
                        "reason": "ortho_excluded_by_glob",
                        "split": "skip",
                        "ortho_path": str(path),
                        "ortho_root": str(root),
                        "exclude_glob": matched_exclude,
                    },
                )
                continue
            described = describe_ortho(path, date_overrides)
            if isinstance(described, tuple):
                _, reason = described
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "orthophoto",
                        "status": "skipped",
                        "reason": reason,
                        "split": "skip",
                        "ortho_path": str(path),
                        "ortho_root": str(root),
                    },
                )
                continue
            orthos.append(described)
    orthos.sort(key=lambda item: str(item.path))
    return orthos


def stable_split(key: str, *, val_ratio: float, seed: int) -> str:
    if val_ratio <= 0.0:
        return "train"
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    return "val" if value < val_ratio else "train"


def iter_reconstruction_crop_windows(width: int, height: int, patch_size: int, stride: int) -> Iterable[Window]:
    max_x = int(width) - int(patch_size)
    max_y = int(height) - int(patch_size)
    if max_x < 0 or max_y < 0:
        return
    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            yield Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
            x += stride
        y += stride


def find_intersecting_orthos(
    bounds: tuple[float, float, float, float],
    crs: Any,
    orthos: list[RasterInfo],
) -> list[RasterInfo]:
    matches: list[RasterInfo] = []
    for ortho in orthos:
        ortho_bounds = transform_bounds_safe(ortho.bounds, ortho.crs, crs)
        if ortho_bounds is not None and bounds_intersect(bounds, ortho_bounds):
            matches.append(ortho)
    return matches


def build_ortho_bounds_index(
    orthos: list[RasterInfo],
    crs: Any,
) -> list[tuple[RasterInfo, tuple[float, float, float, float]]]:
    indexed: list[tuple[RasterInfo, tuple[float, float, float, float]]] = []
    for ortho in orthos:
        ortho_bounds = transform_bounds_safe(ortho.bounds, ortho.crs, crs)
        if ortho_bounds is not None:
            indexed.append((ortho, ortho_bounds))
    return indexed


def find_intersecting_orthos_indexed(
    bounds: tuple[float, float, float, float],
    indexed_orthos: list[tuple[RasterInfo, tuple[float, float, float, float]]],
) -> list[RasterInfo]:
    return [ortho for ortho, ortho_bounds in indexed_orthos if bounds_intersect(bounds, ortho_bounds)]


def filter_orthos_by_day_diff(
    orthos: list[RasterInfo],
    source_date: date,
    *,
    max_day_diff: int,
) -> tuple[list[RasterInfo], int | None]:
    candidates: list[tuple[int, str, RasterInfo]] = []
    for ortho in orthos:
        day_diff = abs((source_date - ortho.date).days)
        candidates.append((day_diff, str(ortho.path), ortho))
    if not candidates:
        return [], None
    candidates.sort(key=lambda item: (item[0], item[1]))
    closest_day_diff = candidates[0][0]
    return [ortho for day_diff, _, ortho in candidates if day_diff <= max_day_diff], closest_day_diff


def render_best_ortho_patch(
    orthos: list[RasterInfo],
    stats_cache: dict[str, tuple[tuple[int, int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]],
    *,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    q_low: float,
    q_high: float,
    sample_max_dim: int,
    stats_mode: str,
) -> tuple[np.ndarray, np.ndarray, RasterInfo] | None:
    best: tuple[float, np.ndarray, np.ndarray, RasterInfo] | None = None
    for ortho in orthos:
        with rasterio.open(ortho.path) as ortho_ds:
            if stats_mode == "global" and str(ortho.path) not in stats_cache:
                bands = resolve_rgb_band_indices(ortho_ds)
                stats = compute_rgb_stats(
                    ortho_ds,
                    bands,
                    q_low=q_low,
                    q_high=q_high,
                    sample_max_dim=sample_max_dim,
                )
                stats_cache[str(ortho.path)] = (bands, stats)
            if stats_mode == "global":
                bands, stats = stats_cache[str(ortho.path)]
            else:
                bands = resolve_rgb_band_indices(ortho_ds)
                stats = None
            hr_patch, valid_mask = render_ortho_patch(
                ortho_ds,
                bands,
                stats,
                dst_shape=dst_shape,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                q_low=q_low,
                q_high=q_high,
            )
            valid_ratio = float((valid_mask > 0).mean())
            if best is None or valid_ratio > best[0]:
                best = (valid_ratio, hr_patch, valid_mask, ortho)
    if best is None:
        return None
    _, hr_patch, valid_mask, ortho = best
    return hr_patch, valid_mask, ortho


def run_sharded_reconstruction_workers(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    worker_count = min(int(args.num_workers), len(records))
    manifest_path = (args.manifest_path or default_manifest_path(args.train_out, args.val_out)).resolve()
    shard_dir = worker_artifact_dir(args.train_out, args.val_out).resolve()

    train_dirs = prepare_output_root(
        args.train_out,
        l_res=args.lr_patch_size,
        r_res=args.hr_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=False,
    )
    val_dirs = prepare_output_root(
        args.val_out,
        l_res=args.lr_patch_size,
        r_res=args.hr_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=False,
    )
    del train_dirs, val_dirs

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    per_worker_limit = None
    if args.limit_samples is not None:
        per_worker_limit = max(1, (int(args.limit_samples) + worker_count - 1) // worker_count)

    script_path = Path(__file__).resolve()
    base_argv = sys.argv[1:]
    procs: list[tuple[int, subprocess.Popen[bytes], Path]] = []
    env = os.environ.copy()
    for shard_index in range(worker_count):
        shard_manifest = shard_dir / f"pairs_manifest.worker{shard_index:02d}.jsonl"
        cmd = [
            sys.executable,
            str(script_path),
            *base_argv,
            "--num-workers",
            "1",
            "--reconstruction-shard-index",
            str(shard_index),
            "--reconstruction-shard-count",
            str(worker_count),
            "--sample-id-offset",
            str(shard_index),
            "--sample-id-stride",
            str(worker_count),
            "--manifest-path",
            str(shard_manifest),
        ]
        if per_worker_limit is not None:
            cmd.extend(["--limit-samples", str(per_worker_limit)])
        procs.append((shard_index, subprocess.Popen(cmd, env=env), shard_manifest))

    failed: list[int] = []
    for shard_index, proc, _ in procs:
        return_code = proc.wait()
        if return_code != 0:
            failed.append(shard_index)
    if failed:
        raise RuntimeError(f"Worker shard(s) failed: {failed}")

    merge_worker_jsonl([item[2] for item in procs], manifest_path)

    kept_train = 0
    kept_val = 0
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "kept":
                continue
            if record.get("split") == "train":
                kept_train += 1
            elif record.get("split") == "val":
                kept_val += 1

    print(f"[done] train_samples={kept_train} val_samples={kept_val} manifest={manifest_path}")


def cmd_reconstruction_ortho(args: argparse.Namespace) -> None:
    validate_reconstruction_ortho_args(args)
    if not args.reconstruction_root.exists():
        raise FileNotFoundError(f"Reconstruction root not found: {args.reconstruction_root}")
    if not args.reconstruction_manifest.exists():
        raise FileNotFoundError(f"Reconstruction manifest not found: {args.reconstruction_manifest}")

    records = load_reconstruction_records(args.reconstruction_manifest, args.reconstruction_root)
    if not records:
        raise RuntimeError("No reconstruction TIFFs matched kept records in the manifest.")
    if int(args.num_workers) > 1 and args.reconstruction_shard_index is None:
        run_sharded_reconstruction_workers(args, records)
        return
    records = filter_reconstruction_shard(
        records,
        shard_index=args.reconstruction_shard_index,
        shard_count=args.reconstruction_shard_count,
    )
    if args.reconstruction_shard_index is not None:
        print(
            f"[info] shard={args.reconstruction_shard_index + 1}/{args.reconstruction_shard_count} "
            f"reconstructions={len(records)}"
        )
    if not records:
        print("[done] shard has no reconstructions to process")
        return

    train_dirs = prepare_output_root(
        args.train_out,
        l_res=args.lr_patch_size,
        r_res=args.hr_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=args.reconstruction_shard_index is not None,
    )
    val_dirs = prepare_output_root(
        args.val_out,
        l_res=args.lr_patch_size,
        r_res=args.hr_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=args.reconstruction_shard_index is not None,
    )
    split_dirs = {"train": train_dirs, "val": val_dirs}
    split_counts = {"train": 0, "val": 0}
    manifest_path = args.manifest_path or default_manifest_path(args.train_out, args.val_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")

    date_overrides = load_date_overrides(args.ortho_date_manifest)
    stats_cache: dict[str, tuple[tuple[int, int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = {}
    ortho_bounds_cache: dict[str, list[tuple[RasterInfo, tuple[float, float, float, float]]]] = {}
    kept_total = 0

    with manifest_path.open("a", encoding="utf-8") as manifest_fp:
        orthos = collect_ortho_infos(
            args.ortho_root,
            date_overrides,
            manifest_fp,
            exclude_globs=list(args.exclude_ortho_glob or []),
        )
        if not orthos:
            raise RuntimeError("No usable orthophotos found.")

        for record in tqdm(records, desc="Aligning reconstructions"):
            recon_path = Path(record["reconstruction_path"])
            source_sentinel_date = parse_iso_date(record.get("date"))
            with rasterio.open(recon_path) as recon_ds:
                crs_key = str(recon_ds.crs)
                if crs_key not in ortho_bounds_cache:
                    ortho_bounds_cache[crs_key] = build_ortho_bounds_index(orthos, recon_ds.crs)
                indexed_orthos = ortho_bounds_cache[crs_key]
                recon_rgb = np.moveaxis(recon_ds.read([1, 2, 3]), 0, -1).astype(np.uint8, copy=False)
                for crop_index, window in enumerate(
                    iter_reconstruction_crop_windows(
                        recon_ds.width,
                        recon_ds.height,
                        args.lr_patch_size,
                        args.reconstruction_crop_stride,
                    )
                ):
                    if args.limit_samples is not None and kept_total >= int(args.limit_samples):
                        print(f"[done] kept_samples={kept_total} manifest={manifest_path}")
                        return

                    crop_bounds = tuple(float(v) for v in window_bounds(window, recon_ds.transform))
                    candidate_orthos = find_intersecting_orthos_indexed(crop_bounds, indexed_orthos)
                    if not candidate_orthos:
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "no_ortho_overlap",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "crs": str(recon_ds.crs) if recon_ds.crs is not None else None,
                            },
                        )
                        continue
                    closest_ortho_day_diff = None
                    if args.max_ortho_day_diff is not None:
                        if source_sentinel_date is None:
                            write_manifest_line(
                                manifest_fp,
                                {
                                    "kind": "patch",
                                    "status": "skipped",
                                    "reason": "missing_source_sentinel_date",
                                    "split": "skip",
                                    "parent_reconstruction_id": record["sample_id"],
                                    "reconstruction_path": str(recon_path),
                                    "crop_index": crop_index,
                                    "lr_bounds": crop_bounds,
                                },
                            )
                            continue
                        candidate_orthos, closest_ortho_day_diff = filter_orthos_by_day_diff(
                            candidate_orthos,
                            source_sentinel_date,
                            max_day_diff=int(args.max_ortho_day_diff),
                        )
                        if not candidate_orthos:
                            write_manifest_line(
                                manifest_fp,
                                {
                                    "kind": "patch",
                                    "status": "skipped",
                                    "reason": "ortho_day_diff_exceeds_limit",
                                    "split": "skip",
                                    "parent_reconstruction_id": record["sample_id"],
                                    "reconstruction_path": str(recon_path),
                                    "source_sentinel_date": record.get("date"),
                                    "crop_index": crop_index,
                                    "lr_bounds": crop_bounds,
                                    "max_ortho_day_diff": int(args.max_ortho_day_diff),
                                    "closest_ortho_day_diff": closest_ortho_day_diff,
                                },
                            )
                            continue

                    lr_patch = recon_rgb[
                        int(window.row_off) : int(window.row_off + window.height),
                        int(window.col_off) : int(window.col_off + window.width),
                        :,
                    ]
                    lr_black_ratio = black_ratio(lr_patch, args.black_threshold)
                    if lr_black_ratio > float(args.max_black_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "lr_black_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "lr_black_ratio": lr_black_ratio,
                            },
                        )
                        continue
                    lr_mean_brightness, lr_bright_ratio = bright_patch_stats(
                        lr_patch,
                        int(args.lr_bright_threshold),
                    )
                    lr_white_ratio, lr_dark_ratio = white_dark_patch_stats(
                        lr_patch,
                        white_brightness_threshold=int(args.lr_white_brightness_threshold),
                        white_saturation_threshold=float(args.lr_white_saturation_threshold),
                        dark_threshold=int(args.lr_dark_threshold),
                    )
                    if lr_mean_brightness > float(args.max_lr_mean_brightness):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "lr_mean_brightness",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "lr_black_ratio": lr_black_ratio,
                                "lr_mean_brightness": lr_mean_brightness,
                                "lr_bright_ratio": lr_bright_ratio,
                            },
                        )
                        continue
                    if lr_mean_brightness < float(args.min_lr_mean_brightness):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "lr_min_mean_brightness",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "lr_mean_brightness": lr_mean_brightness,
                                "min_lr_mean_brightness": float(args.min_lr_mean_brightness),
                            },
                        )
                        continue
                    if lr_bright_ratio > float(args.max_lr_bright_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "lr_bright_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "lr_mean_brightness": lr_mean_brightness,
                                "lr_bright_ratio": lr_bright_ratio,
                                "lr_bright_threshold": int(args.lr_bright_threshold),
                            },
                        )
                        continue
                    if lr_white_ratio > float(args.max_lr_white_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "lr_white_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "lr_mean_brightness": lr_mean_brightness,
                                "lr_white_ratio": lr_white_ratio,
                                "lr_white_brightness_threshold": int(args.lr_white_brightness_threshold),
                                "lr_white_saturation_threshold": float(args.lr_white_saturation_threshold),
                            },
                        )
                        continue
                    if lr_dark_ratio > float(args.max_lr_dark_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "lr_dark_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "lr_mean_brightness": lr_mean_brightness,
                                "lr_dark_ratio": lr_dark_ratio,
                                "lr_dark_threshold": int(args.lr_dark_threshold),
                            },
                        )
                        continue

                    hr_transform = from_bounds(*crop_bounds, args.hr_patch_size, args.hr_patch_size)
                    rendered = render_best_ortho_patch(
                        candidate_orthos,
                        stats_cache,
                        dst_shape=(args.hr_patch_size, args.hr_patch_size),
                        dst_transform=hr_transform,
                        dst_crs=recon_ds.crs,
                        q_low=float(args.q_low),
                        q_high=float(args.q_high),
                        sample_max_dim=int(args.sample_max_dim),
                        stats_mode=str(args.ortho_stats_mode),
                    )
                    if rendered is None:
                        continue
                    hr_patch, valid_mask, ortho = rendered
                    ortho_day_diff = (
                        abs((source_sentinel_date - ortho.date).days)
                        if source_sentinel_date is not None
                        else None
                    )
                    valid_ratio = float((valid_mask > 0).mean())
                    hr_black_ratio = black_ratio(hr_patch, args.black_threshold)

                    if valid_ratio < float(args.min_valid_hr_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "low_hr_valid_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "ortho_date": serialize_date(ortho.date),
                                "abs_ortho_day_diff": ortho_day_diff,
                                "valid_hr_ratio": valid_ratio,
                            },
                        )
                        continue

                    if hr_black_ratio > float(args.max_black_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "hr_black_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "ortho_date": serialize_date(ortho.date),
                                "abs_ortho_day_diff": ortho_day_diff,
                                "hr_black_ratio": hr_black_ratio,
                            },
                        )
                        continue
                    hr_mean_brightness, hr_bright_ratio = bright_patch_stats(
                        hr_patch,
                        int(args.hr_bright_threshold),
                    )
                    hr_white_ratio, hr_dark_ratio = white_dark_patch_stats(
                        hr_patch,
                        white_brightness_threshold=int(args.hr_white_brightness_threshold),
                        white_saturation_threshold=float(args.hr_white_saturation_threshold),
                        dark_threshold=int(args.hr_dark_threshold),
                    )
                    if hr_mean_brightness > float(args.max_hr_mean_brightness):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "hr_mean_brightness",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "hr_black_ratio": hr_black_ratio,
                                "hr_mean_brightness": hr_mean_brightness,
                                "hr_bright_ratio": hr_bright_ratio,
                            },
                        )
                        continue
                    if hr_mean_brightness < float(args.min_hr_mean_brightness):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "hr_min_mean_brightness",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "hr_mean_brightness": hr_mean_brightness,
                                "min_hr_mean_brightness": float(args.min_hr_mean_brightness),
                            },
                        )
                        continue
                    if hr_bright_ratio > float(args.max_hr_bright_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "hr_bright_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "hr_mean_brightness": hr_mean_brightness,
                                "hr_bright_ratio": hr_bright_ratio,
                                "hr_bright_threshold": int(args.hr_bright_threshold),
                            },
                        )
                        continue
                    if hr_white_ratio > float(args.max_hr_white_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "hr_white_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "hr_mean_brightness": hr_mean_brightness,
                                "hr_white_ratio": hr_white_ratio,
                                "hr_white_brightness_threshold": int(args.hr_white_brightness_threshold),
                                "hr_white_saturation_threshold": float(args.hr_white_saturation_threshold),
                            },
                        )
                        continue
                    if hr_dark_ratio > float(args.max_hr_dark_ratio):
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "hr_dark_ratio",
                                "split": "skip",
                                "parent_reconstruction_id": record["sample_id"],
                                "reconstruction_path": str(recon_path),
                                "crop_index": crop_index,
                                "lr_bounds": crop_bounds,
                                "ortho_path": str(ortho.path),
                                "hr_mean_brightness": hr_mean_brightness,
                                "hr_dark_ratio": hr_dark_ratio,
                                "hr_dark_threshold": int(args.hr_dark_threshold),
                            },
                        )
                        continue

                    split_key = f"{record['sample_id']}:{crop_index}:{crop_bounds}"
                    split = stable_split(split_key, val_ratio=float(args.val_ratio), seed=int(args.seed))
                    sample_number = int(args.sample_id_offset) + split_counts[split] * int(args.sample_id_stride)
                    sample_id = f"{sample_number:06d}"
                    split_counts[split] += 1
                    file_name = f"{sample_id}.tif"
                    dirs = split_dirs[split]
                    lr_fp = dirs["lr"] / file_name
                    hr_fp = dirs["hr"] / file_name
                    sr_fp = dirs["sr"] / file_name
                    sr_patch = build_sr_patch(lr_patch, args.hr_patch_size)
                    save_rgb_tiff(lr_fp, lr_patch)
                    save_rgb_tiff(hr_fp, hr_patch)
                    save_rgb_tiff(sr_fp, sr_patch)
                    if args.save_debug_png:
                        png_name = f"{sample_id}.png"
                        save_rgb_png(dirs["debug_lr"] / png_name, lr_patch)
                        save_rgb_png(dirs["debug_hr"] / png_name, hr_patch)
                        save_rgb_png(dirs["debug_sr"] / png_name, sr_patch)

                    crop_row = int(window.row_off // args.lr_patch_size)
                    crop_col = int(window.col_off // args.lr_patch_size)
                    write_manifest_line(
                        manifest_fp,
                        {
                            "kind": "patch",
                            "status": "kept",
                            "reason": "kept",
                            "split": split,
                            "sample_id": sample_id,
                            "parent_reconstruction_id": record["sample_id"],
                            "reconstruction_path": str(recon_path),
                            "source_sentinel_root": record.get("source_root"),
                            "source_sentinel_date": record.get("date"),
                            "crop_index": crop_index,
                            "crop_row": crop_row,
                            "crop_col": crop_col,
                            "lr_bounds": crop_bounds,
                            "crs": str(recon_ds.crs) if recon_ds.crs is not None else None,
                            "ortho_path": str(ortho.path),
                            "ortho_date": serialize_date(ortho.date),
                            "abs_ortho_day_diff": ortho_day_diff,
                            "max_ortho_day_diff": int(args.max_ortho_day_diff)
                            if args.max_ortho_day_diff is not None
                            else None,
                            "ortho_bounds": transform_bounds_safe(ortho.bounds, ortho.crs, recon_ds.crs),
                            "valid_hr_ratio": valid_ratio,
                            "lr_black_ratio": lr_black_ratio,
                            "lr_mean_brightness": lr_mean_brightness,
                            "lr_bright_ratio": lr_bright_ratio,
                            "lr_white_ratio": lr_white_ratio,
                            "lr_dark_ratio": lr_dark_ratio,
                            "hr_black_ratio": hr_black_ratio,
                            "hr_mean_brightness": hr_mean_brightness,
                            "hr_bright_ratio": hr_bright_ratio,
                            "hr_white_ratio": hr_white_ratio,
                            "hr_dark_ratio": hr_dark_ratio,
                            "lr_path": str(lr_fp),
                            "hr_path": str(hr_fp),
                            "sr_path": str(sr_fp),
                        },
                    )
                    kept_total += 1

    print(
        f"[done] train_samples={split_counts['train']} val_samples={split_counts['val']} "
        f"manifest={manifest_path}"
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "reconstruction-ortho":
        cmd_reconstruction_ortho(parse_reconstruction_ortho_args(sys.argv[2:]))
        return

    args = parse_args()
    validate_args(args)

    if not args.planet_root.exists():
        raise FileNotFoundError(f"Planet root not found: {args.planet_root}")
    if not args.ortho_root.exists():
        raise FileNotFoundError(f"Orthophoto root not found: {args.ortho_root}")

    date_overrides = load_date_overrides(args.ortho_date_manifest)

    train_dirs = prepare_output_root(
        args.train_out,
        l_res=args.planet_patch_size,
        r_res=args.hr_patch_size,
        save_debug_png=bool(args.save_debug_png),
    )
    val_dirs = prepare_output_root(
        args.val_out,
        l_res=args.planet_patch_size,
        r_res=args.hr_patch_size,
        save_debug_png=bool(args.save_debug_png),
    )

    manifest_path = args.manifest_path or default_manifest_path(args.train_out, args.val_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("")

    print(f"[info] discovering Planet scenes under {args.planet_root}")
    planet_paths = collect_planet_scenes(args.planet_root)
    if not planet_paths:
        raise RuntimeError(f"No Planet scene TIFFs found under {args.planet_root}")

    print(f"[info] discovering Orthophotos under {args.ortho_root}")
    ortho_paths = collect_tiffs(args.ortho_root)
    if not ortho_paths:
        raise RuntimeError(f"No Orthophoto TIFFs found under {args.ortho_root}")

    planets: list[RasterInfo] = []
    orthos: list[RasterInfo] = []

    with manifest_path.open("a", encoding="utf-8") as manifest_fp:
        for path in planet_paths:
            described = describe_planet_scene(path)
            if isinstance(described, tuple):
                _, reason = described
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "scene",
                        "status": "skipped",
                        "reason": reason,
                        "split": "skip",
                        "planet_path": str(path),
                    },
                )
                continue
            planets.append(described)

        for path in ortho_paths:
            described = describe_ortho(path, date_overrides)
            if isinstance(described, tuple):
                _, reason = described
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "orthophoto",
                        "status": "skipped",
                        "reason": reason,
                        "split": "skip",
                        "ortho_path": str(path),
                    },
                )
                continue
            orthos.append(described)

        if not planets:
            raise RuntimeError("No usable Planet scenes were found after metadata checks.")
        if not orthos:
            raise RuntimeError("No usable Orthophotos were found after metadata checks.")

        selected_pairs: list[tuple[str, RasterInfo, RasterInfo, int]] = []
        for planet in planets:
            ortho, day_diff = choose_best_ortho(planet, orthos, max_day_diff=args.max_day_diff)
            if ortho is None:
                reason = "no_spatial_overlap" if day_diff is None else "date_diff_exceeds_limit"
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "scene",
                        "status": "skipped",
                        "reason": reason,
                        "split": "skip",
                        "planet_path": str(planet.path),
                        "planet_date": serialize_date(planet.date),
                        "max_day_diff": int(args.max_day_diff),
                        "candidate_day_diff": int(day_diff) if day_diff is not None else None,
                    },
                )
                continue
            pair_key = f"{planet.path}::{ortho.path}"
            selected_pairs.append((pair_key, planet, ortho, int(day_diff)))

        if not selected_pairs:
            diagnostics = build_no_pair_diagnostics(planets, orthos)
            raise RuntimeError(
                "No Planet scene could be paired with an Orthophoto under the current constraints.\n"
                f"{diagnostics}"
            )

        split_map = split_pairs_by_source(
            [pair_key for pair_key, _, _, _ in selected_pairs],
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )

        split_dirs = {"train": train_dirs, "val": val_dirs}
        split_counts = {"train": 0, "val": 0}
        stats_cache: dict[str, tuple[tuple[int, int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = {}
        step = resolve_patch_step(args.planet_patch_size, args.patch_overlap)

        for pair_key, planet, ortho, day_diff in selected_pairs:
            split = split_map[pair_key]
            dirs = split_dirs[split]

            print(
                f"[info] processing split={split} planet={planet.path.name} "
                f"ortho={ortho.path.name} day_diff={day_diff}"
            )

            with rasterio.open(planet.path) as planet_ds, rasterio.open(ortho.path) as ortho_ds:
                if str(planet.path) not in stats_cache:
                    planet_bands = resolve_rgb_band_indices(planet_ds)
                    planet_stats = compute_rgb_stats(
                        planet_ds,
                        planet_bands,
                        q_low=float(args.q_low),
                        q_high=float(args.q_high),
                        sample_max_dim=int(args.sample_max_dim),
                    )
                    stats_cache[str(planet.path)] = (planet_bands, planet_stats)
                planet_bands, planet_stats = stats_cache[str(planet.path)]

                if str(ortho.path) not in stats_cache:
                    ortho_bands = resolve_rgb_band_indices(ortho_ds)
                    ortho_stats = compute_rgb_stats(
                        ortho_ds,
                        ortho_bands,
                        q_low=float(args.q_low),
                        q_high=float(args.q_high),
                        sample_max_dim=int(args.sample_max_dim),
                    )
                    stats_cache[str(ortho.path)] = (ortho_bands, ortho_stats)
                ortho_bands, ortho_stats = stats_cache[str(ortho.path)]

                full_ortho_bounds_in_planet = transform_bounds_safe(ortho.bounds, ortho.crs, planet.crs)

                for window in iter_complete_windows(planet.width, planet.height, args.planet_patch_size, step):
                    lr_patch_raw = read_rgb_patch(planet_ds, planet_bands, window)
                    lr_patch_uint8 = normalize_rgb_array(lr_patch_raw, planet_stats)

                    planet_patch_black_ratio = black_ratio(lr_patch_uint8, args.black_threshold)
                    patch_bounds = tuple(float(v) for v in window_bounds(window, planet_ds.transform))

                    if planet_patch_black_ratio > args.max_black_ratio:
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "planet_black_ratio",
                                "split": "skip",
                                "planet_path": str(planet.path),
                                "ortho_path": str(ortho.path),
                                "planet_date": serialize_date(planet.date),
                                "ortho_date": serialize_date(ortho.date),
                                "abs_day_diff": day_diff,
                                "planet_bounds": patch_bounds,
                                "ortho_bounds": full_ortho_bounds_in_planet,
                                "planet_black_ratio": planet_patch_black_ratio,
                            },
                        )
                        continue

                    dst_transform = from_bounds(*patch_bounds, args.hr_patch_size, args.hr_patch_size)
                    hr_patch_uint8, valid_mask = render_ortho_patch(
                        ortho_ds,
                        ortho_bands,
                        ortho_stats,
                        dst_shape=(args.hr_patch_size, args.hr_patch_size),
                        dst_transform=dst_transform,
                        dst_crs=planet_ds.crs,
                    )
                    valid_ratio = float((valid_mask > 0).mean())
                    hr_patch_black_ratio = black_ratio(hr_patch_uint8, args.black_threshold)

                    if valid_ratio < args.min_valid_hr_ratio:
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "low_hr_valid_ratio",
                                "split": "skip",
                                "planet_path": str(planet.path),
                                "ortho_path": str(ortho.path),
                                "planet_date": serialize_date(planet.date),
                                "ortho_date": serialize_date(ortho.date),
                                "abs_day_diff": day_diff,
                                "planet_bounds": patch_bounds,
                                "ortho_bounds": full_ortho_bounds_in_planet,
                                "valid_hr_ratio": valid_ratio,
                            },
                        )
                        continue

                    if hr_patch_black_ratio > args.max_black_ratio:
                        write_manifest_line(
                            manifest_fp,
                            {
                                "kind": "patch",
                                "status": "skipped",
                                "reason": "orthophoto_black_ratio",
                                "split": "skip",
                                "planet_path": str(planet.path),
                                "ortho_path": str(ortho.path),
                                "planet_date": serialize_date(planet.date),
                                "ortho_date": serialize_date(ortho.date),
                                "abs_day_diff": day_diff,
                                "planet_bounds": patch_bounds,
                                "ortho_bounds": full_ortho_bounds_in_planet,
                                "hr_black_ratio": hr_patch_black_ratio,
                            },
                        )
                        continue

                    sample_id = f"{split_counts[split]:06d}"
                    split_counts[split] += 1
                    file_name = f"{sample_id}.tif"
                    lr_fp = dirs["lr"] / file_name
                    hr_fp = dirs["hr"] / file_name
                    sr_fp = dirs["sr"] / file_name
                    sr_patch_uint8 = build_sr_patch(lr_patch_uint8, args.hr_patch_size)

                    save_rgb_tiff(lr_fp, lr_patch_uint8)
                    save_rgb_tiff(hr_fp, hr_patch_uint8)
                    save_rgb_tiff(sr_fp, sr_patch_uint8)

                    if args.save_debug_png:
                        png_name = f"{sample_id}.png"
                        save_rgb_png(dirs["debug_lr"] / png_name, lr_patch_uint8)
                        save_rgb_png(dirs["debug_hr"] / png_name, hr_patch_uint8)
                        save_rgb_png(dirs["debug_sr"] / png_name, sr_patch_uint8)

                    write_manifest_line(
                        manifest_fp,
                        {
                            "kind": "patch",
                            "status": "kept",
                            "reason": "kept",
                            "split": split,
                            "sample_id": sample_id,
                            "planet_path": str(planet.path),
                            "ortho_path": str(ortho.path),
                            "planet_date": serialize_date(planet.date),
                            "ortho_date": serialize_date(ortho.date),
                            "abs_day_diff": day_diff,
                            "planet_bounds": patch_bounds,
                            "ortho_bounds": full_ortho_bounds_in_planet,
                            "planet_black_ratio": planet_patch_black_ratio,
                            "hr_black_ratio": hr_patch_black_ratio,
                            "valid_hr_ratio": valid_ratio,
                            "lr_path": str(lr_fp),
                            "hr_path": str(hr_fp),
                            "sr_path": str(sr_fp),
                        },
                    )

    print(
        f"[done] train_samples={split_counts['train']} val_samples={split_counts['val']} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
