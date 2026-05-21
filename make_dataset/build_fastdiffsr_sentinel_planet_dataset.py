#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageFilter
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.windows import bounds as window_bounds

from build_fastdiffsr_planet_ortho_dataset import (
    PLANET_DEFAULT_ROOT,
    RasterInfo,
    black_ratio,
    build_sr_patch,
    collect_planet_scenes,
    compute_rgb_stats,
    default_manifest_path,
    describe_planet_scene,
    iter_complete_windows,
    normalize_rgb_array,
    prepare_output_root,
    read_rgb_patch,
    resolve_patch_step,
    resolve_rgb_band_indices,
    save_rgb_png,
    save_rgb_tiff,
    serialize_date,
    split_pairs_by_source,
    transform_bounds_safe,
    write_manifest_line,
)


REQUIRED_SENTINEL_BANDS = ("B2", "B3", "B4", "SCL")
GRID_DIR_PATTERN = re.compile(r"^\d+_\d+$")
DATE_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STAGE_RANK = {
    "pm5": 0,
    "pm10": 1,
    "month": 2,
}
DEFAULT_SENTINEL_ROOT = Path(__file__).resolve().parent / "raw_sentinel"


def bright_patch_stats(img_uint8: np.ndarray, threshold: int) -> tuple[float, float]:
    brightness = img_uint8.astype(np.float32, copy=False).mean(axis=-1)
    return float(brightness.mean()), float((brightness >= threshold).mean())


def match_rgb_percentiles(
    src_uint8: np.ndarray,
    target_uint8: np.ndarray,
    *,
    src_mask: np.ndarray | None = None,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    matched = src_uint8.astype(np.float32, copy=True)
    target = target_uint8.astype(np.float32, copy=False)
    if src_mask is not None:
        valid = src_mask.astype(bool, copy=False)
    else:
        valid = np.ones(src_uint8.shape[:2], dtype=bool)

    if not np.any(valid):
        return src_uint8

    for channel in range(3):
        src_values = matched[..., channel][valid]
        target_values = target[..., channel].reshape(-1)
        src_low, src_high = np.percentile(src_values, [low, high])
        target_low, target_high = np.percentile(target_values, [low, high])
        src_span = float(src_high - src_low)
        if src_span <= 1e-6:
            matched[..., channel] = float(target_low)
            continue
        matched[..., channel] = ((matched[..., channel] - src_low) / src_span) * (
            target_high - target_low
        ) + target_low

    return np.clip(matched, 0, 255).astype(np.uint8)


def match_rgb_histograms(
    src_uint8: np.ndarray,
    target_uint8: np.ndarray,
    *,
    src_mask: np.ndarray | None = None,
) -> np.ndarray:
    matched = src_uint8.copy()
    if src_mask is not None:
        valid = src_mask.astype(bool, copy=False)
    else:
        valid = np.ones(src_uint8.shape[:2], dtype=bool)

    if not np.any(valid):
        return matched

    for channel in range(3):
        src_values = src_uint8[..., channel][valid]
        target_values = target_uint8[..., channel].reshape(-1)
        src_counts = np.bincount(src_values, minlength=256).astype(np.float64)
        target_counts = np.bincount(target_values, minlength=256).astype(np.float64)
        src_cdf = np.cumsum(src_counts)
        target_cdf = np.cumsum(target_counts)
        src_cdf /= src_cdf[-1]
        target_cdf /= target_cdf[-1]
        lookup = np.interp(src_cdf, target_cdf, np.arange(256))
        matched[..., channel] = np.clip(lookup[src_uint8[..., channel]], 0, 255).astype(np.uint8)

    return matched


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


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32, copy=False)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32, copy=False))
    return np.sqrt(gx * gx + gy * gy)


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32, copy=False)
    b = b.reshape(-1).astype(np.float32, copy=False)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def gaussian_blur_rgb_array(rgb_uint8: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma <= 0.0:
        return rgb_uint8
    image = Image.fromarray(rgb_uint8.astype(np.uint8, copy=False), mode="RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    return np.asarray(image, dtype=np.float32)


def edge_quality_score(
    sr_patch_uint8: np.ndarray,
    hr_patch_uint8: np.ndarray,
    hr_blur_sigma: float = 0.0,
) -> tuple[float, float, float]:
    if sr_patch_uint8.shape != hr_patch_uint8.shape:
        raise ValueError(f"shape mismatch: sr {sr_patch_uint8.shape} vs hr {hr_patch_uint8.shape}")

    sr_edge = gradient_magnitude(rgb_to_gray(sr_patch_uint8))
    hr_patch_for_edges = gaussian_blur_rgb_array(hr_patch_uint8, hr_blur_sigma)
    hr_edge = gradient_magnitude(rgb_to_gray(hr_patch_for_edges))
    return normalized_corr(sr_edge, hr_edge), float(sr_edge.mean()), float(hr_edge.mean())


def color_difference_score(
    sr_patch_uint8: np.ndarray,
    hr_patch_uint8: np.ndarray,
    *,
    blur_sigma: float = 8.0,
) -> tuple[float, float, float]:
    if sr_patch_uint8.shape != hr_patch_uint8.shape:
        raise ValueError(f"shape mismatch: sr {sr_patch_uint8.shape} vs hr {hr_patch_uint8.shape}")

    sr_patch = gaussian_blur_rgb_array(sr_patch_uint8, blur_sigma).astype(np.float32, copy=False)
    hr_patch = gaussian_blur_rgb_array(hr_patch_uint8, blur_sigma).astype(np.float32, copy=False)
    abs_diff = np.abs(sr_patch - hr_patch)
    return (
        float(abs_diff.mean()),
        float(np.percentile(abs_diff.mean(axis=-1), 95)),
        float(np.abs(sr_patch.mean(axis=(0, 1)) - hr_patch.mean(axis=(0, 1))).mean()),
    )


@dataclass(frozen=True)
class PlanetSceneRecord:
    scene_id: str
    source_side: str
    raster: RasterInfo

    @property
    def year_month(self) -> str:
        return self.raster.date.strftime("%Y-%m")


@dataclass(frozen=True)
class SentinelRequestRoot:
    request_id: str
    stage: str
    request_root: Path
    raw_dir: Path
    covered_scene_ids: tuple[str, ...]


@dataclass(frozen=True)
class SentinelCandidate:
    request_id: str
    stage: str
    request_root: Path
    raw_dir: Path
    sentinel_date: date
    files_by_band: dict[str, list[Path]]


@dataclass(frozen=True)
class PreparedSentinelCandidate:
    request_id: str
    stage: str
    request_root: Path
    raw_dir: Path
    sentinel_date: date
    day_diff: int
    crs: Any
    rgb: np.ndarray
    rgb_transform: Any
    rgb_valid_mask: np.ndarray
    rgb_stats: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    scl: np.ndarray
    scl_transform: Any
    scene_cloud_ratio: float


@dataclass
class CandidatePatchStats:
    kept: int = 0
    too_cloudy: int = 0
    empty_patch: int = 0
    low_valid: int = 0
    planet_black: int = 0
    planet_bright: int = 0
    sentinel_bright: int = 0
    planet_udm2_cloud: int = 0
    edge_mismatch: int = 0
    color_mismatch: int = 0


@dataclass
class ResumeState:
    processed_patch_keys: set[str]
    scene_candidate_keys: dict[str, set[str]]
    scene_patch_counts: dict[str, int]
    split_next_ids: dict[str, int]
    split_sample_totals: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FastDiffSR training data for Model 1 using Sentinel-2 RGB as LR "
            "(16x16) and Planet RGB as HR (64x64)."
        )
    )
    parser.add_argument("--planet-root", type=Path, default=PLANET_DEFAULT_ROOT)
    parser.add_argument(
        "--planet-years",
        type=int,
        nargs="+",
        default=None,
        help="Optional Planet acquisition year filter, e.g. --planet-years 2020 2021.",
    )
    parser.add_argument("--sentinel-root", type=Path, default=DEFAULT_SENTINEL_ROOT)
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("FastDiffSR/FastDiffSR/dataset/train_sentinel_planet_16_64"),
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path("FastDiffSR/FastDiffSR/dataset/val_sentinel_planet_16_64"),
    )
    parser.add_argument("--sentinel-patch-size", type=int, default=16)
    parser.add_argument("--planet-patch-size", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scl-cloud-ratio", type=float, default=0.10)
    parser.add_argument(
        "--max-planet-udm2-cloud-ratio",
        type=float,
        default=0.10,
        help=(
            "Maximum Planet UDM2 bad-pixel ratio per HR patch. Uses cloud, shadow, "
            "light haze, and heavy haze bands when a sibling *_udm2_clip.tif exists."
        ),
    )
    parser.add_argument(
        "--planet-udm2-cloud-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter Planet HR patches with sibling UDM2 cloud/shadow/haze masks.",
    )
    parser.add_argument("--black-threshold", type=int, default=50)
    parser.add_argument("--max-black-ratio", type=float, default=0.5)
    parser.add_argument(
        "--bright-threshold",
        type=int,
        default=245,
        help="Pixel brightness threshold used to reject washed-out Planet HR patches.",
    )
    parser.add_argument(
        "--max-bright-ratio",
        type=float,
        default=0.50,
        help=(
            "Maximum fraction of Planet HR pixels with mean RGB >= --bright-threshold. "
            "Set to 1.0 to disable this bright-pixel ratio check."
        ),
    )
    parser.add_argument(
        "--max-mean-brightness",
        type=float,
        default=245.0,
        help=(
            "Maximum mean RGB brightness for Planet HR patches after normalization. "
            "Set to 255 to disable this mean-brightness check."
        ),
    )
    parser.add_argument(
        "--min-mean-brightness",
        type=float,
        default=0.0,
        help=(
            "Minimum mean RGB brightness for Planet HR patches after normalization. "
            "Use this to reject very dark shadowy patches."
        ),
    )
    parser.add_argument(
        "--white-brightness-threshold",
        type=int,
        default=170,
        help="Brightness floor for the Planet HR white/haze pixel test.",
    )
    parser.add_argument(
        "--white-saturation-threshold",
        type=float,
        default=0.25,
        help="Saturation ceiling for the Planet HR white/haze pixel test.",
    )
    parser.add_argument(
        "--max-white-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum fraction of Planet HR pixels that are bright and low-saturation. "
            "Lower this to reject haze, shine, and thin cloud not caught by UDM2."
        ),
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=45,
        help="Pixel brightness threshold used by the Planet HR dark-pixel ratio check.",
    )
    parser.add_argument(
        "--max-dark-ratio",
        type=float,
        default=1.0,
        help="Maximum fraction of Planet HR pixels with mean RGB <= --dark-threshold.",
    )
    parser.add_argument(
        "--sentinel-bright-threshold",
        type=int,
        default=245,
        help="Pixel brightness threshold used to reject washed-out Sentinel-2 LR patches.",
    )
    parser.add_argument(
        "--max-sentinel-bright-ratio",
        type=float,
        default=0.50,
        help=(
            "Maximum fraction of Sentinel-2 LR pixels with mean RGB >= "
            "--sentinel-bright-threshold. Set to 1.0 to disable this check."
        ),
    )
    parser.add_argument(
        "--max-sentinel-mean-brightness",
        type=float,
        default=245.0,
        help=(
            "Maximum mean RGB brightness for Sentinel-2 LR patches after normalization. "
            "Set to 255 to disable this mean-brightness check."
        ),
    )
    parser.add_argument(
        "--min-sentinel-mean-brightness",
        type=float,
        default=0.0,
        help="Minimum mean RGB brightness for Sentinel-2 LR patches after normalization.",
    )
    parser.add_argument(
        "--sentinel-white-brightness-threshold",
        type=int,
        default=170,
        help="Brightness floor for the Sentinel-2 LR white/haze pixel test.",
    )
    parser.add_argument(
        "--sentinel-white-saturation-threshold",
        type=float,
        default=0.25,
        help="Saturation ceiling for the Sentinel-2 LR white/haze pixel test.",
    )
    parser.add_argument(
        "--max-sentinel-white-ratio",
        type=float,
        default=1.0,
        help="Maximum fraction of Sentinel-2 LR pixels that are bright and low-saturation.",
    )
    parser.add_argument(
        "--sentinel-dark-threshold",
        type=int,
        default=45,
        help="Pixel brightness threshold used by the Sentinel-2 LR dark-pixel ratio check.",
    )
    parser.add_argument(
        "--max-sentinel-dark-ratio",
        type=float,
        default=1.0,
        help="Maximum fraction of Sentinel-2 LR pixels with mean RGB <= --sentinel-dark-threshold.",
    )
    parser.add_argument("--min-valid-sentinel-ratio", type=float, default=0.95)
    parser.add_argument(
        "--color-match-sentinel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save Sentinel LR/SR patches after per-patch RGB color matching "
            "to the paired Planet HR patch."
        ),
    )
    parser.add_argument(
        "--color-match-method",
        choices=("histogram", "percentile"),
        default="percentile",
        help="Color matching method used when --color-match-sentinel is enabled.",
    )
    parser.add_argument(
        "--color-match-low",
        type=float,
        default=2.0,
        help="Lower percentile used by --color-match-method percentile.",
    )
    parser.add_argument(
        "--color-match-high",
        type=float,
        default=98.0,
        help="Upper percentile used by --color-match-method percentile.",
    )
    parser.add_argument(
        "--min-edge-corr",
        type=float,
        default=None,
        help=(
            "Optional minimum edge correlation between the upsampled Sentinel patch "
            "and Planet HR patch. Use values like 0.10 or 0.15 to reject structurally "
            "mismatched super-resolution pairs."
        ),
    )
    parser.add_argument(
        "--edge-blur-sigma",
        type=float,
        default=0.0,
        help=(
            "Optional Gaussian blur sigma applied to Planet HR patches before edge "
            "comparison. Try 1.0-2.0 to reduce Planet-only narrow paths/textures."
        ),
    )
    parser.add_argument(
        "--max-color-mad",
        type=float,
        default=None,
        help=(
            "Optional maximum mean absolute RGB difference between the color-matched "
            "upsampled Sentinel SR patch and Planet HR patch. The comparison is made "
            "after --color-diff-blur-sigma smoothing, so it targets broad color "
            "balance mismatches rather than fine texture."
        ),
    )
    parser.add_argument(
        "--max-color-p95-diff",
        type=float,
        default=None,
        help=(
            "Optional maximum 95th-percentile per-pixel mean RGB difference between "
            "the smoothed SR and HR patches. Useful for rejecting local color balance "
            "failures that the global mean can hide."
        ),
    )
    parser.add_argument(
        "--color-diff-blur-sigma",
        type=float,
        default=8.0,
        help=(
            "Gaussian blur sigma used by --max-color-mad and --max-color-p95-diff. "
            "Higher values ignore more fine detail and focus on broad color balance."
        ),
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--missing-path", type=Path, default=None)
    parser.add_argument(
        "--download-plan-tsv",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional planner TSV(s) produced by plan_planet_sentinel_downloads.py. "
            "When omitted, the builder will auto-discover sentinel_download_plan_*.tsv "
            "next to --sentinel-root."
        ),
    )
    parser.add_argument("--q-low", type=float, default=2.0)
    parser.add_argument("--q-high", type=float, default=98.0)
    parser.add_argument("--sample-max-dim", type=int, default=1024)
    parser.add_argument("--save-debug-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-pair-day-diff",
        type=int,
        default=5,
        help=(
            "Maximum absolute day difference allowed between a Planet scene and "
            "same-year Sentinel-2 candidate. Defaults to 5 days."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume safely from existing outputs by appending new samples, preserving "
            "existing sample IDs, and skipping already kept patches."
        ),
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Only process the first N Planet scenes after sorting. Useful for dry runs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes to use by sharding scenes. Requires --no-resume when > 1.",
    )
    parser.add_argument("--scene-shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scene-shard-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sample-id-offset", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--sample-id-stride", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--resume-state-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")
    if args.sentinel_patch_size <= 0 or args.planet_patch_size <= 0:
        raise ValueError("Patch sizes must be positive.")
    if args.planet_patch_size % args.sentinel_patch_size != 0:
        raise ValueError("--planet-patch-size must be divisible by --sentinel-patch-size.")
    expected_scale = args.planet_patch_size // args.sentinel_patch_size
    if int(args.scale) != int(expected_scale):
        raise ValueError(
            f"--scale={args.scale} does not match hr/lr ratio {args.planet_patch_size}/{args.sentinel_patch_size}={expected_scale}."
        )
    if not (0.0 <= args.patch_overlap < 1.0):
        raise ValueError("--patch-overlap must be in [0, 1).")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1).")
    if not (0.0 <= args.max_scl_cloud_ratio <= 1.0):
        raise ValueError("--max-scl-cloud-ratio must be in [0, 1].")
    if not (0.0 <= args.max_planet_udm2_cloud_ratio <= 1.0):
        raise ValueError("--max-planet-udm2-cloud-ratio must be in [0, 1].")
    if not (0.0 <= args.max_black_ratio <= 1.0):
        raise ValueError("--max-black-ratio must be in [0, 1].")
    if not (0 <= int(args.bright_threshold) <= 255):
        raise ValueError("--bright-threshold must be in [0, 255].")
    if not (0.0 <= args.max_bright_ratio <= 1.0):
        raise ValueError("--max-bright-ratio must be in [0, 1].")
    if not (0.0 <= float(args.max_mean_brightness) <= 255.0):
        raise ValueError("--max-mean-brightness must be in [0, 255].")
    if not (0.0 <= float(args.min_mean_brightness) <= 255.0):
        raise ValueError("--min-mean-brightness must be in [0, 255].")
    if float(args.min_mean_brightness) > float(args.max_mean_brightness):
        raise ValueError("--min-mean-brightness must be <= --max-mean-brightness.")
    if not (0 <= int(args.white_brightness_threshold) <= 255):
        raise ValueError("--white-brightness-threshold must be in [0, 255].")
    if not (0.0 <= float(args.white_saturation_threshold) <= 1.0):
        raise ValueError("--white-saturation-threshold must be in [0, 1].")
    if not (0.0 <= float(args.max_white_ratio) <= 1.0):
        raise ValueError("--max-white-ratio must be in [0, 1].")
    if not (0 <= int(args.dark_threshold) <= 255):
        raise ValueError("--dark-threshold must be in [0, 255].")
    if not (0.0 <= float(args.max_dark_ratio) <= 1.0):
        raise ValueError("--max-dark-ratio must be in [0, 1].")
    if not (0 <= int(args.sentinel_bright_threshold) <= 255):
        raise ValueError("--sentinel-bright-threshold must be in [0, 255].")
    if not (0.0 <= args.max_sentinel_bright_ratio <= 1.0):
        raise ValueError("--max-sentinel-bright-ratio must be in [0, 1].")
    if not (0.0 <= float(args.max_sentinel_mean_brightness) <= 255.0):
        raise ValueError("--max-sentinel-mean-brightness must be in [0, 255].")
    if not (0.0 <= float(args.min_sentinel_mean_brightness) <= 255.0):
        raise ValueError("--min-sentinel-mean-brightness must be in [0, 255].")
    if float(args.min_sentinel_mean_brightness) > float(args.max_sentinel_mean_brightness):
        raise ValueError("--min-sentinel-mean-brightness must be <= --max-sentinel-mean-brightness.")
    if not (0 <= int(args.sentinel_white_brightness_threshold) <= 255):
        raise ValueError("--sentinel-white-brightness-threshold must be in [0, 255].")
    if not (0.0 <= float(args.sentinel_white_saturation_threshold) <= 1.0):
        raise ValueError("--sentinel-white-saturation-threshold must be in [0, 1].")
    if not (0.0 <= float(args.max_sentinel_white_ratio) <= 1.0):
        raise ValueError("--max-sentinel-white-ratio must be in [0, 1].")
    if not (0 <= int(args.sentinel_dark_threshold) <= 255):
        raise ValueError("--sentinel-dark-threshold must be in [0, 255].")
    if not (0.0 <= float(args.max_sentinel_dark_ratio) <= 1.0):
        raise ValueError("--max-sentinel-dark-ratio must be in [0, 1].")
    if not (0.0 <= args.min_valid_sentinel_ratio <= 1.0):
        raise ValueError("--min-valid-sentinel-ratio must be in [0, 1].")
    if args.min_edge_corr is not None and not (-1.0 <= float(args.min_edge_corr) <= 1.0):
        raise ValueError("--min-edge-corr must be in [-1, 1] when provided.")
    if float(args.edge_blur_sigma) < 0.0:
        raise ValueError("--edge-blur-sigma must be non-negative.")
    if args.max_color_mad is not None and float(args.max_color_mad) < 0.0:
        raise ValueError("--max-color-mad must be non-negative when provided.")
    if args.max_color_p95_diff is not None and float(args.max_color_p95_diff) < 0.0:
        raise ValueError("--max-color-p95-diff must be non-negative when provided.")
    if float(args.color_diff_blur_sigma) < 0.0:
        raise ValueError("--color-diff-blur-sigma must be non-negative.")
    if not (0.0 <= args.q_low < args.q_high <= 100.0):
        raise ValueError("--q-low and --q-high must satisfy 0 <= q_low < q_high <= 100.")
    if int(args.sample_max_dim) < 32:
        raise ValueError("--sample-max-dim must be at least 32.")
    if int(args.max_pair_day_diff) < 0:
        raise ValueError("--max-pair-day-diff must be non-negative.")
    if not 0.0 <= float(args.color_match_low) < float(args.color_match_high) <= 100.0:
        raise ValueError("--color-match-low must be less than --color-match-high within [0, 100].")
    if args.limit_scenes is not None and int(args.limit_scenes) <= 0:
        raise ValueError("--limit-scenes must be positive when provided.")
    if args.planet_years is not None and not args.planet_years:
        raise ValueError("--planet-years must include at least one year when provided.")
    if int(args.num_workers) <= 0:
        raise ValueError("--num-workers must be positive.")
    if int(args.sample_id_stride) <= 0:
        raise ValueError("--sample-id-stride must be positive.")
    if int(args.sample_id_offset) < 0:
        raise ValueError("--sample-id-offset must be non-negative.")
    if args.scene_shard_index is None and args.scene_shard_count is not None:
        raise ValueError("--scene-shard-count requires --scene-shard-index.")
    if args.scene_shard_index is not None:
        if args.scene_shard_count is None:
            raise ValueError("--scene-shard-index requires --scene-shard-count.")
        if not (0 <= int(args.scene_shard_index) < int(args.scene_shard_count)):
            raise ValueError("--scene-shard-index must satisfy 0 <= index < count.")
    if int(args.num_workers) > 1 and args.resume:
        raise ValueError("--num-workers > 1 requires --no-resume.")


def detect_source_side(planet_root: Path, scene_path: Path) -> str:
    try:
        rel_parts = [part.lower() for part in scene_path.relative_to(planet_root).parts]
    except ValueError:
        rel_parts = []
    full_parts = [part.lower() for part in scene_path.parts]
    for parts in (rel_parts, full_parts):
        if "lower" in parts:
            return "lower"
        if "upper" in parts:
            return "upper"
    if rel_parts:
        return rel_parts[0]
    return "unknown"


def build_scene_id(source_side: str, scene_stem: str) -> str:
    prefix = source_side.lower().strip().replace(" ", "_") or "unknown"
    return f"{prefix}__{scene_stem}"


def collect_planet_scene_records(
    planet_root: Path,
    *,
    planet_years: set[int] | None = None,
) -> list[PlanetSceneRecord]:
    scene_paths = collect_planet_scenes(planet_root)
    records: list[PlanetSceneRecord] = []
    skipped = 0
    skipped_year = 0
    for scene_path in scene_paths:
        described = describe_planet_scene(scene_path)
        if isinstance(described, tuple):
            skipped += 1
            continue
        if planet_years is not None and described.date.year not in planet_years:
            skipped_year += 1
            continue
        source_side = detect_source_side(planet_root, scene_path)
        records.append(
            PlanetSceneRecord(
                scene_id=build_scene_id(source_side, described.stem),
                source_side=source_side,
                raster=described,
            )
        )
    records.sort(key=lambda item: (item.raster.date, item.scene_id, str(item.raster.path)))
    year_filter_label = (
        ",".join(str(year) for year in sorted(planet_years)) if planet_years is not None else "all"
    )
    print(
        f"[info] discovered usable_planet_scenes={len(records)} skipped={skipped} "
        f"skipped_year={skipped_year} planet_years={year_filter_label}"
    )
    return records


def filter_scene_shard(
    scenes: list[PlanetSceneRecord],
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> list[PlanetSceneRecord]:
    if shard_index is None or shard_count is None or shard_count <= 1:
        return scenes
    return [scene for idx, scene in enumerate(scenes) if idx % shard_count == shard_index]


def iter_download_plan_paths(
    sentinel_root: Path,
    explicit_paths: list[Path] | None,
) -> list[Path]:
    if explicit_paths:
        resolved = [path.resolve() for path in explicit_paths]
        missing = [str(path) for path in resolved if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Download plan TSV not found: " + ", ".join(sorted(missing))
            )
        return resolved
    return sorted(sentinel_root.resolve().parent.glob("sentinel_download_plan_*.tsv"))


def parse_scene_ids_from_plan_row(row: dict[str, str]) -> tuple[str, ...]:
    covered_scene_ids = (row.get("covered_scene_ids") or "").strip()
    if covered_scene_ids:
        return tuple(item.strip() for item in covered_scene_ids.split(",") if item.strip())
    scene_id = (row.get("scene_id") or "").strip()
    if scene_id:
        return (scene_id,)
    return tuple()


def collect_request_roots(
    sentinel_root: Path,
    download_plan_tsvs: list[Path] | None = None,
) -> dict[str, list[SentinelRequestRoot]]:
    request_index: dict[str, list[SentinelRequestRoot]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    metadata_request_roots: set[Path] = set()
    plan_paths = iter_download_plan_paths(sentinel_root, download_plan_tsvs)

    for plan_path in plan_paths:
        with plan_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            if reader.fieldnames is None:
                continue
            for row in reader:
                raw_data_root = (row.get("data_root") or "").strip()
                if not raw_data_root:
                    continue
                data_root = Path(raw_data_root)
                if not data_root.is_absolute():
                    data_root = (plan_path.parent / data_root).resolve()
                else:
                    data_root = data_root.resolve()
                raw_dirs = sorted(path for path in data_root.glob("AR_*_raw") if path.is_dir())
                if not raw_dirs:
                    continue
                scene_ids = parse_scene_ids_from_plan_row(row)
                if not scene_ids:
                    continue
                metadata_request_roots.add(data_root)
                request_id = (row.get("request_id") or data_root.name).strip() or data_root.name
                stage = (row.get("stage") or data_root.parent.name).strip() or "unknown"
                for raw_dir in raw_dirs:
                    raw_dir = raw_dir.resolve()
                    for scene_id in scene_ids:
                        pair_key = (scene_id, str(raw_dir))
                        if pair_key in seen_pairs:
                            continue
                        request_index.setdefault(scene_id, []).append(
                            SentinelRequestRoot(
                                request_id=request_id,
                                stage=stage,
                                request_root=data_root,
                                raw_dir=raw_dir,
                                covered_scene_ids=scene_ids,
                            )
                        )
                        seen_pairs.add(pair_key)

    for raw_dir in sorted(sentinel_root.rglob("AR_*_raw")):
        if not raw_dir.is_dir():
            continue
        raw_dir = raw_dir.resolve()
        request_root = raw_dir.parent.resolve()
        if request_root in metadata_request_roots:
            continue
        scene_id = request_root.name
        stage = request_root.parent.name if request_root.parent != request_root else "unknown"
        pair_key = (scene_id, str(raw_dir))
        if pair_key in seen_pairs:
            continue
        request_index.setdefault(scene_id, []).append(
            SentinelRequestRoot(
                request_id=scene_id,
                stage=stage,
                request_root=request_root,
                raw_dir=raw_dir,
                covered_scene_ids=(scene_id,),
            )
        )
        seen_pairs.add(pair_key)

    for scene_id, request_roots in request_index.items():
        request_index[scene_id] = sorted(
            request_roots,
            key=lambda item: (
                stage_rank(item.stage),
                str(item.request_root),
                str(item.raw_dir),
            ),
        )

    unique_request_roots = {
        str(request_root.request_root)
        for request_roots in request_index.values()
        for request_root in request_roots
    }
    print(
        f"[info] indexed sentinel_request_roots={len(unique_request_roots)} "
        f"scene_links={sum(len(v) for v in request_index.values())} "
        f"plan_tsvs={len(plan_paths)}"
    )
    return request_index


def parse_date_dir(name: str) -> date | None:
    if not DATE_DIR_PATTERN.fullmatch(name):
        return None
    try:
        return datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        return None


def collect_candidates_for_request_root(request_root: SentinelRequestRoot) -> list[SentinelCandidate]:
    candidates: list[SentinelCandidate] = []
    per_date: dict[str, dict[str, list[Path]]] = {}
    for grid_dir in sorted(request_root.raw_dir.iterdir()):
        if not grid_dir.is_dir() or not GRID_DIR_PATTERN.fullmatch(grid_dir.name):
            continue
        for date_dir in sorted(grid_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            parsed_date = parse_date_dir(date_dir.name)
            if parsed_date is None:
                continue
            files_by_band = per_date.setdefault(
                date_dir.name,
                {band: [] for band in REQUIRED_SENTINEL_BANDS},
            )
            for band in REQUIRED_SENTINEL_BANDS:
                fp = date_dir / f"{band}_{date_dir.name}.tif"
                if fp.is_file():
                    files_by_band[band].append(fp)
    for date_str, files_by_band in sorted(per_date.items()):
        parsed_date = parse_date_dir(date_str)
        if parsed_date is None:
            continue
        candidates.append(
            SentinelCandidate(
                request_id=request_root.request_id,
                stage=request_root.stage,
                request_root=request_root.request_root,
                raw_dir=request_root.raw_dir,
                sentinel_date=parsed_date,
                files_by_band={band: sorted(paths) for band, paths in files_by_band.items()},
            )
        )
    return candidates


def collect_scene_candidates(
    request_roots: list[SentinelRequestRoot],
    candidate_cache: dict[tuple[str, str], list[SentinelCandidate]],
) -> list[SentinelCandidate]:
    candidates: list[SentinelCandidate] = []
    seen_requests: set[tuple[str, str, str]] = set()
    for request_root in request_roots:
        cache_key = (str(request_root.request_root), str(request_root.raw_dir))
        if cache_key not in candidate_cache:
            candidate_cache[cache_key] = collect_candidates_for_request_root(request_root)
        for candidate in candidate_cache[cache_key]:
            dedupe_key = (
                candidate.request_id,
                candidate.sentinel_date.isoformat(),
                str(candidate.raw_dir),
            )
            if dedupe_key in seen_requests:
                continue
            candidates.append(candidate)
            seen_requests.add(dedupe_key)
    return candidates


def stage_rank(stage: str) -> int:
    return STAGE_RANK.get(stage, 99)


def candidate_sort_key(scene: PlanetSceneRecord, candidate: SentinelCandidate) -> tuple[int, int, str, str]:
    return (
        abs((scene.raster.date - candidate.sentinel_date).days),
        stage_rank(candidate.stage),
        candidate.sentinel_date.isoformat(),
        str(candidate.request_root),
    )


def filter_candidates_by_day_diff(
    scene: PlanetSceneRecord,
    candidates: list[SentinelCandidate],
    *,
    max_day_diff: int,
) -> list[SentinelCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.sentinel_date.year == scene.raster.date.year
        and abs((scene.raster.date - candidate.sentinel_date).days) <= max_day_diff
    ]


def merge_single_band(
    paths: list[Path],
    *,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, Any, Any] | None:
    if not paths:
        return None
    datasets = []
    try:
        for path in sorted(paths):
            datasets.append(rasterio.open(path))
        if not datasets:
            return None
        merged, transform = merge(datasets, bounds=bounds, nodata=0)
        return merged[0], transform, datasets[0].crs
    finally:
        for dataset in datasets:
            try:
                dataset.close()
            except Exception:
                pass


def compute_rgb_stats_from_array(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    q_low: float,
    q_high: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    stats: list[tuple[float, float]] = []
    valid_mask = valid_mask.astype(bool)
    for channel in range(3):
        band = rgb[:, :, channel]
        finite = band[np.isfinite(band) & valid_mask]
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


def reproject_mask(
    src_mask: np.ndarray,
    *,
    src_transform: Any,
    src_crs: Any,
    dst_shape: tuple[int, int],
    dst_transform: Any,
    dst_crs: Any,
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=src_mask.astype(np.uint8) * 255,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst > 0


def render_rgb_patch(
    candidate: PreparedSentinelCandidate,
    *,
    dst_shape: tuple[int, int],
    dst_transform: Any,
    dst_crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    dst = np.zeros((3, dst_shape[0], dst_shape[1]), dtype=np.float32)
    for idx in range(3):
        reproject(
            source=candidate.rgb[:, :, idx],
            destination=dst[idx],
            src_transform=candidate.rgb_transform,
            src_crs=candidate.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
    valid_mask = reproject_mask(
        candidate.rgb_valid_mask,
        src_transform=candidate.rgb_transform,
        src_crs=candidate.crs,
        dst_shape=dst_shape,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
    )
    rgb = np.moveaxis(dst, 0, -1)
    return normalize_rgb_array(rgb, candidate.rgb_stats), valid_mask


def render_scl_patch(
    candidate: PreparedSentinelCandidate,
    *,
    dst_shape: tuple[int, int],
    dst_transform: Any,
    dst_crs: Any,
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=candidate.scl.astype(np.uint8),
        destination=dst,
        src_transform=candidate.scl_transform,
        src_crs=candidate.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst


def render_scl_to_match_rgb_grid(
    scl: np.ndarray,
    *,
    scl_transform: Any,
    scl_crs: Any,
    dst_shape: tuple[int, int],
    dst_transform: Any,
    dst_crs: Any,
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=scl.astype(np.uint8),
        destination=dst,
        src_transform=scl_transform,
        src_crs=scl_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst


def compute_cloud_ratio(scl_patch: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = valid_mask.astype(bool)
    if not valid.any():
        return 1.0
    cloudy = (scl_patch > 7) & valid
    return float(cloudy.sum()) / float(valid.sum())


def find_planet_udm2_path(planet_path: Path) -> Path | None:
    stem = planet_path.stem
    prefix = stem.split("_3B_", 1)[0]
    candidates = [
        planet_path.with_name(f"{prefix}_3B_udm2_clip{planet_path.suffix}"),
        planet_path.with_name(f"{prefix}_3B_UDM2_clip{planet_path.suffix}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(planet_path.parent.glob(f"{prefix}*udm2*{planet_path.suffix}"))
    if matches:
        return matches[0]
    matches = sorted(planet_path.parent.glob(f"{prefix}*UDM2*{planet_path.suffix}"))
    if matches:
        return matches[0]
    return None


def resolve_udm2_bad_band_indices(udm2_ds: rasterio.DatasetReader) -> tuple[int, ...]:
    desired = {"cloud", "shadow", "haze_light", "haze_heavy"}
    descriptions = [desc.lower() if isinstance(desc, str) else "" for desc in udm2_ds.descriptions]
    indices = tuple(idx for idx, desc in enumerate(descriptions, 1) if desc in desired)
    if indices:
        return indices

    # Planet UDM2 convention: 3=shadow, 4=light haze, 5=heavy haze, 6=cloud.
    fallback = tuple(idx for idx in (3, 4, 5, 6) if idx <= udm2_ds.count)
    return fallback


def render_udm2_bad_mask(
    udm2_ds: rasterio.DatasetReader,
    bad_band_indices: tuple[int, ...],
    *,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    if not bad_band_indices:
        return np.zeros(dst_shape, dtype=bool)

    bad = np.zeros(dst_shape, dtype=bool)
    for band_index in bad_band_indices:
        dst = np.zeros(dst_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(udm2_ds, band_index),
            destination=dst,
            src_transform=udm2_ds.transform,
            src_crs=udm2_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        bad |= dst > 0
    return bad


def prepare_candidate(
    scene: PlanetSceneRecord,
    candidate: SentinelCandidate,
    *,
    q_low: float,
    q_high: float,
) -> tuple[PreparedSentinelCandidate | None, str]:
    if any(not candidate.files_by_band.get(band) for band in REQUIRED_SENTINEL_BANDS):
        return None, "missing_band"

    first_band_files = candidate.files_by_band["B2"]
    with rasterio.open(first_band_files[0]) as src:
        candidate_crs = src.crs
    scene_bounds_in_candidate = transform_bounds_safe(scene.raster.bounds, scene.raster.crs, candidate_crs)
    if scene_bounds_in_candidate is None:
        return None, "no_overlap"

    red_merged = merge_single_band(candidate.files_by_band["B4"], bounds=scene_bounds_in_candidate)
    green_merged = merge_single_band(candidate.files_by_band["B3"], bounds=scene_bounds_in_candidate)
    blue_merged = merge_single_band(candidate.files_by_band["B2"], bounds=scene_bounds_in_candidate)
    scl_merged = merge_single_band(candidate.files_by_band["SCL"], bounds=scene_bounds_in_candidate)
    if red_merged is None or green_merged is None or blue_merged is None or scl_merged is None:
        return None, "missing_band"

    red, rgb_transform, rgb_crs = red_merged
    green, _, _ = green_merged
    blue, _, _ = blue_merged
    scl, scl_transform, _ = scl_merged
    rgb = np.dstack([red, green, blue]).astype(np.float32)
    rgb_valid_mask = (red != 0) | (green != 0) | (blue != 0)
    if not rgb_valid_mask.any():
        return None, "no_overlap"

    scl_on_rgb_grid = render_scl_to_match_rgb_grid(
        scl.astype(np.uint8),
        scl_transform=scl_transform,
        scl_crs=rgb_crs,
        dst_shape=rgb_valid_mask.shape,
        dst_transform=rgb_transform,
        dst_crs=rgb_crs,
    )
    scl_valid_mask = scl_on_rgb_grid != 0
    valid_mask = rgb_valid_mask | scl_valid_mask
    if not valid_mask.any():
        return None, "no_overlap"

    rgb_stats = compute_rgb_stats_from_array(
        rgb,
        rgb_valid_mask,
        q_low=float(q_low),
        q_high=float(q_high),
    )
    scene_cloud_ratio = compute_cloud_ratio(scl_on_rgb_grid, valid_mask)
    return (
        PreparedSentinelCandidate(
            request_id=candidate.request_id,
            stage=candidate.stage,
            request_root=candidate.request_root,
            raw_dir=candidate.raw_dir,
            sentinel_date=candidate.sentinel_date,
            day_diff=abs((scene.raster.date - candidate.sentinel_date).days),
            crs=rgb_crs,
            rgb=rgb,
            rgb_transform=rgb_transform,
            rgb_valid_mask=rgb_valid_mask,
            rgb_stats=rgb_stats,
            scl=scl.astype(np.uint8),
            scl_transform=scl_transform,
            scene_cloud_ratio=scene_cloud_ratio,
        ),
        "ready",
    )


def default_missing_path(train_out: Path, val_out: Path) -> Path:
    return default_manifest_path(train_out, val_out).parent / "missing_pairs.tsv"


def default_resume_state_path(train_out: Path, val_out: Path) -> Path:
    return default_manifest_path(train_out, val_out).parent / "resume_kept_patches.tsv"


def worker_artifact_dir(train_out: Path, val_out: Path) -> Path:
    return default_manifest_path(train_out, val_out).parent / "_worker_runs"


def merge_worker_jsonl(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fp:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as in_fp:
                shutil.copyfileobj(in_fp, out_fp)


def merge_worker_tsv(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    with output_path.open("w", encoding="utf-8", newline="") as out_fp:
        for path in paths:
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open("r", encoding="utf-8", newline="") as in_fp:
                lines = in_fp.readlines()
            if not lines:
                continue
            if not header_written:
                out_fp.write(lines[0])
                header_written = True
            for line in lines[1:]:
                out_fp.write(line)
    if not header_written:
        output_path.write_text("", encoding="utf-8")


def serialize_patch_bounds(bounds: tuple[float, float, float, float]) -> str:
    return ",".join(f"{float(value):.8f}" for value in bounds)


def build_candidate_key(
    *,
    request_id: str,
    request_root: Path,
    sentinel_date: date,
) -> str:
    return "	".join(
        [
            request_id,
            str(request_root),
            sentinel_date.isoformat(),
        ]
    )


def candidate_key_for_candidate(candidate: SentinelCandidate) -> str:
    return build_candidate_key(
        request_id=candidate.request_id,
        request_root=candidate.request_root,
        sentinel_date=candidate.sentinel_date,
    )


def candidate_key_for_prepared(candidate: PreparedSentinelCandidate) -> str:
    return build_candidate_key(
        request_id=candidate.request_id,
        request_root=candidate.request_root,
        sentinel_date=candidate.sentinel_date,
    )


def build_patch_key(
    *,
    scene_id: str,
    candidate_key: str,
    patch_bounds: tuple[float, float, float, float],
) -> str:
    return "	".join([scene_id, candidate_key, serialize_patch_bounds(patch_bounds)])


def count_split_tiffs(split_dirs: dict[str, Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("lr", "hr", "sr"):
        counts[key] = sum(1 for _ in split_dirs[key].glob("*.tif"))
    return counts


def split_dirs_have_outputs(split_dirs: dict[str, Path]) -> bool:
    return any(count > 0 for count in count_split_tiffs(split_dirs).values())


def load_resume_state(state_path: Path) -> ResumeState:
    processed_patch_keys: set[str] = set()
    scene_candidate_keys: dict[str, set[str]] = {}
    scene_patch_counts: dict[str, int] = {}
    split_next_ids = {"train": 0, "val": 0}
    split_sample_totals = {"train": 0, "val": 0}

    if not state_path.exists() or state_path.stat().st_size == 0:
        return ResumeState(
            processed_patch_keys=processed_patch_keys,
            scene_candidate_keys=scene_candidate_keys,
            scene_patch_counts=scene_patch_counts,
            split_next_ids=split_next_ids,
            split_sample_totals=split_sample_totals,
        )

    with state_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="	")
        if reader.fieldnames is None:
            return ResumeState(
                processed_patch_keys=processed_patch_keys,
                scene_candidate_keys=scene_candidate_keys,
                scene_patch_counts=scene_patch_counts,
                split_next_ids=split_next_ids,
                split_sample_totals=split_sample_totals,
            )
        for row in reader:
            scene_id = (row.get("scene_id") or "").strip()
            request_id = (row.get("sentinel_request_id") or "").strip()
            request_root_raw = (row.get("sentinel_request_root") or "").strip()
            sentinel_date_raw = (row.get("sentinel_date") or "").strip()
            patch_bounds_raw = (row.get("planet_bounds") or "").strip()
            split = (row.get("split") or "").strip()
            sample_id_raw = (row.get("sample_id") or "").strip()
            if (
                not scene_id
                or not request_id
                or not request_root_raw
                or not sentinel_date_raw
                or not patch_bounds_raw
            ):
                continue
            sentinel_day = datetime.strptime(sentinel_date_raw, "%Y-%m-%d").date()
            candidate_key = build_candidate_key(
                request_id=request_id,
                request_root=Path(request_root_raw),
                sentinel_date=sentinel_day,
            )
            patch_key = "	".join([scene_id, candidate_key, patch_bounds_raw])
            processed_patch_keys.add(patch_key)
            scene_candidate_keys.setdefault(scene_id, set()).add(candidate_key)
            scene_patch_counts[scene_id] = scene_patch_counts.get(scene_id, 0) + 1
            if split in split_sample_totals:
                split_sample_totals[split] += 1
            if split in split_next_ids and sample_id_raw.isdigit():
                split_next_ids[split] = max(split_next_ids[split], int(sample_id_raw) + 1)

    return ResumeState(
        processed_patch_keys=processed_patch_keys,
        scene_candidate_keys=scene_candidate_keys,
        scene_patch_counts=scene_patch_counts,
        split_next_ids=split_next_ids,
        split_sample_totals=split_sample_totals,
    )


def validate_resume_inputs(
    *,
    resume_state: ResumeState,
    state_path: Path,
    split_dirs: dict[str, dict[str, Path]],
    manifest_path: Path,
) -> None:
    outputs_present = any(split_dirs_have_outputs(dirs) for dirs in split_dirs.values())
    has_state = state_path.exists() and state_path.stat().st_size > 0
    if outputs_present and not has_state:
        raise RuntimeError(
            "Existing dataset outputs were found, but no resume state file exists. "
            "Use a fresh output directory or remove the old partial outputs before "
            "rerunning with --resume."
        )

    if outputs_present and not manifest_path.exists():
        raise RuntimeError(
            "Existing dataset outputs were found, but the manifest file is missing. "
            "Use a fresh output directory or restore the manifest before resuming."
        )

    for split, dirs in split_dirs.items():
        expected = resume_state.split_sample_totals.get(split, 0)
        counts = count_split_tiffs(dirs)
        for kind, count in counts.items():
            if count != expected:
                raise RuntimeError(
                    f"Resume check failed for split={split} kind={kind}: "
                    f"state has {expected} samples but directory contains {count} TIFFs. "
                    "Use a fresh output directory or reconcile the partial outputs first."
                )


def register_kept_patch(
    *,
    resume_state: ResumeState,
    state_writer: csv.DictWriter,
    split: str,
    sample_id: str,
    scene_id: str,
    prepared: PreparedSentinelCandidate,
    patch_bounds: tuple[float, float, float, float],
) -> None:
    candidate_key = candidate_key_for_prepared(prepared)
    patch_bounds_str = serialize_patch_bounds(patch_bounds)
    patch_key = "	".join([scene_id, candidate_key, patch_bounds_str])
    resume_state.processed_patch_keys.add(patch_key)
    resume_state.scene_candidate_keys.setdefault(scene_id, set()).add(candidate_key)
    resume_state.scene_patch_counts[scene_id] = resume_state.scene_patch_counts.get(scene_id, 0) + 1
    resume_state.split_sample_totals[split] = resume_state.split_sample_totals.get(split, 0) + 1
    state_writer.writerow(
        {
            "split": split,
            "sample_id": sample_id,
            "scene_id": scene_id,
            "sentinel_request_id": prepared.request_id,
            "sentinel_request_root": str(prepared.request_root),
            "sentinel_date": serialize_date(prepared.sentinel_date),
            "planet_bounds": patch_bounds_str,
        }
    )


def allocate_sample_id(
    *,
    split_counts: dict[str, int],
    split: str,
    stride: int,
) -> str:
    sample_id = f"{split_counts[split]:06d}"
    split_counts[split] += stride
    return sample_id


def process_candidate_patches(
    *,
    scene: PlanetSceneRecord,
    prepared: PreparedSentinelCandidate,
    planet_ds: rasterio.DatasetReader,
    planet_bands: tuple[int, int, int],
    planet_stats: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    planet_udm2_ds: rasterio.DatasetReader | None,
    planet_udm2_bad_bands: tuple[int, ...],
    split: str,
    dirs: dict[str, Path],
    split_counts: dict[str, int],
    args: argparse.Namespace,
    manifest_fp,
    resume_state: ResumeState,
    state_writer: csv.DictWriter,
) -> CandidatePatchStats:
    stats = CandidatePatchStats()
    step = resolve_patch_step(args.planet_patch_size, args.patch_overlap)
    candidate_key = candidate_key_for_prepared(prepared)

    for window in iter_complete_windows(scene.raster.width, scene.raster.height, args.planet_patch_size, step):
        patch_bounds = tuple(float(v) for v in window_bounds(window, planet_ds.transform))
        patch_key = build_patch_key(
            scene_id=scene.scene_id,
            candidate_key=candidate_key,
            patch_bounds=patch_bounds,
        )
        if patch_key in resume_state.processed_patch_keys:
            continue

        hr_patch_raw = read_rgb_patch(planet_ds, planet_bands, window)
        hr_patch_uint8 = normalize_rgb_array(hr_patch_raw, planet_stats)
        hr_black_ratio = black_ratio(hr_patch_uint8, args.black_threshold)
        hr_mean_brightness, hr_bright_ratio = bright_patch_stats(
            hr_patch_uint8,
            int(args.bright_threshold),
        )
        hr_white_ratio, hr_dark_ratio = white_dark_patch_stats(
            hr_patch_uint8,
            white_brightness_threshold=int(args.white_brightness_threshold),
            white_saturation_threshold=float(args.white_saturation_threshold),
            dark_threshold=int(args.dark_threshold),
        )

        if hr_black_ratio > args.max_black_ratio:
            stats.planet_black += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "planet_black_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_black_ratio": hr_black_ratio,
                },
            )
            continue

        if hr_mean_brightness > float(args.max_mean_brightness):
            stats.planet_bright += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "planet_mean_brightness",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_black_ratio": hr_black_ratio,
                    "planet_mean_brightness": hr_mean_brightness,
                    "planet_bright_ratio": hr_bright_ratio,
                    "bright_threshold": int(args.bright_threshold),
                },
            )
            continue

        if hr_mean_brightness < float(args.min_mean_brightness):
            stats.planet_black += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "planet_min_mean_brightness",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_black_ratio": hr_black_ratio,
                    "planet_mean_brightness": hr_mean_brightness,
                    "min_mean_brightness": float(args.min_mean_brightness),
                },
            )
            continue

        if hr_bright_ratio > float(args.max_bright_ratio):
            stats.planet_bright += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "planet_bright_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_black_ratio": hr_black_ratio,
                    "planet_mean_brightness": hr_mean_brightness,
                    "planet_bright_ratio": hr_bright_ratio,
                    "bright_threshold": int(args.bright_threshold),
                },
            )
            continue

        if hr_white_ratio > float(args.max_white_ratio):
            stats.planet_bright += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "planet_white_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_mean_brightness": hr_mean_brightness,
                    "planet_white_ratio": hr_white_ratio,
                    "white_brightness_threshold": int(args.white_brightness_threshold),
                    "white_saturation_threshold": float(args.white_saturation_threshold),
                },
            )
            continue

        if hr_dark_ratio > float(args.max_dark_ratio):
            stats.planet_black += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "planet_dark_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_mean_brightness": hr_mean_brightness,
                    "planet_dark_ratio": hr_dark_ratio,
                    "dark_threshold": int(args.dark_threshold),
                },
            )
            continue

        planet_udm2_cloud_ratio = None
        if args.planet_udm2_cloud_check and planet_udm2_ds is not None and planet_udm2_bad_bands:
            hr_transform = from_bounds(*patch_bounds, args.planet_patch_size, args.planet_patch_size)
            planet_bad_mask = render_udm2_bad_mask(
                planet_udm2_ds,
                planet_udm2_bad_bands,
                dst_shape=(args.planet_patch_size, args.planet_patch_size),
                dst_transform=hr_transform,
                dst_crs=planet_ds.crs,
            )
            planet_udm2_cloud_ratio = float(planet_bad_mask.mean())
            if planet_udm2_cloud_ratio > args.max_planet_udm2_cloud_ratio:
                stats.planet_udm2_cloud += 1
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "patch",
                        "status": "skipped",
                        "reason": "planet_udm2_cloud_ratio",
                        "split": "skip",
                        "scene_id": scene.scene_id,
                        "planet_path": str(scene.raster.path),
                        "planet_udm2_path": str(planet_udm2_ds.name),
                        "planet_date": serialize_date(scene.raster.date),
                        "sentinel_stage": prepared.stage,
                        "sentinel_date": serialize_date(prepared.sentinel_date),
                        "abs_day_diff": int(prepared.day_diff),
                        "planet_bounds": patch_bounds,
                        "planet_black_ratio": hr_black_ratio,
                        "planet_udm2_cloud_ratio": planet_udm2_cloud_ratio,
                    },
                )
                continue

        lr_transform = from_bounds(*patch_bounds, args.sentinel_patch_size, args.sentinel_patch_size)
        lr_patch_uint8, valid_mask = render_rgb_patch(
            prepared,
            dst_shape=(args.sentinel_patch_size, args.sentinel_patch_size),
            dst_transform=lr_transform,
            dst_crs=planet_ds.crs,
        )
        valid_ratio = float(valid_mask.mean())
        if valid_ratio < args.min_valid_sentinel_ratio:
            stats.low_valid += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "low_valid_sentinel_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "valid_sentinel_ratio": valid_ratio,
                },
            )
            continue

        scl_patch = render_scl_patch(
            prepared,
            dst_shape=(args.sentinel_patch_size, args.sentinel_patch_size),
            dst_transform=lr_transform,
            dst_crs=planet_ds.crs,
        )
        cloud_ratio = compute_cloud_ratio(scl_patch, valid_mask)
        if cloud_ratio > args.max_scl_cloud_ratio:
            stats.too_cloudy += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "too_cloudy",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "cloud_ratio": cloud_ratio,
                },
            )
            continue

        lr_black_ratio = black_ratio(lr_patch_uint8, args.black_threshold)
        if lr_black_ratio > args.max_black_ratio:
            stats.empty_patch += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "sentinel_black_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "sentinel_black_ratio": lr_black_ratio,
                },
            )
            continue

        lr_mean_brightness, lr_bright_ratio = bright_patch_stats(
            lr_patch_uint8,
            int(args.sentinel_bright_threshold),
        )
        lr_white_ratio, lr_dark_ratio = white_dark_patch_stats(
            lr_patch_uint8,
            white_brightness_threshold=int(args.sentinel_white_brightness_threshold),
            white_saturation_threshold=float(args.sentinel_white_saturation_threshold),
            dark_threshold=int(args.sentinel_dark_threshold),
        )
        if lr_mean_brightness > float(args.max_sentinel_mean_brightness):
            stats.sentinel_bright += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "sentinel_mean_brightness",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_mean_brightness": hr_mean_brightness,
                    "planet_bright_ratio": hr_bright_ratio,
                    "sentinel_black_ratio": lr_black_ratio,
                    "sentinel_mean_brightness": lr_mean_brightness,
                    "sentinel_bright_ratio": lr_bright_ratio,
                    "sentinel_bright_threshold": int(args.sentinel_bright_threshold),
                },
            )
            continue

        if lr_mean_brightness < float(args.min_sentinel_mean_brightness):
            stats.empty_patch += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "sentinel_min_mean_brightness",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "sentinel_mean_brightness": lr_mean_brightness,
                    "min_sentinel_mean_brightness": float(args.min_sentinel_mean_brightness),
                },
            )
            continue

        if lr_bright_ratio > float(args.max_sentinel_bright_ratio):
            stats.sentinel_bright += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "sentinel_bright_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "planet_mean_brightness": hr_mean_brightness,
                    "planet_bright_ratio": hr_bright_ratio,
                    "sentinel_black_ratio": lr_black_ratio,
                    "sentinel_mean_brightness": lr_mean_brightness,
                    "sentinel_bright_ratio": lr_bright_ratio,
                    "sentinel_bright_threshold": int(args.sentinel_bright_threshold),
                },
            )
            continue

        if lr_white_ratio > float(args.max_sentinel_white_ratio):
            stats.sentinel_bright += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "sentinel_white_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "sentinel_mean_brightness": lr_mean_brightness,
                    "sentinel_white_ratio": lr_white_ratio,
                    "sentinel_white_brightness_threshold": int(args.sentinel_white_brightness_threshold),
                    "sentinel_white_saturation_threshold": float(args.sentinel_white_saturation_threshold),
                },
            )
            continue

        if lr_dark_ratio > float(args.max_sentinel_dark_ratio):
            stats.empty_patch += 1
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "patch",
                    "status": "skipped",
                    "reason": "sentinel_dark_ratio",
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "sentinel_stage": prepared.stage,
                    "sentinel_date": serialize_date(prepared.sentinel_date),
                    "abs_day_diff": int(prepared.day_diff),
                    "planet_bounds": patch_bounds,
                    "sentinel_mean_brightness": lr_mean_brightness,
                    "sentinel_dark_ratio": lr_dark_ratio,
                    "sentinel_dark_threshold": int(args.sentinel_dark_threshold),
                },
            )
            continue

        lr_patch_to_save = lr_patch_uint8
        if args.color_match_sentinel:
            if args.color_match_method == "histogram":
                lr_patch_to_save = match_rgb_histograms(
                    lr_patch_uint8,
                    hr_patch_uint8,
                    src_mask=valid_mask,
                )
            else:
                lr_patch_to_save = match_rgb_percentiles(
                    lr_patch_uint8,
                    hr_patch_uint8,
                    src_mask=valid_mask,
                    low=float(args.color_match_low),
                    high=float(args.color_match_high),
                )

        sr_patch_uint8 = build_sr_patch(lr_patch_to_save, args.planet_patch_size)
        color_mad = None
        color_p95_diff = None
        color_channel_mean_diff = None
        if args.max_color_mad is not None or args.max_color_p95_diff is not None:
            color_mad, color_p95_diff, color_channel_mean_diff = color_difference_score(
                sr_patch_uint8,
                hr_patch_uint8,
                blur_sigma=float(args.color_diff_blur_sigma),
            )
            if (
                args.max_color_mad is not None
                and color_mad > float(args.max_color_mad)
            ) or (
                args.max_color_p95_diff is not None
                and color_p95_diff > float(args.max_color_p95_diff)
            ):
                stats.color_mismatch += 1
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "patch",
                        "status": "skipped",
                        "reason": "color_mismatch",
                        "split": "skip",
                        "scene_id": scene.scene_id,
                        "planet_path": str(scene.raster.path),
                        "planet_date": serialize_date(scene.raster.date),
                        "sentinel_stage": prepared.stage,
                        "sentinel_date": serialize_date(prepared.sentinel_date),
                        "abs_day_diff": int(prepared.day_diff),
                        "planet_bounds": patch_bounds,
                        "color_mad": color_mad,
                        "max_color_mad": float(args.max_color_mad)
                        if args.max_color_mad is not None
                        else None,
                        "color_p95_diff": color_p95_diff,
                        "max_color_p95_diff": float(args.max_color_p95_diff)
                        if args.max_color_p95_diff is not None
                        else None,
                        "color_channel_mean_diff": color_channel_mean_diff,
                        "color_diff_blur_sigma": float(args.color_diff_blur_sigma),
                    },
                )
                continue

        edge_corr = None
        sr_edge_mean = None
        hr_edge_mean = None
        if args.min_edge_corr is not None:
            edge_corr, sr_edge_mean, hr_edge_mean = edge_quality_score(
                sr_patch_uint8,
                hr_patch_uint8,
                hr_blur_sigma=float(args.edge_blur_sigma),
            )
            if edge_corr < float(args.min_edge_corr):
                stats.edge_mismatch += 1
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "patch",
                        "status": "skipped",
                        "reason": "edge_corr",
                        "split": "skip",
                        "scene_id": scene.scene_id,
                        "planet_path": str(scene.raster.path),
                        "planet_date": serialize_date(scene.raster.date),
                        "sentinel_stage": prepared.stage,
                        "sentinel_date": serialize_date(prepared.sentinel_date),
                        "abs_day_diff": int(prepared.day_diff),
                        "planet_bounds": patch_bounds,
                        "edge_corr": edge_corr,
                        "min_edge_corr": float(args.min_edge_corr),
                        "edge_blur_sigma": float(args.edge_blur_sigma),
                        "sr_edge_mean": sr_edge_mean,
                        "hr_edge_mean": hr_edge_mean,
                    },
                )
                continue

        sample_id = allocate_sample_id(
            split_counts=split_counts,
            split=split,
            stride=int(args.sample_id_stride),
        )
        file_name = f"{sample_id}.tif"

        lr_fp = dirs["lr"] / file_name
        hr_fp = dirs["hr"] / file_name
        sr_fp = dirs["sr"] / file_name
        save_rgb_tiff(lr_fp, lr_patch_to_save)
        save_rgb_tiff(hr_fp, hr_patch_uint8)
        save_rgb_tiff(sr_fp, sr_patch_uint8)

        if args.save_debug_png:
            png_name = f"{sample_id}.png"
            save_rgb_png(dirs["debug_lr"] / png_name, lr_patch_to_save)
            save_rgb_png(dirs["debug_hr"] / png_name, hr_patch_uint8)
            save_rgb_png(dirs["debug_sr"] / png_name, sr_patch_uint8)

        stats.kept += 1
        register_kept_patch(
            resume_state=resume_state,
            state_writer=state_writer,
            split=split,
            sample_id=sample_id,
            scene_id=scene.scene_id,
            prepared=prepared,
            patch_bounds=patch_bounds,
        )
        write_manifest_line(
            manifest_fp,
            {
                "kind": "patch",
                "status": "kept",
                "reason": "kept",
                "split": split,
                "sample_id": sample_id,
                "scene_id": scene.scene_id,
                "source_side": scene.source_side,
                "planet_path": str(scene.raster.path),
                "planet_date": serialize_date(scene.raster.date),
                "sentinel_stage": prepared.stage,
                "sentinel_request_id": prepared.request_id,
                "sentinel_request_root": str(prepared.request_root),
                "sentinel_date": serialize_date(prepared.sentinel_date),
                "abs_day_diff": int(prepared.day_diff),
                "scene_cloud_ratio": float(prepared.scene_cloud_ratio),
                "planet_bounds": patch_bounds,
                "cloud_ratio": cloud_ratio,
                "valid_sentinel_ratio": valid_ratio,
                "planet_black_ratio": hr_black_ratio,
                "planet_mean_brightness": hr_mean_brightness,
                "planet_bright_ratio": hr_bright_ratio,
                "planet_white_ratio": hr_white_ratio,
                "planet_dark_ratio": hr_dark_ratio,
                "planet_udm2_cloud_ratio": planet_udm2_cloud_ratio,
                "sentinel_black_ratio": lr_black_ratio,
                "sentinel_mean_brightness": lr_mean_brightness,
                "sentinel_bright_ratio": lr_bright_ratio,
                "sentinel_white_ratio": lr_white_ratio,
                "sentinel_dark_ratio": lr_dark_ratio,
                "color_match_sentinel": bool(args.color_match_sentinel),
                "color_match_method": args.color_match_method if args.color_match_sentinel else None,
                "color_match_low": float(args.color_match_low)
                if args.color_match_sentinel and args.color_match_method == "percentile"
                else None,
                "color_match_high": float(args.color_match_high)
                if args.color_match_sentinel and args.color_match_method == "percentile"
                else None,
                "color_mad": color_mad,
                "color_p95_diff": color_p95_diff,
                "color_channel_mean_diff": color_channel_mean_diff,
                "color_diff_blur_sigma": float(args.color_diff_blur_sigma)
                if args.max_color_mad is not None or args.max_color_p95_diff is not None
                else None,
                "edge_corr": edge_corr,
                "edge_blur_sigma": float(args.edge_blur_sigma) if args.min_edge_corr is not None else None,
                "sr_edge_mean": sr_edge_mean,
                "hr_edge_mean": hr_edge_mean,
                "lr_path": str(lr_fp),
                "hr_path": str(hr_fp),
                "sr_path": str(sr_fp),
            },
        )

    return stats


def classify_missing_reason(
    *,
    has_any_candidate: bool,
    has_complete_candidate: bool,
    has_overlap_candidate: bool,
    too_bright_only: bool,
    too_cloudy_only: bool,
    edge_mismatch_only: bool,
    color_mismatch_only: bool,
) -> str:
    if not has_any_candidate:
        return "no_sentinel"
    if not has_complete_candidate:
        return "missing_band"
    if not has_overlap_candidate:
        return "no_overlap"
    if too_bright_only:
        return "too_bright"
    if too_cloudy_only:
        return "too_cloudy"
    if edge_mismatch_only:
        return "edge_mismatch"
    if color_mismatch_only:
        return "color_mismatch"
    return "empty_patch"


def run_sharded_workers(args: argparse.Namespace, scenes: list[PlanetSceneRecord]) -> None:
    worker_count = min(int(args.num_workers), len(scenes))
    manifest_path = (args.manifest_path or default_manifest_path(args.train_out, args.val_out)).resolve()
    missing_path = (args.missing_path or default_missing_path(args.train_out, args.val_out)).resolve()
    resume_state_path = default_resume_state_path(args.train_out, args.val_out).resolve()
    shard_dir = worker_artifact_dir(args.train_out, args.val_out).resolve()

    train_dirs = prepare_output_root(
        args.train_out,
        l_res=args.sentinel_patch_size,
        r_res=args.planet_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=False,
    )
    val_dirs = prepare_output_root(
        args.val_out,
        l_res=args.sentinel_patch_size,
        r_res=args.planet_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=False,
    )
    del train_dirs, val_dirs

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resume_state_path.parent.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    base_argv = sys.argv[1:]
    procs: list[tuple[int, subprocess.Popen[bytes], Path, Path, Path]] = []
    env = os.environ.copy()
    for shard_index in range(worker_count):
        shard_manifest = shard_dir / f"pairs_manifest.worker{shard_index:02d}.jsonl"
        shard_missing = shard_dir / f"missing_pairs.worker{shard_index:02d}.tsv"
        shard_resume = shard_dir / f"resume_kept_patches.worker{shard_index:02d}.tsv"
        cmd = [
            sys.executable,
            str(script_path),
            *base_argv,
            "--num-workers",
            "1",
            "--scene-shard-index",
            str(shard_index),
            "--scene-shard-count",
            str(worker_count),
            "--sample-id-offset",
            str(shard_index),
            "--sample-id-stride",
            str(worker_count),
            "--manifest-path",
            str(shard_manifest),
            "--missing-path",
            str(shard_missing),
            "--resume-state-path",
            str(shard_resume),
            "--no-resume",
        ]
        procs.append(
            (
                shard_index,
                subprocess.Popen(cmd, env=env),
                shard_manifest,
                shard_missing,
                shard_resume,
            )
        )

    failed: list[int] = []
    for shard_index, proc, _, _, _ in procs:
        return_code = proc.wait()
        if return_code != 0:
            failed.append(shard_index)
    if failed:
        raise RuntimeError(f"Worker shard(s) failed: {failed}")

    merge_worker_jsonl([item[2] for item in procs], manifest_path)
    merge_worker_tsv([item[3] for item in procs], missing_path)
    merge_worker_tsv([item[4] for item in procs], resume_state_path)

    merged_state = load_resume_state(resume_state_path)
    print(
        f"[done] train_samples={merged_state.split_sample_totals['train']} "
        f"val_samples={merged_state.split_sample_totals['val']} "
        f"manifest={manifest_path} missing_tsv={missing_path}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.planet_root.exists():
        raise FileNotFoundError(f"Planet root not found: {args.planet_root}")
    if not args.sentinel_root.exists():
        raise FileNotFoundError(f"Sentinel root not found: {args.sentinel_root}")

    planet_years = set(int(year) for year in args.planet_years) if args.planet_years else None
    scenes = collect_planet_scene_records(
        args.planet_root.resolve(),
        planet_years=planet_years,
    )
    if args.limit_scenes is not None:
        scenes = scenes[: int(args.limit_scenes)]
        print(f"[info] limited scenes to first {len(scenes)} records")
    if not scenes:
        raise RuntimeError("No usable Planet scenes were found after metadata checks.")
    if int(args.num_workers) > 1 and args.scene_shard_index is None:
        run_sharded_workers(args, scenes)
        return

    scenes = filter_scene_shard(
        scenes,
        shard_index=args.scene_shard_index,
        shard_count=args.scene_shard_count,
    )
    if args.scene_shard_index is not None:
        print(
            f"[info] shard={args.scene_shard_index + 1}/{args.scene_shard_count} "
            f"scenes={len(scenes)}"
        )
    if not scenes:
        print("[done] shard has no scenes to process")
        return

    train_dirs = prepare_output_root(
        args.train_out,
        l_res=args.sentinel_patch_size,
        r_res=args.planet_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=bool(args.resume or args.scene_shard_index is not None),
    )
    val_dirs = prepare_output_root(
        args.val_out,
        l_res=args.sentinel_patch_size,
        r_res=args.planet_patch_size,
        save_debug_png=bool(args.save_debug_png),
        allow_existing=bool(args.resume or args.scene_shard_index is not None),
    )
    split_dirs = {"train": train_dirs, "val": val_dirs}

    manifest_path = (args.manifest_path or default_manifest_path(args.train_out, args.val_out)).resolve()
    missing_path = (args.missing_path or default_missing_path(args.train_out, args.val_out)).resolve()
    resume_state_path = (
        args.resume_state_path.resolve()
        if hasattr(args, "resume_state_path") and args.resume_state_path is not None
        else default_resume_state_path(args.train_out, args.val_out).resolve()
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resume_state_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_state = load_resume_state(resume_state_path)
        validate_resume_inputs(
            resume_state=resume_state,
            state_path=resume_state_path,
            split_dirs=split_dirs,
            manifest_path=manifest_path,
        )
        split_counts = dict(resume_state.split_next_ids)
        manifest_path.touch(exist_ok=True)
    else:
        resume_state = ResumeState(
            processed_patch_keys=set(),
            scene_candidate_keys={},
            scene_patch_counts={},
            split_next_ids={"train": 0, "val": 0},
            split_sample_totals={"train": 0, "val": 0},
        )
        split_counts = {
            "train": int(args.sample_id_offset),
            "val": int(args.sample_id_offset),
        }
        manifest_path.write_text("", encoding="utf-8")
        resume_state_path.write_text("", encoding="utf-8")

    request_roots_by_scene = collect_request_roots(
        args.sentinel_root.resolve(),
        download_plan_tsvs=[path.resolve() for path in args.download_plan_tsv]
        if args.download_plan_tsv
        else None,
    )
    candidate_cache: dict[tuple[str, str], list[SentinelCandidate]] = {}
    split_map = split_pairs_by_source(
        [scene.scene_id for scene in scenes],
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    missing_rows: list[dict[str, str | int]] = []
    state_fieldnames = [
        "split",
        "sample_id",
        "scene_id",
        "sentinel_request_id",
        "sentinel_request_root",
        "sentinel_date",
        "planet_bounds",
    ]

    with manifest_path.open("a", encoding="utf-8") as manifest_fp, resume_state_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as state_fp:
        state_writer = csv.DictWriter(state_fp, fieldnames=state_fieldnames, delimiter="	")
        if state_fp.tell() == 0:
            state_writer.writeheader()
        for scene in scenes:
            scene_candidates = collect_scene_candidates(
                request_roots_by_scene.get(scene.scene_id, []),
                candidate_cache,
            )
            scene_candidates.sort(key=lambda item: candidate_sort_key(scene, item))
            raw_candidate_count = len(scene_candidates)
            scene_candidates = filter_candidates_by_day_diff(
                scene,
                scene_candidates,
                max_day_diff=int(args.max_pair_day_diff),
            )
            existing_candidate_keys = resume_state.scene_candidate_keys.get(scene.scene_id, set())
            existing_patch_count = resume_state.scene_patch_counts.get(scene.scene_id, 0)
            if existing_candidate_keys:
                matched_candidates = [
                    candidate
                    for candidate in scene_candidates
                    if candidate_key_for_candidate(candidate) in existing_candidate_keys
                ]
                if matched_candidates:
                    scene_candidates = matched_candidates
                else:
                    print(
                        f"[info] scene={scene.scene_id} resume_skip=existing_samples_no_candidate "
                        f"existing_patches={existing_patch_count}"
                    )
                    continue
            available_stages = sorted({candidate.stage for candidate in scene_candidates}, key=stage_rank)
            print(
                f"[info] scene={scene.scene_id} date={scene.raster.date.isoformat()} "
                f"candidates={len(scene_candidates)}/{raw_candidate_count} "
                f"max_pair_day_diff={int(args.max_pair_day_diff)} "
                f"existing_patches={existing_patch_count}"
            )

            has_any_candidate = bool(scene_candidates)
            has_complete_candidate = False
            has_overlap_candidate = False
            any_empty_signal = False
            any_too_bright_signal = False
            any_too_cloudy_signal = False
            any_edge_mismatch_signal = False
            any_color_mismatch_signal = False
            kept_scene = False

            try:
                with rasterio.open(scene.raster.path) as planet_ds:
                    planet_bands = resolve_rgb_band_indices(planet_ds)
                    planet_stats = compute_rgb_stats(
                        planet_ds,
                        planet_bands,
                        q_low=float(args.q_low),
                        q_high=float(args.q_high),
                        sample_max_dim=int(args.sample_max_dim),
                    )
                    planet_udm2_ds = None
                    planet_udm2_bad_bands: tuple[int, ...] = ()
                    planet_udm2_path = None
                    if args.planet_udm2_cloud_check:
                        planet_udm2_path = find_planet_udm2_path(scene.raster.path)
                        if planet_udm2_path is not None:
                            try:
                                planet_udm2_ds = rasterio.open(planet_udm2_path)
                                planet_udm2_bad_bands = resolve_udm2_bad_band_indices(
                                    planet_udm2_ds
                                )
                            except rasterio.errors.RasterioError as exc:
                                print(
                                    f"[warn] scene={scene.scene_id} reason=planet_udm2_read_error "
                                    f"path={planet_udm2_path} error={exc}"
                                )
                                planet_udm2_ds = None
                                planet_udm2_bad_bands = ()

                    try:
                        for candidate in scene_candidates:
                            prepared, candidate_status = prepare_candidate(
                                scene,
                                candidate,
                                q_low=float(args.q_low),
                                q_high=float(args.q_high),
                            )
                            write_manifest_line(
                                manifest_fp,
                                {
                                    "kind": "candidate",
                                    "status": "ready" if prepared is not None else "skipped",
                                    "reason": candidate_status,
                                    "scene_id": scene.scene_id,
                                    "source_side": scene.source_side,
                                    "planet_path": str(scene.raster.path),
                                    "planet_udm2_path": str(planet_udm2_path)
                                    if planet_udm2_path is not None
                                    else None,
                                    "planet_date": serialize_date(scene.raster.date),
                                    "sentinel_stage": candidate.stage,
                                    "sentinel_request_id": candidate.request_id,
                                    "sentinel_request_root": str(candidate.request_root),
                                    "sentinel_date": serialize_date(candidate.sentinel_date),
                                    "abs_day_diff": abs((scene.raster.date - candidate.sentinel_date).days),
                                },
                            )
                            if candidate_status == "missing_band":
                                continue
                            has_complete_candidate = True
                            if prepared is None:
                                continue
                            has_overlap_candidate = True

                            split = split_map[scene.scene_id]
                            patch_stats = process_candidate_patches(
                                scene=scene,
                                prepared=prepared,
                                planet_ds=planet_ds,
                                planet_bands=planet_bands,
                                planet_stats=planet_stats,
                                planet_udm2_ds=planet_udm2_ds,
                                planet_udm2_bad_bands=planet_udm2_bad_bands,
                                split=split,
                                dirs=split_dirs[split],
                                split_counts=split_counts,
                                args=args,
                                manifest_fp=manifest_fp,
                                resume_state=resume_state,
                                state_writer=state_writer,
                            )
                            if patch_stats.kept > 0:
                                kept_scene = True
                                write_manifest_line(
                                    manifest_fp,
                                    {
                                        "kind": "pair",
                                        "status": "kept",
                                        "reason": "kept",
                                        "split": split,
                                        "scene_id": scene.scene_id,
                                        "source_side": scene.source_side,
                                        "planet_path": str(scene.raster.path),
                                        "planet_udm2_path": str(planet_udm2_path)
                                        if planet_udm2_path is not None
                                        else None,
                                        "planet_date": serialize_date(scene.raster.date),
                                        "sentinel_stage": prepared.stage,
                                        "sentinel_request_id": prepared.request_id,
                                        "sentinel_request_root": str(prepared.request_root),
                                        "sentinel_date": serialize_date(prepared.sentinel_date),
                                        "abs_day_diff": int(prepared.day_diff),
                                        "scene_cloud_ratio": float(prepared.scene_cloud_ratio),
                                        "kept_patch_count": int(patch_stats.kept),
                                    },
                                )
                                break
                            if existing_candidate_keys:
                                kept_scene = True
                                break

                            if patch_stats.too_cloudy > 0 or patch_stats.planet_udm2_cloud > 0:
                                any_too_cloudy_signal = True
                            if patch_stats.planet_bright > 0 or patch_stats.sentinel_bright > 0:
                                any_too_bright_signal = True
                            if patch_stats.edge_mismatch > 0:
                                any_edge_mismatch_signal = True
                            if patch_stats.color_mismatch > 0:
                                any_color_mismatch_signal = True
                            if (
                                patch_stats.empty_patch > 0
                                or patch_stats.low_valid > 0
                                or patch_stats.planet_black > 0
                            ):
                                any_empty_signal = True
                    finally:
                        if planet_udm2_ds is not None:
                            planet_udm2_ds.close()
            except rasterio.errors.RasterioError as exc:
                reason = "planet_read_error"
                nearest_candidate = scene_candidates[0] if scene_candidates else None
                print(
                    f"[warn] scene={scene.scene_id} reason={reason} "
                    f"path={scene.raster.path} error={exc}"
                )
                missing_rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "source_side": scene.source_side,
                        "planet_date": serialize_date(scene.raster.date),
                        "year_month": scene.year_month,
                        "planet_path": str(scene.raster.path),
                        "reason": reason,
                        "available_stage_count": len(available_stages),
                        "available_stages": ",".join(available_stages),
                        "available_candidate_count": len(scene_candidates),
                        "best_candidate_date": serialize_date(nearest_candidate.sentinel_date)
                        if nearest_candidate
                        else None,
                        "best_candidate_day_diff": abs((scene.raster.date - nearest_candidate.sentinel_date).days)
                        if nearest_candidate is not None
                        else None,
                    }
                )
                write_manifest_line(
                    manifest_fp,
                    {
                        "kind": "pair",
                        "status": "skipped",
                        "reason": reason,
                        "split": "skip",
                        "scene_id": scene.scene_id,
                        "source_side": scene.source_side,
                        "planet_path": str(scene.raster.path),
                        "planet_date": serialize_date(scene.raster.date),
                        "available_stage_count": len(available_stages),
                        "available_stages": available_stages,
                        "available_candidate_count": len(scene_candidates),
                        "best_candidate_date": serialize_date(nearest_candidate.sentinel_date)
                        if nearest_candidate
                        else None,
                        "best_candidate_day_diff": abs((scene.raster.date - nearest_candidate.sentinel_date).days)
                        if nearest_candidate is not None
                        else None,
                    },
                )
                continue

            if kept_scene or existing_patch_count > 0:
                continue

            reason = classify_missing_reason(
                has_any_candidate=has_any_candidate,
                has_complete_candidate=has_complete_candidate,
                has_overlap_candidate=has_overlap_candidate,
                too_bright_only=bool(
                    any_too_bright_signal
                    and not any_too_cloudy_signal
                    and not any_empty_signal
                    and not any_edge_mismatch_signal
                    and not any_color_mismatch_signal
                ),
                too_cloudy_only=bool(
                    any_too_cloudy_signal
                    and not any_empty_signal
                    and not any_edge_mismatch_signal
                    and not any_color_mismatch_signal
                ),
                edge_mismatch_only=bool(
                    any_edge_mismatch_signal
                    and not any_empty_signal
                    and not any_too_cloudy_signal
                    and not any_too_bright_signal
                    and not any_color_mismatch_signal
                ),
                color_mismatch_only=bool(
                    any_color_mismatch_signal
                    and not any_empty_signal
                    and not any_too_cloudy_signal
                    and not any_too_bright_signal
                    and not any_edge_mismatch_signal
                ),
            )
            nearest_candidate = scene_candidates[0] if scene_candidates else None
            missing_row = {
                "scene_id": scene.scene_id,
                "source_side": scene.source_side,
                "planet_date": serialize_date(scene.raster.date),
                "year_month": scene.year_month,
                "planet_path": str(scene.raster.path),
                "reason": reason,
                "available_stage_count": len(available_stages),
                "available_stages": ",".join(available_stages),
                "available_candidate_count": len(scene_candidates),
                "best_candidate_date": serialize_date(nearest_candidate.sentinel_date) if nearest_candidate else None,
                "best_candidate_day_diff": abs((scene.raster.date - nearest_candidate.sentinel_date).days)
                if nearest_candidate is not None
                else None,
            }
            missing_rows.append(missing_row)
            write_manifest_line(
                manifest_fp,
                {
                    "kind": "pair",
                    "status": "skipped",
                    "reason": reason,
                    "split": "skip",
                    "scene_id": scene.scene_id,
                    "source_side": scene.source_side,
                    "planet_path": str(scene.raster.path),
                    "planet_date": serialize_date(scene.raster.date),
                    "available_stage_count": len(available_stages),
                    "available_stages": available_stages,
                    "available_candidate_count": len(scene_candidates),
                    "best_candidate_date": serialize_date(nearest_candidate.sentinel_date) if nearest_candidate else None,
                    "best_candidate_day_diff": abs((scene.raster.date - nearest_candidate.sentinel_date).days)
                    if nearest_candidate is not None
                    else None,
                },
            )

    missing_fieldnames = [
        "scene_id",
        "source_side",
        "planet_date",
        "year_month",
        "planet_path",
        "reason",
        "available_stage_count",
        "available_stages",
        "available_candidate_count",
        "best_candidate_date",
        "best_candidate_day_diff",
    ]
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    with missing_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=missing_fieldnames, delimiter="	")
        writer.writeheader()
        writer.writerows(missing_rows)

    print(
        f"[done] train_samples={resume_state.split_sample_totals['train']} "
        f"val_samples={resume_state.split_sample_totals['val']} "
        f"missing_pairs={len(missing_rows)} manifest={manifest_path} missing_tsv={missing_path}"
    )


if __name__ == "__main__":
    main()
