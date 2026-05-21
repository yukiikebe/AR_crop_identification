#!/usr/bin/env python
"""Score Sentinel/Planet debug pairs with a simple edge-agreement metric.

This is intended as a lightweight QA tool for datasets created by
build_fastdiffsr_sentinel_planet_dataset.py. It compares the debug PNGs:

    debug_png/sr_64_256/<id>.png
    debug_png/hr_256/<id>.png

By default, it checks sample ids 000000 through 000100 inclusive.

The score is a correlation between gradient magnitudes. Low scores usually
mean the Planet patch has visible structure that is weak, shifted, shadowed,
or missing in the Sentinel patch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageFilter


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def load_rgb_with_blur(path: Path, blur_sigma: float) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if blur_sigma > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    return np.asarray(image, dtype=np.float32)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def edge_quality_score(sr_png: Path, hr_png: Path, hr_blur_sigma: float = 0.0) -> tuple[float, float, float]:
    sr = load_rgb(sr_png)
    hr = load_rgb_with_blur(hr_png, hr_blur_sigma)

    if sr.shape != hr.shape:
        raise ValueError(f"shape mismatch: {sr_png} {sr.shape} vs {hr_png} {hr.shape}")

    sr_edge = gradient_magnitude(rgb_to_gray(sr))
    hr_edge = gradient_magnitude(rgb_to_gray(hr))
    score = normalized_corr(sr_edge, hr_edge)
    return score, float(sr_edge.mean()), float(hr_edge.mean())


def iter_pair_ids(explicit_ids: list[str], start_id: int = 0, end_id: int = 100) -> list[str]:
    if explicit_ids:
        return [sample_id if sample_id.endswith(".png") else f"{sample_id}.png" for sample_id in explicit_ids]
    return [f"{sample_id:06d}.png" for sample_id in range(start_id, end_id + 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path such as make_dataset/fastdiffsr_s2_planet_clean_v2/val_64_256",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=[],
        help="Optional sample ids, for example: 000005 000012",
    )
    parser.add_argument(
        "--min-edge-corr",
        type=float,
        default=0.10,
        help="Pairs below this edge correlation are reported as reject.",
    )
    parser.add_argument(
        "--edge-blur-sigma",
        type=float,
        default=0.0,
        help=(
            "Optional Gaussian blur sigma applied to Planet HR debug PNGs before "
            "edge comparison. Try 1.0-2.0 to reduce Planet-only narrow paths/textures."
        ),
    )
    args = parser.parse_args()
    if args.edge_blur_sigma < 0.0:
        raise ValueError("--edge-blur-sigma must be non-negative.")

    sr_dir = args.dataset_root / "debug_png" / "sr_64_256"
    hr_dir = args.dataset_root / "debug_png" / "hr_256"

    bad = 0
    total = 0
    for filename in iter_pair_ids(args.ids):
        sr_png = sr_dir / filename
        hr_png = hr_dir / filename
        if not sr_png.exists() or not hr_png.exists():
            print(f"{filename}\tmissing_pair")
            continue

        score, sr_edge_mean, hr_edge_mean = edge_quality_score(
            sr_png,
            hr_png,
            hr_blur_sigma=float(args.edge_blur_sigma),
        )
        decision = "reject" if score < args.min_edge_corr else "keep"
        print(
            f"{filename}\t{decision}\t"
            f"edge_corr={score:.4f}\t"
            f"sr_edge_mean={sr_edge_mean:.4f}\t"
            f"hr_edge_mean={hr_edge_mean:.4f}\t"
            f"edge_blur_sigma={float(args.edge_blur_sigma):.2f}"
        )
        bad += int(decision == "reject")
        total += 1

    print(
        f"summary\ttotal={total}\treject={bad}\t"
        f"threshold={args.min_edge_corr:.4f}\t"
        f"edge_blur_sigma={float(args.edge_blur_sigma):.2f}"
    )


if __name__ == "__main__":
    main()
