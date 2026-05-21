#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


TIFF_EXTENSIONS = (".tif", ".tiff")


@dataclass(frozen=True)
class PairDecision:
    split: str
    sample_id: str
    status: str
    reason: str
    hr_mean_brightness: float | None
    hr_bright_ratio: float | None
    lr_mean_brightness: float | None
    lr_bright_ratio: float | None
    hr_path: Path
    lr_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cleaned FastDiffSR dataset by copying paired samples from an "
            "existing dataset root while rejecting washed-out bright HR or LR patches."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Existing dataset root containing split directories like train_64_256 and val_64_256.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New dataset root to write. Refuses to use an existing path unless --allow-existing is set.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Split directory names to filter. Defaults to all child dirs containing hr_* and lr_* folders.",
    )
    parser.add_argument("--hr-bright-threshold", type=int, default=245)
    parser.add_argument("--hr-max-bright-ratio", type=float, default=0.35)
    parser.add_argument("--hr-max-mean-brightness", type=float, default=240.0)
    parser.add_argument("--hr-min-mean-brightness", type=float, default=0.0)
    parser.add_argument("--hr-white-brightness-threshold", type=int, default=170)
    parser.add_argument("--hr-white-saturation-threshold", type=float, default=0.25)
    parser.add_argument("--hr-max-white-ratio", type=float, default=1.0)
    parser.add_argument("--hr-dark-threshold", type=int, default=45)
    parser.add_argument("--hr-max-dark-ratio", type=float, default=1.0)
    parser.add_argument("--lr-bright-threshold", type=int, default=245)
    parser.add_argument("--lr-max-bright-ratio", type=float, default=0.50)
    parser.add_argument("--lr-max-mean-brightness", type=float, default=245.0)
    parser.add_argument("--lr-min-mean-brightness", type=float, default=0.0)
    parser.add_argument("--lr-white-brightness-threshold", type=int, default=170)
    parser.add_argument("--lr-white-saturation-threshold", type=float, default=0.25)
    parser.add_argument("--lr-max-white-ratio", type=float, default=1.0)
    parser.add_argument("--lr-dark-threshold", type=int, default=45)
    parser.add_argument("--lr-max-dark-ratio", type=float, default=1.0)
    parser.add_argument(
        "--max-bright-ratio",
        type=float,
        default=None,
        help="Deprecated alias for --hr-max-bright-ratio.",
    )
    parser.add_argument(
        "--max-mean-brightness",
        type=float,
        default=None,
        help="Deprecated alias for --hr-max-mean-brightness.",
    )
    parser.add_argument(
        "--bright-threshold",
        type=int,
        default=None,
        help="Deprecated alias for --hr-bright-threshold.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Hard-link files into the output instead of copying them. Saves space but shares file data.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow writing into an existing output root. Existing files with the same names may be replaced.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report kept/rejected counts; do not write the output dataset.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bright_threshold is not None:
        args.hr_bright_threshold = args.bright_threshold
    if args.max_bright_ratio is not None:
        args.hr_max_bright_ratio = args.max_bright_ratio
    if args.max_mean_brightness is not None:
        args.hr_max_mean_brightness = args.max_mean_brightness

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not args.input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {args.input_root}")
    if args.output_root.exists() and not args.allow_existing and not args.dry_run:
        raise FileExistsError(
            f"Output root already exists: {args.output_root}. Use a new path or pass --allow-existing."
        )
    for name in ("hr_bright_threshold", "lr_bright_threshold"):
        if not (0 <= int(getattr(args, name)) <= 255):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 255].")
    for name in ("hr_max_bright_ratio", "lr_max_bright_ratio"):
        if not (0.0 <= float(getattr(args, name)) <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    for name in ("hr_max_mean_brightness", "lr_max_mean_brightness"):
        if not (0.0 <= float(getattr(args, name)) <= 255.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 255].")
    for name in ("hr_min_mean_brightness", "lr_min_mean_brightness"):
        if not (0.0 <= float(getattr(args, name)) <= 255.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 255].")
    if float(args.hr_min_mean_brightness) > float(args.hr_max_mean_brightness):
        raise ValueError("--hr-min-mean-brightness must be <= --hr-max-mean-brightness.")
    if float(args.lr_min_mean_brightness) > float(args.lr_max_mean_brightness):
        raise ValueError("--lr-min-mean-brightness must be <= --lr-max-mean-brightness.")
    for name in ("hr_white_brightness_threshold", "hr_dark_threshold", "lr_white_brightness_threshold", "lr_dark_threshold"):
        if not (0 <= int(getattr(args, name)) <= 255):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 255].")
    for name in ("hr_white_saturation_threshold", "lr_white_saturation_threshold"):
        if not (0.0 <= float(getattr(args, name)) <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    for name in ("hr_max_white_ratio", "hr_max_dark_ratio", "lr_max_white_ratio", "lr_max_dark_ratio"):
        if not (0.0 <= float(getattr(args, name)) <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")


def discover_split_names(input_root: Path) -> list[str]:
    split_names: list[str] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if find_hr_dir(child) is None:
            continue
        if not any(path.is_dir() and path.name.startswith("lr_") for path in child.iterdir()):
            continue
        split_names.append(child.name)
    return split_names


def find_hr_dir(split_dir: Path) -> Path | None:
    hr_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith("hr_"))
    if not hr_dirs:
        return None
    if len(hr_dirs) > 1:
        raise RuntimeError(f"Expected one hr_* directory in {split_dir}, found {len(hr_dirs)}.")
    return hr_dirs[0]


def find_lr_dir(split_dir: Path) -> Path | None:
    lr_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith("lr_"))
    if not lr_dirs:
        return None
    if len(lr_dirs) > 1:
        raise RuntimeError(f"Expected one lr_* directory in {split_dir}, found {len(lr_dirs)}.")
    return lr_dirs[0]


def iter_tiffs(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS:
            yield path


def brightness_stats(path: Path, bright_threshold: int) -> tuple[float, float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    brightness = arr.astype(np.float32, copy=False).mean(axis=-1)
    return float(brightness.mean()), float((brightness >= bright_threshold).mean())


def white_dark_stats(
    path: Path,
    *,
    white_brightness_threshold: int,
    white_saturation_threshold: float,
    dark_threshold: int,
) -> tuple[float, float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB")).astype(np.float32, copy=False)
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


def paired_dirs(split_dir: Path) -> list[Path]:
    dirs = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_dir() and path.name != "debug_png" and not path.name.startswith(".")
    )
    debug_root = split_dir / "debug_png"
    if debug_root.is_dir():
        dirs.extend(sorted(path for path in debug_root.iterdir() if path.is_dir()))
    return dirs


def build_file_index(directories: list[Path]) -> dict[Path, dict[str, list[Path]]]:
    index: dict[Path, dict[str, list[Path]]] = {}
    for directory in directories:
        by_stem: dict[str, list[Path]] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file():
                by_stem.setdefault(path.stem, []).append(path)
        index[directory] = by_stem
    return index


def files_for_sample(index: dict[Path, dict[str, list[Path]]], sample_id: str) -> list[Path]:
    files: list[Path] = []
    for by_stem in index.values():
        files.extend(by_stem.get(sample_id, []))
    return files


def required_tiff_dirs(split_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in split_dir.iterdir()
        if path.is_dir() and path.name != "debug_png" and not path.name.startswith(".")
    )


def has_required_pair_files(
    required_dirs: list[Path],
    index: dict[Path, dict[str, list[Path]]],
    sample_id: str,
) -> bool:
    for directory in required_dirs:
        if sample_id not in index.get(directory, {}):
            return False
    return True


def copy_or_link(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link:
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def copy_sample_files(
    *,
    input_split_dir: Path,
    output_split_dir: Path,
    file_index: dict[Path, dict[str, list[Path]]],
    sample_id: str,
    link: bool,
) -> None:
    for src in files_for_sample(file_index, sample_id):
        rel = src.relative_to(input_split_dir)
        copy_or_link(src, output_split_dir / rel, link=link)


def write_decisions(path: Path, decisions: list[PairDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "split",
                "sample_id",
                "status",
                "reason",
                "hr_mean_brightness",
                "hr_bright_ratio",
                "lr_mean_brightness",
                "lr_bright_ratio",
                "hr_path",
                "lr_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for item in decisions:
            writer.writerow(
                {
                    "split": item.split,
                    "sample_id": item.sample_id,
                    "status": item.status,
                    "reason": item.reason,
                    "hr_mean_brightness": f"{item.hr_mean_brightness:.6f}"
                    if item.hr_mean_brightness is not None
                    else "",
                    "hr_bright_ratio": f"{item.hr_bright_ratio:.6f}" if item.hr_bright_ratio is not None else "",
                    "lr_mean_brightness": f"{item.lr_mean_brightness:.6f}"
                    if item.lr_mean_brightness is not None
                    else "",
                    "lr_bright_ratio": f"{item.lr_bright_ratio:.6f}" if item.lr_bright_ratio is not None else "",
                    "hr_path": str(item.hr_path),
                    "lr_path": str(item.lr_path) if item.lr_path is not None else "",
                }
            )


def write_filter_summary(path: Path, args: argparse.Namespace, decisions: list[PairDecision]) -> None:
    by_split: dict[str, dict[str, int]] = {}
    for item in decisions:
        split_counts = by_split.setdefault(item.split, {"kept": 0, "rejected": 0})
        if item.status == "kept":
            split_counts["kept"] += 1
        else:
            split_counts["rejected"] += 1
    payload: dict[str, Any] = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "hr_bright_threshold": int(args.hr_bright_threshold),
        "hr_max_bright_ratio": float(args.hr_max_bright_ratio),
        "hr_max_mean_brightness": float(args.hr_max_mean_brightness),
        "hr_min_mean_brightness": float(args.hr_min_mean_brightness),
        "hr_white_brightness_threshold": int(args.hr_white_brightness_threshold),
        "hr_white_saturation_threshold": float(args.hr_white_saturation_threshold),
        "hr_max_white_ratio": float(args.hr_max_white_ratio),
        "hr_dark_threshold": int(args.hr_dark_threshold),
        "hr_max_dark_ratio": float(args.hr_max_dark_ratio),
        "lr_bright_threshold": int(args.lr_bright_threshold),
        "lr_max_bright_ratio": float(args.lr_max_bright_ratio),
        "lr_max_mean_brightness": float(args.lr_max_mean_brightness),
        "lr_min_mean_brightness": float(args.lr_min_mean_brightness),
        "lr_white_brightness_threshold": int(args.lr_white_brightness_threshold),
        "lr_white_saturation_threshold": float(args.lr_white_saturation_threshold),
        "lr_max_white_ratio": float(args.lr_max_white_ratio),
        "lr_dark_threshold": int(args.lr_dark_threshold),
        "lr_max_dark_ratio": float(args.lr_max_dark_ratio),
        "splits": by_split,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def filter_manifest(input_root: Path, output_root: Path, rejected_keys: set[tuple[str, str]]) -> None:
    manifest_path = input_root / "pairs_manifest.jsonl"
    if not manifest_path.exists():
        return
    out_path = output_root / "pairs_manifest.filtered.jsonl"
    with manifest_path.open("r", encoding="utf-8") as src_fp, out_path.open("w", encoding="utf-8") as dst_fp:
        for line in src_fp:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            split = payload.get("split")
            sample_id = payload.get("sample_id")
            if isinstance(split, str) and isinstance(sample_id, str) and (split, sample_id) in rejected_keys:
                continue
            dst_fp.write(json.dumps(payload, sort_keys=True) + "\n")


def filter_split(args: argparse.Namespace, split_name: str) -> list[PairDecision]:
    input_split_dir = args.input_root / split_name
    output_split_dir = args.output_root / split_name
    hr_dir = find_hr_dir(input_split_dir)
    if hr_dir is None:
        raise RuntimeError(f"No hr_* directory found in split: {input_split_dir}")
    lr_dir = find_lr_dir(input_split_dir)
    if lr_dir is None:
        raise RuntimeError(f"No lr_* directory found in split: {input_split_dir}")
    sample_dirs = paired_dirs(input_split_dir)
    sample_index = build_file_index(sample_dirs)
    required_dirs = required_tiff_dirs(input_split_dir)

    decisions: list[PairDecision] = []
    for hr_path in iter_tiffs(hr_dir):
        sample_id = hr_path.stem
        if not has_required_pair_files(required_dirs, sample_index, sample_id):
            decisions.append(
                PairDecision(split_name, sample_id, "rejected", "missing_pair_file", None, None, None, None, hr_path, None)
            )
            continue

        lr_candidates = sample_index.get(lr_dir, {}).get(sample_id, [])
        lr_path = lr_candidates[0] if lr_candidates else None
        if lr_path is None:
            decisions.append(
                PairDecision(split_name, sample_id, "rejected", "missing_lr_file", None, None, None, None, hr_path, None)
            )
            continue

        hr_mean_brightness, hr_bright_ratio = brightness_stats(hr_path, int(args.hr_bright_threshold))
        lr_mean_brightness, lr_bright_ratio = brightness_stats(lr_path, int(args.lr_bright_threshold))
        hr_white_ratio, hr_dark_ratio = white_dark_stats(
            hr_path,
            white_brightness_threshold=int(args.hr_white_brightness_threshold),
            white_saturation_threshold=float(args.hr_white_saturation_threshold),
            dark_threshold=int(args.hr_dark_threshold),
        )
        lr_white_ratio, lr_dark_ratio = white_dark_stats(
            lr_path,
            white_brightness_threshold=int(args.lr_white_brightness_threshold),
            white_saturation_threshold=float(args.lr_white_saturation_threshold),
            dark_threshold=int(args.lr_dark_threshold),
        )

        if hr_mean_brightness > float(args.hr_max_mean_brightness):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "planet_mean_brightness",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if hr_mean_brightness < float(args.hr_min_mean_brightness):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "planet_min_mean_brightness",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if hr_bright_ratio > float(args.hr_max_bright_ratio):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "planet_bright_ratio",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if hr_white_ratio > float(args.hr_max_white_ratio):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "planet_white_ratio",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if hr_dark_ratio > float(args.hr_max_dark_ratio):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "planet_dark_ratio",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if lr_mean_brightness > float(args.lr_max_mean_brightness):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "sentinel_mean_brightness",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if lr_mean_brightness < float(args.lr_min_mean_brightness):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "sentinel_min_mean_brightness",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if lr_bright_ratio > float(args.lr_max_bright_ratio):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "sentinel_bright_ratio",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if lr_white_ratio > float(args.lr_max_white_ratio):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "sentinel_white_ratio",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue
        if lr_dark_ratio > float(args.lr_max_dark_ratio):
            decisions.append(
                PairDecision(
                    split_name,
                    sample_id,
                    "rejected",
                    "sentinel_dark_ratio",
                    hr_mean_brightness,
                    hr_bright_ratio,
                    lr_mean_brightness,
                    lr_bright_ratio,
                    hr_path,
                    lr_path,
                )
            )
            continue

        decisions.append(
            PairDecision(
                split_name,
                sample_id,
                "kept",
                "kept",
                hr_mean_brightness,
                hr_bright_ratio,
                lr_mean_brightness,
                lr_bright_ratio,
                hr_path,
                lr_path,
            )
        )
        if not args.dry_run:
            copy_sample_files(
                input_split_dir=input_split_dir,
                output_split_dir=output_split_dir,
                file_index=sample_index,
                sample_id=sample_id,
                link=bool(args.link),
            )
    return decisions


def main() -> None:
    args = parse_args()
    validate_args(args)

    split_names = args.splits or discover_split_names(args.input_root)
    if not split_names:
        raise RuntimeError(f"No split directories found under {args.input_root}")

    all_decisions: list[PairDecision] = []
    for split_name in split_names:
        decisions = filter_split(args, split_name)
        all_decisions.extend(decisions)
        kept = sum(1 for item in decisions if item.status == "kept")
        rejected = len(decisions) - kept
        print(f"[split] {split_name}: kept={kept} rejected={rejected}")

    if not args.dry_run:
        write_decisions(args.output_root / "brightness_filter_decisions.tsv", all_decisions)
        write_filter_summary(args.output_root / "brightness_filter_summary.json", args, all_decisions)
        rejected_keys = {(item.split, item.sample_id) for item in all_decisions if item.status != "kept"}
        filter_manifest(args.input_root, args.output_root, rejected_keys)

    kept_total = sum(1 for item in all_decisions if item.status == "kept")
    rejected_total = len(all_decisions) - kept_total
    print(f"[done] kept={kept_total} rejected={rejected_total}")


if __name__ == "__main__":
    main()
