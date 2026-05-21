#!/usr/bin/env python3
"""Create side-by-side preview PNGs from reconstruction manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


def read_rgb(path: Path) -> Image.Image:
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3])
    arr = np.moveaxis(data, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("status") == "kept":
                    records.append(record)
    return records


def choose_records(records: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records
    indices = np.linspace(0, len(records) - 1, limit, dtype=int)
    return [records[int(i)] for i in indices]


def make_comparison(record: dict, out_path: Path, use_sr_upsample_left: bool, label_height: int) -> None:
    lr_path = Path(record["sr_path"] if use_sr_upsample_left else record["lr_path"])
    sr_path = Path(record.get("reconstruction_path") or record["sr_path"])

    lr = read_rgb(lr_path)
    if lr.size != (256, 256):
        lr = lr.resize((256, 256), Image.Resampling.NEAREST)
    sr = read_rgb(sr_path)
    if sr.size != (256, 256):
        sr = sr.resize((256, 256), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (512, 256 + label_height), "white")
    canvas.paste(lr, (0, label_height))
    canvas.paste(sr, (256, label_height))

    if label_height:
        sample_id = record.get("sample_id", lr_path.stem)
        date = record.get("date", "unknown-date")
        grid_id = record.get("grid_id", "unknown-grid")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), "LR 64 -> 256", fill=(0, 0, 0))
        draw.text((264, 4), "FastDiffSR reconstruction", fill=(0, 0, 0))
        draw.text((8, 20), f"id={sample_id} date={date} grid={grid_id}", fill=(0, 0, 0))
    canvas.save(out_path)


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int) -> None:
    if not image_paths:
        return
    thumbs = [Image.open(p).resize((256, 146), Image.Resampling.BICUBIC) for p in image_paths]
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 256, rows * 146), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 256
        y = (i // cols) * 146
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--contact-cols", type=int, default=4)
    parser.add_argument("--left", choices=("lr_64", "sr_64_256"), default="sr_64_256")
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = choose_records(load_manifest(args.manifest), args.limit)
    written: list[Path] = []
    for record in records:
        sample_id = record.get("sample_id", Path(record["lr_path"]).stem)
        out_path = args.out / f"{sample_id}_lr_vs_fastdiffsr.png"
        make_comparison(
            record,
            out_path,
            use_sr_upsample_left=args.left == "sr_64_256",
            label_height=38 if args.labels else 0,
        )
        written.append(out_path)

    if args.contact_sheet:
        make_contact_sheet(written, args.out / "contact_sheet_lr_vs_fastdiffsr.png", args.contact_cols)

    print(f"[done] wrote {len(written)} comparison PNGs to {args.out}")
    if args.contact_sheet:
        print(f"[done] contact sheet: {args.out / 'contact_sheet_lr_vs_fastdiffsr.png'}")


if __name__ == "__main__":
    main()
