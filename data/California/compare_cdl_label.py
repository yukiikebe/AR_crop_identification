#!/usr/bin/env python3
# make_cdl_grid.py  –  build a CDL patch mosaic + stats + one legend (Python 3.8)
"""
Combine every 'roi_mask_<X>_<Y>.tif(f)' in --cdl_dir into one big mosaic.
Left‑hand panels are regenerated from raw CDL values; right‑hand column is a
shared legend (colour + crop‑type name).

NEW: While mosaicking, the script now gathers **per‑class pixel statistics**.
Unknown or non‑selected classes are tallied under ID 0 ("Others").
A table with raw counts and percentages is printed after the image is saved.

Example:
python make_cdl_grid.py \
    --cdl_dir /home/mzarvani/ca/out1 \
    --class_map classnames.json \
    --thumb 128 \
    --out vis_cdl_grid.png
"""
from pathlib import Path
import re
import json
import argparse
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import math
from collections import Counter
import os
from glob import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
import blosc2

# --------------------------------------------------------------------------- #
#  Colour table and human‑readable names (exactly the same as your script)    #
# --------------------------------------------------------------------------- #
class_names: Dict[int, str] = {
    1:  "Unknown",
    12: "Corn / Sorghum / Sudan",
    13: "Alfalfa & mixtures",
    3:  "Mixed pasture",
    5:  "Misc. grain & hay",
    51: "Rice",
    14: "Wheat",
    37: "Tomato",
    15: "Misc. grasses",
    16: "Native pasture",
    44: "Cotton",
    27: "Leafy greens",
    38: "Onions & garlic",
    20: "Melons / squash / cucumbers",
    33: "Safflower",
    31: "Strawberries",
    40: "Carrots",
    49: "Sunflower",
    17: "Potatoes",
    29: "Bush berries",
    46: "Sweet potatoes",
    54: "Sugar beets",
    41: "Dry beans",
    42: "Peppers",
}

# (Optional) give the user‑visible label for aggregated pixels
class_names_with_other = {0: "Others", **class_names}

colormaps: Dict[int, np.ndarray] = {
    1:  np.array((0, 0, 0)),
    12: np.array((255, 0, 0)),
    13: np.array((0, 255, 0)),
    3:  np.array((0, 0, 255)),
    5:  np.array((255, 255, 0)),
    51: np.array((255, 165, 0)),
    14: np.array((128, 0, 128)),
    37: np.array((0, 255, 255)),
    15: np.array((128, 128, 0)),
    16: np.array((255, 192, 203)),
    44: np.array((0, 128, 128)),
    27: np.array((128, 0, 0)),
    38: np.array((0, 128, 0)),
    20: np.array((0, 0, 128)),
    33: np.array((192, 192, 192)),
    31: np.array((255, 20, 147)),
    40: np.array((255, 140, 0)),
    49: np.array((255, 215, 0)),
    17: np.array((139, 69, 19)),
    29: np.array((75, 0, 130)),
    46: np.array((210, 105, 30)),
    54: np.array((0, 100, 0)),
    41: np.array((70, 130, 180)),
    42: np.array((220, 20, 60)),
}

# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cdl_dir", type=Path, required=True,
                   help="Folder containing roi_mask_<X>_<Y>.tif(f)")
    p.add_argument("--label_dir", type=Path, required=True,
                   help="Folder containing roi_mask_<X>_<Y>.tif(f)")
    p.add_argument("--class_map", type=Path, required=True,
                   help="JSON file holding orig‑to‑remapped ID mapping")
    p.add_argument("--thumb", type=int, default=None,
                   help="Resize each patch (square longest side) before "
                        "mosaic (default = full resolution)")
    p.add_argument("--out", type=Path, default="cdl_grid.png",
                   help="Output image filename (PNG or JPG)")
    return p.parse_args()

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
TIF_RE = re.compile(r"roi_mask_(?P<x>\d+)_(?P<y>\d+)\.tiff?$", re.I)


def load_remap_table(path: Path) -> Dict[int, int]:
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        out[int(k)] = int(v["remapped_id"])
    return out


def tif_to_rgb(tif_path: Path,
               remap_table: Dict[int, int],
               thumb: Optional[int],
               stats_counter: Optional[Counter] = None) -> Image.Image:
    """Read 1‑band CDL tif → remap → colourise → optional resize → PIL.Image.
        If *stats_counter* is supplied, pixel counts are accumulated in it.
    """
    with rasterio.open(tif_path) as src:
        band = src.read(1)

    # remap: unknown values → -1
    remapped = np.full_like(band, -1, dtype=np.int32)
    for orig, new in remap_table.items():
        remapped[band == orig] = new

    # ---- stats ----------------------------------------------------------- #
    if stats_counter is not None:
        unique, cnts = np.unique(remapped, return_counts=True)
        for cid, c in zip(unique, cnts):
            if cid <= 1 or cid not in colormaps:   # non‑selected / unknown
                continue
            else:
                stats_counter[int(cid)] += int(c)

    # colourise
    h, w = remapped.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(remapped):
        if cid < 0 or cid not in colormaps:
            continue
        rgb[remapped == cid] = colormaps[cid]

    img = Image.fromarray(rgb)

    if thumb:
        img.thumbnail((thumb, thumb), Image.NEAREST)
    return img


def build_legend(present_ids, tile_h, font) -> Image.Image:
    """Return a PIL Image with coloured boxes + text."""
    box_h = int(tile_h * 0.6)         # height of each colour row
    pad   = int(box_h * 0.2)

    def _text_w(txt):
        try:                        # Pillow ≥10
            return font.getlength(txt)
        except AttributeError:      # Pillow <10
            return font.getsize(txt)[0]          # width from (w, h)

    w_text = max(_text_w(class_names[cid]) for cid in present_ids)

    legend_w = int(box_h * 1.6 + w_text + 3 * pad)
    legend_h = (box_h + pad) * len(present_ids)

    legend = Image.new("RGB", (legend_w, legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    for idx, cid in enumerate(reversed(present_ids)):
        y0 = idx * (box_h + pad)
        # colour rectangle
        draw.rectangle(
            [pad, y0, pad + box_h, y0 + box_h],
            fill=tuple(colormaps[cid])
        )
        # text
        draw.text(
            (pad + box_h + pad, y0 + box_h // 10),
            class_names[cid],
            fill="black",
            font=font
        )
    return legend


def get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Cross-version Pillow helper: return (w, h) of rendered *text* with *font*.
    Uses draw.textbbox() on Pillow ≥10 and falls back to draw.textsize().
    """
    if hasattr(draw, "textbbox"):           # Pillow 10+
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    return draw.textsize(text, font=font)   # older Pillow


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main_cdl():
    args = parse_args()

    remap_table = load_remap_table(args.class_map)

    # Pixel‑count accumulator --------------------------------------------- #
    pixel_counts: Counter = Counter()

    # Collect tiles -------------------------------------------------------- #
    tiles: Dict[Tuple[int, int], Path] = {}
    for p in args.cdl_dir.iterdir():
        m = TIF_RE.match(p.name)
        if m:
            x, y = int(m["x"]), int(m["y"])
            tiles[(x, y)] = p
    if not tiles:
        raise SystemExit("No roi_mask_<X>_<Y>.tif(f) found.")

    max_x = max(x for x, _ in tiles)
    max_y = max(y for _, y in tiles)

    print(f"Found {len(tiles)} tiles, grid = {(max_x+1)} × {(max_y+1)}")

    # Paste every tile ----------------------------------------------------- #
    for (x, y), tif_path in tqdm(tiles.items()):
        tif_to_rgb(tif_path, remap_table, args.thumb, pixel_counts)

    # --------------------------------------------------------------------- #
    # Print pixel statistics                                                #
    # --------------------------------------------------------------------- #
    if pixel_counts:
        total = sum(pixel_counts.values())
        print("\nPer‑class pixel statistics (after remapping):")
        print("ID  Class name                      Pixels        Share")
        print("––  –––––––––––––––––––––––––––  –––––––––  –––––––––––")
        for cid in [0] + sorted(k for k in pixel_counts if k != 0):
            cnt = pixel_counts[cid]
            share = cnt / total if total else 0
            label = class_names_with_other.get(cid, f"Class {cid}")
            print(f"{cid:>2}  {label:<30} {cnt:>10,d}  {share:7.2%}")


def read_b2frame(fp: str, dtype=np.uint8) -> np.ndarray:
    sch   = blosc2.open(fp, mode="r")
    shape = np.frombuffer(sch.vlmeta["shape"], np.int32)
    arr   = np.empty(shape, dtype=dtype)
    sch.get_slice(out=arr)
    return arr


def main_bframe2():
    args = parse_args()
    label_files = sorted(glob(os.path.join(args.label_dir, "*", "label", "*_label.b2frame")))

    for fp in tqdm(label_files, desc="Filtering tiles"):
        arr = read_b2frame(fp)                  # uint8
        unique, counts = np.unique(arr, return_counts=True)
        counts = dict(zip(unique, counts))

        counts = {k:v for k,v in counts.items() if k not in REMOVED_CLASSES}
        if not counts: continue

        seasonal_pixels = sum(v for k,v in counts.items() if k in ALL_CLASSES)
        if seasonal_pixels / arr.size <= 0.01:    # reject
            continue

        xy_patch = Path(fp).parent.parent.name          # X_Y
        tile_id  = Path(fp).stem.replace("_label","")   # iy_ix
        kept.append((xy_patch, tile_id, fp))

    return kept


if __name__ == "__main__":
    main_cdl()
