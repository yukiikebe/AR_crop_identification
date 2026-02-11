#!/usr/bin/env python3
# ar_cdl_grid.py – Arkansas CDL mosaic with indices and legend (Py 3.8)

from pathlib import Path
import re
import argparse
import yaml
from typing import Dict, Tuple, Optional, List
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from tqdm import tqdm

# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cdl_dir", type=Path, required=True,
                   help="Folder of roi_mask_<X>_<Y>.tif files")
    p.add_argument("--cdl_yaml", type=Path, required=True,
                   help="configs/Arkansas/cdl.yaml (provides num2class)")
    p.add_argument("--thumb", type=int, default=None,
                   help="Resize each tile to this square size (default full)")
    p.add_argument("--out", type=Path, default="ar_cdl_grid.png",
                   help="Output PNG/JPG filename")
    return p.parse_args()

DIR_RE = re.compile(r"(?P<x>\d+)_(?P<y>\d+)$")

# --------------------------------------------------------------------- #
# Read one CDL TIFF → RGB + colour-table dictionary                     #
# --------------------------------------------------------------------- #
def tif_to_rgb(path: Path, thumb: Optional[int]):
    with rasterio.open(path) as src:
        band = src.read(1)
        try:
            ctable = src.colormap(1)      # {value: (R,G,B,A)}
        except ValueError:
            raise SystemExit(f"No colour table in {path}")

    rgb = np.zeros((*band.shape, 3), dtype=np.uint8)
    for v, (r, g, b, a) in ctable.items():
        rgb[band == v] = (r, g, b)

    img = Image.fromarray(rgb)
    if thumb:
        img.thumbnail((thumb, thumb), Image.NEAREST)
    return img, ctable

# --------------------------------------------------------------------- #
# Build legend – auto wraps into multiple columns if needed             #
# --------------------------------------------------------------------- #
def build_legend(cids: List[int],
                 ctable: Dict[int, Tuple[int,int,int,int]],
                 class_names: Dict[int, str],
                 img_h: int,
                 font) -> Image.Image:
    # decide box size & rows per column
    box_h = max(12, img_h // 25)          # shrink if image is short
    pad   = int(box_h * 0.2)
    rows_per_col = max(1, (img_h - pad) // (box_h + pad))
    ncols = math.ceil(len(cids) / rows_per_col)

    # text width helper
    def tw(txt):
        try:
            return font.getlength(txt)
        except AttributeError:
            return font.getsize(txt)[0]

    max_text = max(tw(class_names.get(cid, str(cid))) for cid in cids)
    col_w = int(box_h * 1.6 + max_text + 3 * pad)

    legend = Image.new("RGB", (ncols * col_w, img_h), (255, 255, 255))
    draw   = ImageDraw.Draw(legend)

    for idx, cid in enumerate(cids):
        col = idx // rows_per_col
        row = idx % rows_per_col
        x0  = col * col_w + pad
        y0  = row * (box_h + pad)

        r, g, b, a = ctable[cid]
        draw.rectangle([x0, y0, x0 + box_h, y0 + box_h],
                       fill=(r, g, b))
        draw.text((x0 + box_h + pad, y0 + box_h // 10),
                  class_names.get(cid, str(cid)),
                  fill="black", font=font)
    return legend

# --------------------------------------------------------------------- #
def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load class-name mapping from YAML                                  #
    # ------------------------------------------------------------------ #
    with open(args.cdl_yaml) as f:
        num2class = yaml.safe_load(f)["num2class"]
        num2class = {int(k): v for k, v in num2class.items()}

    # ------------------------------------------------------------------ #
    # Discover tiles                                                     #
    # ------------------------------------------------------------------ #
    tiles: Dict[Tuple[int, int], Path] = {}
    for sub in args.cdl_dir.iterdir():
        if not sub.is_dir():
            continue
        m = DIR_RE.match(sub.name)
        if not m:
            continue
        x, y = int(m["x"]), int(m["y"])
        cdl_path = sub / "cdl.tif"
        if cdl_path.exists():
            tiles[(x, y)] = cdl_path
    if not tiles:
        raise SystemExit("No roi_mask_<X>_<Y>.tif files found.")

    max_x = max(x for x, _ in tiles)
    max_y = max(y for _, y in tiles)
    print(f"Found {len(tiles)} tiles, grid {max_x+1} × {max_y+1}")

    # ------------------------------------------------------------------ #
    # Build mosaic                                                       #
    # ------------------------------------------------------------------ #
    sample_img, sample_ctable = tif_to_rgb(next(iter(tiles.values())), args.thumb)
    tile_w, tile_h = sample_img.size
    mosaic_w = (max_x + 1) * tile_w
    mosaic_h = (max_y + 1) * tile_h
    mosaic   = Image.new("RGB", (mosaic_w, mosaic_h), (220, 220, 220))

    # keep track of all class IDs to show in legend
    all_ctable = sample_ctable.copy()

    for (x, y), tif_path in tqdm(tiles.items(), desc="Tiles"):
        img, ctbl = tif_to_rgb(tif_path, args.thumb)
        all_ctable.update(ctbl)
        mosaic.paste(img, (x * tile_w, (max_y - y) * tile_h))

    present_cids = sorted(all_ctable.keys())

    # ------------------------------------------------------------------ #
    # Fonts                                                              #
    # ------------------------------------------------------------------ #
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(tile_h // 5, 10))
    except OSError:
        font = ImageFont.load_default()

    # Legend
    legend = build_legend(present_cids, all_ctable, num2class,
                          mosaic_h, font)

    # ------------------------------------------------------------------ #
    # Row & column index gutters                                         #
    # ------------------------------------------------------------------ #
    y_gutter_w = max(60, tile_w // 3)
    x_gutter_h = max(40, tile_h // 3)

    y_gutter = Image.new("RGB", (y_gutter_w, mosaic_h), (255, 255, 255))
    draw_y = ImageDraw.Draw(y_gutter)
    for y in range(max_y + 1):
        ypix = (max_y - y) * tile_h + tile_h // 2
        txt  = str(y)
        w, h = draw_y.textsize(txt, font=font)
        draw_y.text((y_gutter_w - w - 4, ypix - h // 2), txt,
                    fill="black", font=font)

    x_gutter = Image.new("RGB", (mosaic_w, x_gutter_h), (255, 255, 255))
    draw_x = ImageDraw.Draw(x_gutter)
    for x in range(max_x + 1):
        xpix = x * tile_w + tile_w // 2
        txt  = str(x)
        w, h = draw_x.textsize(txt, font=font)
        draw_x.text((xpix - w // 2, 4), txt, fill="black", font=font)

    # ------------------------------------------------------------------ #
    # Assemble final canvas                                              #
    # ------------------------------------------------------------------ #
    final_w = y_gutter_w + mosaic_w + legend.width
    final_h = mosaic_h + x_gutter_h
    final   = Image.new("RGB", (final_w, final_h), (255, 255, 255))

    final.paste(y_gutter, (0,                0))
    final.paste(mosaic,   (y_gutter_w,       0))
    final.paste(legend,   (y_gutter_w + mosaic_w, 0))
    final.paste(x_gutter, (y_gutter_w,       mosaic_h))

    final.save(args.out)
    print("✅ wrote", args.out)

if __name__ == "__main__":
    main()
