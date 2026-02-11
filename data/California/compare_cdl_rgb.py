#!/usr/bin/env python3
"""
view_rgb_vs_cdl.py  –  Quick-look RGB vs CDL

Reads Sentinel-2 B2, B3, B4 GeoTIFFs from --band_dir,
builds a stretched RGB image, and shows it next to the CDL
colour-table render.
"""

from pathlib import Path
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------
# put these two dicts near the top of the script (or import them)
class_names = {
    1:"Unknown", 12:"Corn / Sorghum / Sudan", 13:"Alfalfa & mixtures", 3:"Mixed pasture",
    5:"Misc. grain & hay", 51:"Rice", 14:"Wheat", 37:"Tomato", 15:"Misc. grasses",
    16:"Native pasture", 44:"Cotton", 27:"Leafy greens", 38:"Onions & garlic",
    20:"Melons / squash / cucumbers", 33:"Safflower", 31:"Strawberries",
    40:"Carrots", 49:"Sunflower", 17:"Potatoes", 29:"Bush berries",
    46:"Sweet potatoes", 54:"Sugar beets", 41:"Dry beans", 42:"Peppers",
}

colormap = {           # RGB triples (uint8)
    1:(0,0,0), 12:(255,0,0), 13:(0,255,0), 3:(0,0,255), 5:(255,255,0),
    51:(255,165,0), 14:(128,0,128), 37:(0,255,255), 15:(128,128,0),
    16:(255,192,203), 44:(0,128,128), 27:(128,0,0), 38:(0,128,0),
    20:(0,0,128), 33:(192,192,192), 31:(255,20,147), 40:(255,140,0),
    49:(255,215,0), 17:(139,69,19), 29:(75,0,130), 46:(210,105,30),
    54:(0,100,0), 41:(70,130,180), 42:(220,20,60),
}
# ------------------------------------------------------------------


def find_one(dir_path: Path, pattern: str) -> Path:
    matches = list(dir_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern} in {dir_path}")
    return matches[0]          # first match is fine here

def read_band(fp: Path) -> np.ndarray:
    with rasterio.open(fp) as src:
        return src.read(1).astype(np.float32)

def stretch_uint8(band, div=4000.0):
    band = np.clip(band / div, 0, 1) * 255
    return band.astype(np.uint8)

def build_rgb(dir_path: Path) -> np.ndarray:
    b2 = read_band(find_one(dir_path, "B2*.tif"))
    b3 = read_band(find_one(dir_path, "B3*.tif"))
    b4 = read_band(find_one(dir_path, "B4*.tif"))
    rgb = np.dstack([stretch_uint8(x) for x in (b4, b3, b2)])  # R,G,B
    return rgb


def read_cdl_rgb(cdl_fp: Path) -> np.ndarray:
    """
    Convert CDL raster to RGB using the fixed `colormap` dict above.
    Any value not in the dict becomes black.
    """
    with rasterio.open(cdl_fp) as src:
        band = src.read(1)

    rgb = np.zeros((*band.shape, 3), dtype=np.uint8)      # start black
    for code, color in colormap.items():
        rgb[band == code] = color

    return rgb


def build_legend(cids: List[int],
                 ctable: Dict[int, Tuple[int,int,int,int]],
                 class_names: Dict[int, str],
                 img_h: int,
                 font) -> Image.Image:
    box_h = max(12, img_h // 25)
    pad   = int(box_h * 0.2)
    rows_per_col = max(1, (img_h - pad) // (box_h + pad))
    ncols = int(np.ceil(len(cids) / rows_per_col))

    def text_w(txt):
        try:
            return font.getlength(txt)
        except AttributeError:
            return font.getsize(txt)[0]

    max_text = max(text_w(class_names.get(cid, str(cid))) for cid in cids)
    col_w = int(box_h * 1.6 + max_text + 3 * pad)
    legend = Image.new("RGB", (ncols * col_w, img_h), (255, 255, 255))
    draw   = ImageDraw.Draw(legend)

    for idx, cid in enumerate(cids):
        col = idx // rows_per_col
        row = idx % rows_per_col
        x0  = col * col_w + pad
        y0  = row * (box_h + pad)

        r, g, b, a = ctable[cid]
        draw.rectangle([x0, y0, x0 + box_h, y0 + box_h], fill=(r, g, b))
        draw.text((x0 + box_h + pad, y0 + box_h // 10),
                  class_names.get(cid, str(cid)), fill="black", font=font)
    return legend


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--band_dir", type=Path, required=True,
                    help="Folder containing B2.tif, B3.tif, B4.tif")
    pa.add_argument("--cdl", type=Path, required=True,
                    help="Path to cdl.tif")
    args = pa.parse_args()

    rgb = build_rgb(args.band_dir)
    cdl = read_cdl_rgb(args.cdl)

    # resize RGB to match CDL dimensions (W, H order in PIL)
    rgb_img = Image.fromarray(rgb)
    rgb_resized = np.asarray(rgb_img.resize(cdl.shape[1::-1], Image.BILINEAR))

    # Blend RGB and CDL with alpha
    alpha = 0.4  # adjust between 0 (only RGB) and 1 (only CDL)
    blend = (rgb_resized.astype(np.float32) * (1 - alpha) +
            cdl.astype(np.float32) * alpha).astype(np.uint8)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(cdl.shape[0] // 50, 10))
    except OSError:
        font = ImageFont.load_default()
    
    legend = build_legend(
        sorted(colormap.keys()),
        {k: (*v, 255) for k, v in colormap.items()},
        class_names,
        cdl.shape[0],
        font,
    )

    plt.figure(figsize=(12, 5))

    # 1️⃣ Blended image
    plt.subplot(1, 2, 1)
    plt.title("Blended RGB + CDL")
    plt.imshow(blend)
    plt.axis("off")

    # 2️⃣ Legend
    legend_np = np.asarray(legend)
    plt.subplot(1, 2, 2)
    plt.title("Legend")
    plt.imshow(legend_np)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
