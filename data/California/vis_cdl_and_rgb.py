#!/usr/bin/env python3
# ca_overlay_tile.py – RGB + label-tile overlay for California dataset

from pathlib import Path
import re, argparse, json
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio, blosc2
from tqdm import tqdm

# ─────────── legend & colours (exactly as in make_cdl_grid) ────────────
class_names = {  # id : human-readable
    1:"Unknown", 12:"Corn/Sorghum/Sudan", 13:"Alfalfa & mixtures", 3:"Mixed pasture",
    5:"Misc grain & hay", 51:"Rice", 14:"Wheat", 37:"Tomato", 15:"Misc grasses",
    16:"Native pasture", 44:"Cotton", 27:"Leafy greens", 38:"Onions & garlic",
    20:"Melons/squash/cucumbers", 33:"Safflower", 31:"Strawberries",
    40:"Carrots", 49:"Sunflower", 17:"Potatoes", 29:"Bush berries",
    46:"Sweet potatoes", 54:"Sugar beets", 41:"Dry beans", 42:"Peppers",
}
colormaps = {  # id : RGB
    1:(0,0,0), 12:(255,0,0), 13:(0,255,0), 3:(0,0,255), 5:(255,255,0),
    51:(255,165,0), 14:(128,0,128), 37:(0,255,255), 15:(128,128,0),
    16:(255,192,203), 44:(0,128,128), 27:(128,0,0), 38:(0,128,0),
    20:(0,0,128), 33:(192,192,192), 31:(255,20,147), 40:(255,140,0),
    49:(255,215,0), 17:(139,69,19), 29:(75,0,130), 46:(210,105,30),
    54:(0,100,0), 41:(70,130,180), 42:(220,20,60),
}

PATCH_RE_IMG   = re.compile(r"(?P<row>\d+)_(?P<col>\d+)_img\.b2frame$",   re.I)
PATCH_RE_LABEL = re.compile(r"(?P<row>\d+)_(?P<col>\d+)_label\.b2frame$", re.I)

# ─────────────────────── CLI ───────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frame_dir", type=Path, required=True,
                   help="<X_Y>/img/ folder with *_img.b2frame tiles")
    p.add_argument("--label_dir", type=Path, default=None,
                   help="Optional <X_Y>/label/ folder; by default derived from frame_dir")
    p.add_argument("--class_map", type=Path, required=True,
                   help="JSON remap table (orig ID ➜ remapped ID)")
    p.add_argument("--thumb", type=int, default=None, help="Thumbnail max-side")
    p.add_argument("--alpha", type=float, default=0.4, help="Blend opacity")
    p.add_argument("--out", type=Path, default="overlay.png")
    return p.parse_args()

# ─────────────────────── I/O helpers ───────────────────────
def read_b2frame(fp: Path, dtype) -> np.ndarray:
    sch = blosc2.open(str(fp), mode="r")
    shape = np.frombuffer(sch.vlmeta["shape"], np.int32)
    arr   = np.empty(shape, dtype=dtype)
    sch.get_slice(out=arr)
    return arr

# RGB loader (same as before)
def tile_rgb(fp: Path) -> np.ndarray:
    arr = read_b2frame(fp, np.uint16)      # (T,24,24,11)
    rgb16 = np.take(arr[6], [2,1,0], axis=-1).astype(np.float32)
    return (np.clip(rgb16/4000, 0,1)*255).astype(np.uint8)

# label-to-RGB using remap + colormaps
def tile_label_rgb(fp: Path, remap: Dict[int,int]) -> np.ndarray:
    lbl = read_b2frame(fp, np.uint8)       # (24,24)
    rgb = np.zeros((*lbl.shape,3), np.uint8)
    for orig, new in remap.items():
        if new in colormaps:
            rgb[lbl==orig] = colormaps[new]
    return rgb

# Build mosaics ---------------------------------------------------------
def build_mosaic(folder: Path, pattern, load_fn) -> np.ndarray:
    tiles, coords = {}, []
    for p in folder.glob(pattern):
        m = (PATCH_RE_IMG if "_img" in pattern else PATCH_RE_LABEL).match(p.name)
        if m:
            r,c = int(m["row"]), int(m["col"])
            tiles[(r,c)] = p
            coords.append((r,c))
    if not tiles:
        raise SystemExit(f"No {pattern} found in {folder}")
    max_r = max(r for r,_ in coords); max_c = max(c for _,c in coords)
    mosaic = np.zeros(((max_r+24), (max_c+24), 3), np.uint8)
    for (r,c), fp in tqdm(tiles.items(), desc=pattern):
        mosaic[r:r+24, c:c+24] = load_fn(fp)
    return mosaic

# Legend builder (unchanged)
def build_legend(ids: List[int], img_h:int, font) -> Image.Image:
    box_h = max(12, img_h//25); pad=int(box_h*.2)
    rows = max(1,(img_h-pad)//(box_h+pad)); ncol=int(np.ceil(len(ids)/rows))
    tw = lambda t: font.getlength(t) if hasattr(font,'getlength') else font.getsize(t)[0]
    col_w = int(box_h*1.6 + max(tw(class_names[i]) for i in ids)+3*pad)
    leg=Image.new("RGB",(ncol*col_w,img_h),(255,255,255)); draw=ImageDraw.Draw(leg)
    for k,cid in enumerate(ids):
        col,row=k//rows,k%rows; x0=col*col_w+pad; y0=row*(box_h+pad)
        draw.rectangle([x0,y0,x0+box_h,y0+box_h], fill=tuple(colormaps[cid]))
        draw.text((x0+box_h+pad, y0+box_h//10), class_names[cid], fill="black", font=font)
    return leg

# ─────────────────────── Main ───────────────────────
def main():
    args = parse_args()
    label_dir = args.label_dir or args.frame_dir.parent / "label"
    remap = json.load(open(args.class_map)); remap={int(k):int(v["remapped_id"]) for k,v in remap.items()}

    rgb  = build_mosaic(args.frame_dir, "*_img.b2frame",   tile_rgb)
    cdl  = build_mosaic(label_dir,     "*_label.b2frame", lambda fp: tile_label_rgb(fp, remap))

    if rgb.shape[:2]!=cdl.shape[:2]:
        h,w=cdl.shape[:2]; rgb=rgb[:h,:w]

    if args.thumb:
        res = lambda a: np.asarray(Image.fromarray(a).resize((args.thumb,args.thumb),Image.BILINEAR))
        rgb, cdl = res(rgb), res(cdl)

    blend = (rgb.astype(np.float32)*(1-args.alpha)+cdl.astype(np.float32)*args.alpha).astype(np.uint8)

    font = ImageFont.truetype("DejaVuSans.ttf", size=max(blend.shape[0]//50,10)) \
           if Path("/usr/share/fonts").exists() else ImageFont.load_default()
    legend = build_legend(sorted(colormaps.keys()), blend.shape[0], font)

    canvas = Image.new("RGB",(blend.shape[1]+legend.width,blend.shape[0]),(255,255,255))
    canvas.paste(Image.fromarray(blend),(0,0)); canvas.paste(legend,(blend.shape[1],0))
    canvas.save(args.out)
    print("✅ wrote", args.out)

if __name__=="__main__":
    main()
