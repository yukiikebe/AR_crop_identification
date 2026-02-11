#!/usr/bin/env python3
# overlay_all_subregions.py — visualize ALL <X_Y> subregions in a grid
# - uses arkansas_data.yaml (classes[1] only)
# - optional remap JSON
# - blends only where kept classes are present
# - orientation matches ar_cdl_grid.py: paste at (x, maxY-y)

from pathlib import Path
import re, argparse, json, colorsys
from typing import Dict, List, Tuple, Set
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import blosc2, yaml
from tqdm import tqdm

PATCH_RE_IMG   = re.compile(r"(?P<row>\d+)_(?P<col>\d+)_img\.b2frame$",   re.I)
PATCH_RE_LABEL = re.compile(r"(?P<row>\d+)_(?P<col>\d+)_label\.b2frame$", re.I)
SUBREGION_RE   = re.compile(r"^(?P<X>\d+)_(?P<Y>\d+)$")

# ────────────────────────── YAML + palette ──────────────────────────
def load_yaml_classes(yaml_path: Path) -> Tuple[Dict[int, str], Set[int]]:
    cfg = yaml.safe_load(yaml_path.read_text())
    classes = cfg.get("classes", {})
    def dict_int(d): return {int(k): v for k, v in d.items()} if isinstance(d, dict) else {}
    keep_map = dict_int(classes.get(1, {}))  # crop classes only
    return keep_map, set(keep_map.keys())

def gen_palette(keep_ids: set[int], seed: float = 0.11) -> dict[int, tuple[int,int,int]]:
    """
    Generate many visually distinct colors for class IDs.
    - seed: starting hue in [0,1) to vary the whole palette.
    Uses 4 S/V rings to keep neighbors distinct; good for 50–100 classes.
    """
    ids = sorted(keep_ids)
    phi = 0.6180339887498949  # golden ratio conjugate
    # (S, V) rings: vivid, mid, pastel, dark-ish (kept >=0.65 V for white bg contrast)
    rings = [
        (0.78, 0.95),
        (0.60, 0.85),
        (0.85, 0.75),
        (0.70, 0.68),
    ]
    ring_period = 24  # switch ring every 24 ids to spread similar hues apart

    cm: dict[int, tuple[int,int,int]] = {}
    for i, cid in enumerate(ids):
        h = (seed + i * phi) % 1.0
        s, v = rings[(i // ring_period) % len(rings)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cm[cid] = (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))
    return cm

# ───────────────────────────── I/O helpers ───────────────────────────
def read_b2frame(fp: Path, dtype) -> np.ndarray:
    sch = blosc2.open(str(fp), mode="r")
    shape = tuple(np.frombuffer(sch.vlmeta["shape"], np.int32).tolist())
    arr   = np.empty(shape, dtype=dtype)
    sch.get_slice(out=arr)
    return arr

def tile_rgb(fp: Path, time_idx: int = 6) -> np.ndarray:
    arr = read_b2frame(fp, np.uint16)  # (T,24,24,11)
    rgb16 = np.take(arr[time_idx], [2,1,0], axis=-1).astype(np.float32)
    return (np.clip(rgb16 / 4000.0, 0, 1) * 255).astype(np.uint8)

def make_lut(remap: Dict[int,int]) -> np.ndarray:
    lut = np.arange(256, dtype=np.uint8)
    for k, v in remap.items():
        lut[int(k) & 0xFF] = int(v) & 0xFF
    return lut

def parse_remap(path: Path | None) -> Dict[int,int]:
    if not path: return {}
    raw = json.loads(path.read_text())
    remap = {}
    for k, v in raw.items():
        remap[int(k)] = int(v["remapped_id"]) if isinstance(v, dict) and "remapped_id" in v else int(v)
    return remap

def tile_label_rgb_mask(fp: Path, keep_ids: Set[int],
                        keep_cmap: Dict[int, Tuple[int,int,int]],
                        remap_lut: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
    lbl = read_b2frame(fp, np.uint8)  # (24,24)
    if remap_lut is not None:
        lbl = remap_lut[lbl]
    present = np.unique(lbl)
    used = {int(x) for x in present if int(x) in keep_ids}
    mask = np.isin(lbl, list(used))
    rgb = np.zeros((*lbl.shape, 3), np.uint8)
    for cid in used:
        rgb[lbl == cid] = keep_cmap[cid]
    return rgb, mask, used

# ───────────────────────── subregion mosaics ─────────────────────────
def build_rgb_mosaic(img_dir: Path, time_idx: int = 6) -> np.ndarray:
    tiles, coords = {}, []
    for p in img_dir.glob("*_img.b2frame"):
        m = PATCH_RE_IMG.match(p.name)
        if m:
            r,c = int(m["row"]), int(m["col"])
            tiles[(r,c)] = p
            coords.append((r,c))
    if not tiles:
        raise FileNotFoundError(f"No *_img.b2frame in {img_dir}")
    max_r = max(r for r,_ in coords); max_c = max(c for _,c in coords)
    mosaic = np.zeros(((max_r + 24), (max_c + 24), 3), np.uint8)
    for (r,c), fp in tiles.items():
        mosaic[r:r+24, c:c+24] = tile_rgb(fp, time_idx=time_idx)
    return mosaic

def build_label_mosaic(label_dir: Path, keep_ids: Set[int],
                       keep_cmap: Dict[int, Tuple[int,int,int]],
                       remap_lut: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
    tiles, coords = {}, []
    for p in label_dir.glob("*_label.b2frame"):
        m = PATCH_RE_LABEL.match(p.name)
        if m:
            r,c = int(m["row"]), int(m["col"])
            tiles[(r,c)] = p
            coords.append((r,c))
    if not tiles:
        raise FileNotFoundError(f"No *_label.b2frame in {label_dir}")
    max_r = max(r for r,_ in coords); max_c = max(c for _,c in coords)
    mosaic = np.zeros(((max_r + 24), (max_c + 24), 3), np.uint8)
    mask   = np.zeros((max_r + 24, (max_c + 24)), np.bool_)
    used_all: Set[int] = set()
    for (r,c), fp in tiles.items():
        rgb_t, m_t, used = tile_label_rgb_mask(fp, keep_ids, keep_cmap, remap_lut)
        mosaic[r:r+24, c:c+24] = rgb_t
        mask[r:r+24, c:c+24]   = m_t
        used_all |= used
    return mosaic, mask, used_all

def overlay_subregion(sub_path: Path, keep_ids: Set[int], keep_map: Dict[int,str],
                      keep_cmap: Dict[int, Tuple[int,int,int]], alpha: float,
                      remap_lut: np.ndarray | None, time_idx: int) -> Tuple[np.ndarray, Set[int]]:
    img_dir   = sub_path / "img"
    label_dir = sub_path / "label"
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing img/ or label/ in {sub_path}")
    #rgb  = build_rgb_mosaic(img_dir, time_idx=time_idx)
    lab, msk, used_ids = build_label_mosaic(label_dir, keep_ids, keep_cmap, remap_lut)
    #if rgb.shape[:2] != lab.shape[:2]:
    #    h,w = lab.shape[:2]
    #    rgb = rgb[:h,:w]
    #blend = rgb.copy().astype(np.float32)
    #blend[msk] = rgb[msk].astype(np.float32)*(1 - alpha) + lab[msk].astype(np.float32)*alpha
    blend = lab
    return blend.astype(np.uint8), used_ids

# ───────────────────────────── legend ────────────────────────────────
def build_legend(ids: List[int], class_names: Dict[int, str],
                 cm: Dict[int, Tuple[int,int,int]], img_h:int, font) -> Image.Image:
    if not ids:
        return Image.new("RGB", (1, img_h), (255,255,255))
    box_h = max(12, img_h // 25); pad = int(box_h * 0.2)
    rows = max(1, (img_h - pad) // (box_h + pad))
    ncol = int(np.ceil(len(ids) / rows))
    def tw(txt: str) -> int:
        try:    return int(font.getlength(txt))
        except: return font.getsize(txt)[0]
    col_w = int(box_h * 1.6 + max(tw(class_names.get(i, str(i))) for i in ids) + 3*pad)
    leg = Image.new("RGB", (ncol * col_w, img_h), (255,255,255))
    draw = ImageDraw.Draw(leg)
    for k, cid in enumerate(ids):
        col, row = k // rows, k % rows
        x0 = col * col_w + pad
        y0 = row * (box_h + pad)
        draw.rectangle([x0, y0, x0 + box_h, y0 + box_h], fill=tuple(cm[cid]))
        draw.text((x0 + box_h + pad, y0 + box_h // 10),
                  class_names.get(cid, str(cid)), fill="black", font=font)
    return leg

# ────────────────────────── list subregions ──────────────────────────
def list_subregions(out_root: Path) -> List[Tuple[int,int,Path]]:
    subs = []
    for p in out_root.iterdir():
        if p.is_dir():
            m = SUBREGION_RE.match(p.name)
            if m:
                subs.append((int(m["X"]), int(m["Y"]), p))
    return sorted(subs)  # raw list; placement uses the orientation rules below

def cache_paths(cache_dir: Path, X: int, Y: int) -> tuple[Path, Path]:
    """Return (png_path, json_meta_path) for a subregion X_Y."""
    return cache_dir / f"{X}_{Y}.png", cache_dir / f"{X}_{Y}.json"

def save_cache(png_path: Path, json_path: Path, arr: np.ndarray, used_ids: set[int]) -> None:
    Image.fromarray(arr).save(png_path)
    meta = {"used_ids": sorted(int(x) for x in used_ids),
            "shape": [int(arr.shape[0]), int(arr.shape[1])]}
    json_path.write_text(json.dumps(meta))

def load_cache(png_path: Path, json_path: Path) -> tuple[np.ndarray, set[int]]:
    arr = np.array(Image.open(png_path).convert("RGB"))
    used: set[int] = set()
    try:
        meta = json.loads(json_path.read_text())
        used = set(int(x) for x in meta.get("used_ids", []))
    except Exception:
        pass
    return arr, used

def add_caption(im: Image.Image, text: str, cap_h: int, font: ImageFont.ImageFont) -> Image.Image:
    """Return a new image with a white caption strip of height cap_h appended below im."""
    w = im.width
    cap = Image.new("RGB", (w, cap_h), (255, 255, 255))
    draw = ImageDraw.Draw(cap)
    # center text
    try:
        tw = font.getlength(text)
        th = font.getbbox(text)[3] - font.getbbox(text)[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    draw.text(((w - tw) // 2, (cap_h - th) // 2), text, fill="black", font=font)
    out = Image.new("RGB", (w, im.height + cap_h), (255, 255, 255))
    out.paste(im, (0, 0))
    out.paste(cap, (0, im.height))
    return out

# ───────────────────────────────── main ──────────────────────────────
def main():
    ap = argparse.ArgumentParser("Overlay all <X_Y> subregions into a grid (crop-only via YAML)")
    ap.add_argument("--out_root", type=Path, required=True, help="preprocessor output root (contains <X_Y>/)")
    ap.add_argument("--yaml",     type=Path, required=True, help="configs/Arkansas/arkansas_data.yaml")
    ap.add_argument("--class_map",type=Path, default=None,  help="optional JSON remap")
    ap.add_argument("--alpha",    type=float, default=0.4,  help="blend opacity on kept classes")
    ap.add_argument("--subsize",  type=int, default=512,    help="resize each subregion overlay to (subsize×subsize)")
    ap.add_argument("--time-idx", type=int, default=6,      help="temporal index into (T,24,24,11) for RGB")
    ap.add_argument("--max-subs", type=int, default=None,   help="limit number of subregions (debugging)")
    ap.add_argument("--out",      type=Path, default="overlay_all.png")
    ap.add_argument("--palette-seed", type=float, default=0.11,
                help="Hue offset in [0,1) to globally rotate the palette")
    ap.add_argument("--gap", type=int, default=12,
                help="Gap (pixels) between subregion tiles in the grid")
    ap.add_argument("--caption-h", type=int, default=28,
                help="Caption strip height (pixels) under each subregion tile (draws 'X_Y')")
    ap.add_argument("--cache-dir", type=Path, default=None,
                help="Directory to store per-subregion overlays (default: <out_root>/.overlays)")
    ap.add_argument("--force", action="store_true",
                help="Recompute and overwrite cached subregions")
    args = ap.parse_args()

    try:
        cap_font = ImageFont.truetype("DejaVuSans.ttf", size=max(int(args.caption_h * 0.7), 10))
    except Exception:
        cap_font = ImageFont.load_default()

    keep_map, keep_ids = load_yaml_classes(args.yaml)
    cmap = gen_palette(keep_ids)
    remap = parse_remap(args.class_map)
    remap_lut = make_lut(remap) if remap else None

    cache_dir = args.cache_dir or (args.out_root / ".overlays")
    cache_dir.mkdir(parents=True, exist_ok=True)

    subs = list_subregions(args.out_root)
    if not subs:
        raise SystemExit(f"No <X_Y> subregion folders found in {args.out_root}")
    if args.max_subs:
        subs = subs[:args.max_subs]

    Xs = [x for x,_,_ in subs]; Ys = [y for _,y,_ in subs]
    minX, maxX = min(Xs), max(Xs)
    minY, maxY = min(Ys), max(Ys)
    ncols = (maxX - minX + 1)
    nrows = (maxY - minY + 1)

    used_all: Set[int] = set()
    tiles: List[Tuple[int,int,Image.Image]] = []

    for X, Y, p in tqdm(subs, desc="subregions"):
        png_path, meta_path = cache_paths(cache_dir, X, Y)

        if png_path.exists() and meta_path.exists() and not args.force:
            # load cached overlay + used class IDs
            try:
                arr, used = load_cache(png_path, meta_path)
            except Exception:
                # if cache is corrupt, rebuild it
                arr, used = overlay_subregion(p, keep_ids, keep_map, cmap, args.alpha, remap_lut, args.time_idx)
                save_cache(png_path, meta_path, arr, used)
        else:
            # compute, then cache
            arr, used = overlay_subregion(p, keep_ids, keep_map, cmap, args.alpha, remap_lut, args.time_idx)
            save_cache(png_path, meta_path, arr, used)

        used_all |= used
        im = Image.fromarray(arr).resize((args.subsize, args.subsize), Image.BILINEAR)
        im = add_caption(im, f"{X}_{Y}", args.caption_h, cap_font)

        # Orientation consistent with ar_cdl_grid.py:
        grid_r = (maxY - Y)         # top row is Y = maxY
        grid_c = (X - minX)         # left→right by X
        tiles.append((grid_r, grid_c, im))

    if not tiles:
        raise SystemExit("No subregion overlays produced.")

    cell_w = cell_h = args.subsize
    grid_w = ncols * cell_w
    grid_h = nrows * cell_h

    # Legend (only classes that actually appear)
    used_sorted = sorted(used_all)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(cell_h // 18, 12))
    except Exception:
        font = ImageFont.load_default()
    legend = build_legend(used_sorted, keep_map, cmap, grid_h, font)

    canvas = Image.new("RGB", (grid_w + legend.width, grid_h), (255,255,255))
    for r, c, im in tiles:
        if 0 <= r < nrows and 0 <= c < ncols:
            canvas.paste(im, (c * cell_w, r * cell_h))
    canvas.paste(legend, (grid_w, 0))
    canvas.save(args.out)
    print(f"✅ wrote {args.out}")
    if used_sorted:
        print("Classes visualized:", [(cid, keep_map.get(cid, str(cid))) for cid in used_sorted])
    else:
        print("⚠️ No crop classes found across subregions.")

if __name__ == "__main__":
    main()

