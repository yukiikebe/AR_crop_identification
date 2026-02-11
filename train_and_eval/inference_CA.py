import os, sys, argparse, random, json, yaml
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('tkagg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
import rasterio
import blosc2


# ---------------- project‑specific utilities ----------------
sys.path.insert(0, os.getcwd())     # local packages
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint
from data.California.dataloader import get_dataloader as get_california_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
# ------------------------------------------------------------

# ╭─────────────────── COMPACT‑LABEL DEFINITIONS ───────────────────╮
ALL_CLASSES = [
    12, 13, 3, 5, 51, 14, 37, 15, 16, 44,         # seasonal‑major (10)
    27, 38, 20, 33, 31, 40, 49, 17, 29, 46, 54, 41, 42  # seasonal‑minor (13)
]
CLASS_NAMES = ["Other"] + [
    "Corn/Sorghum", "Alfalfa", "Mixed pasture", "Misc grain", "Rice", "Wheat",
    "Tomato", "Misc grasses", "Native pasture", "Cotton",
    "Leafy greens", "Onions & garlic", "Melons", "Safflower",
    "Strawberries", "Carrots", "Sunflower", "Potatoes", "Bush berries",
    "Sweet potatoes", "Sugar beets", "Dry beans", "Peppers"
]

# deterministic RGB palette (uint8)
PALETTE = np.array([
    (0,   0,   0 ),   # 0 Other
    (255, 0,   0 ),   # 1 Corn/Sorghum
    (0,   255, 0 ),   # 2 Alfalfa
    (0,   0,   255),  # 3 Mixed pasture
    (255, 255, 0 ),   # 4 Misc grain
    (255, 165, 0 ),   # 5 Rice
    (128, 0,   128),  # 6 Wheat
    (0,   255, 255),  # 7 Tomato
    (128, 128, 0 ),   # 8 Misc grasses
    (255, 192, 203),  # 9 Native pasture
    (0,   128, 128),  # 10 Cotton
    (128, 0,   0 ),   # 11 Leafy greens
    (0,   128, 0 ),   # 12 Onions & garlic
    (0,   0,   128),  # 13 Melons
    (192, 192, 192),  # 14 Safflower
    (255, 20, 147),   # 15 Strawberries
    (255, 140, 0 ),   # 16 Carrots
    (255, 215, 0 ),   # 17 Sunflower
    (139, 69,  19),   # 18 Potatoes
    (75,  0,   130),  # 19 Bush berries
    (210, 105, 30),   # 20 Sweet potatoes
    (0,   100, 0 ),   # 21 Sugar beets
    (70,  130, 180),  # 22 Dry beans
    (220, 20,  60)    # 23 Peppers
], dtype=np.uint8)
NUM_CLASSES = 26      # 24

def read_b2frame(fp: str) -> np.ndarray:
    """
    Generic Blosc-2 reader that respects the stored shape.
    `dtype` must match how the frame was written.
    """
    sch     = blosc2.open(fp, mode="r")
    shape   = np.frombuffer(sch.vlmeta["shape"], dtype=np.int32)
    out_arr = np.empty(shape, dtype=np.uint8)
    sch.get_slice(out=out_arr)
    return out_arr

# ────────────────────── CDL mosaic helper ──────────────────────
def build_cdl_mosaic(meta_dir: str, tile_size: int = 24) -> np.ndarray:
    """
    Reconstruct the compact‑index CDL mosaic for *meta_dir* from the
    label_remap/*.b2frame tiles that were written by create_splits.py.
    Returns H×W uint8 image (values 0…23).
    """
    label_paths = sorted(glob(os.path.join(meta_dir, "label_remap", "*_label.b2frame")))
    if not label_paths:
        raise FileNotFoundError("No remapped labels in " + meta_dir)

    patch_gt = {}
    for fp in label_paths:
        tile_id = Path(fp).stem.replace("_label", "")      # "48_72"
        y, x = map(int, tile_id.split('_'))
        arr = read_b2frame(fp)                             # uint8, already compact
        patch_gt[(y, x)] = arr

    max_y = max(y for y, _ in patch_gt) + tile_size
    max_x = max(x for _, x in patch_gt) + tile_size
    mosaic = np.zeros((max_y, max_x), dtype=np.uint8)

    for (y, x), tile in patch_gt.items():
        mosaic[y:y + tile_size, x:x + tile_size] = tile
    return mosaic


# ------------------ visualisation helpers ---------------------
def label_to_rgb(label_img: np.ndarray) -> np.ndarray:
    label_img = np.where(label_img < len(PALETTE), label_img, 0)
    return PALETTE[label_img]

def draw_legend(ax):
    for idx, name in enumerate(CLASS_NAMES):
        ax.add_patch(plt.Rectangle((0, NUM_CLASSES - idx - 1), 1.2, 1.2,
                                   color=PALETTE[idx]/255.0))
        ax.text(2, NUM_CLASSES - idx - .5, name,
                va='center', ha='left', fontsize=10)
    ax.set_ylim(0, NUM_CLASSES); ax.set_xlim(0, 3); ax.axis('off')

def save_visual(label_img: np.ndarray, out_fp: str):
    rgb = label_to_rgb(label_img)
    fig, ax = plt.subplots(1, 2, figsize=(10, 12),
                           gridspec_kw={'width_ratios':[1,1]})
    ax[0].imshow(rgb); ax[0].axis('off'); ax[0].set_title("Prediction")
    draw_legend(ax[1]); ax[1].set_title("Legend")
    plt.tight_layout(); fig.savefig(out_fp, dpi=200); plt.close(fig)
    print("🖼  saved", out_fp)
# --------------------------------------------------------------


def stitch_patches(net, dataloader, device, tile_size=24):
    net.eval(); patch_out = {}
    with torch.no_grad():
        for batch, paths in tqdm(dataloader, desc="inference"):
            preds = net(batch['inputs'].to(device)).argmax(1).cpu().numpy()
            for i, fp in enumerate(paths):
                patch_id = Path(fp).stem.replace("_img", "")  # e.g. "48_72"
                y, x = map(int, patch_id.split('_'))      # "iy_ix"
                patch_out[(y, x)] = preds[i]

    max_y = max(y for y,_ in patch_out)+tile_size
    max_x = max(x for _,x in patch_out)+tile_size
    mosaic = np.zeros((max_y, max_x), dtype=np.uint8)
    for (y,x), tile in patch_out.items():
        mosaic[y:y+tile_size, x:x+tile_size] = tile
    return mosaic


def run_inference(cfg_fp: str, input_dir: str, sub_region: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_png = f"{out_dir}/{sub_region}.png"
    if os.path.exists(out_png):
        print("▶  exists:", out_png); return

    cfg = read_yaml(cfg_fp)
    device = get_device(cfg['DEVICE']['device_id'], allow_cpu=False)
    cfg['MODEL']['num_classes'] = NUM_CLASSES

    meta_dir = f"{input_dir}/{sub_region}"
    img_paths = sorted(glob(os.path.join(meta_dir, "img", "*_img.b2frame")))
    if not img_paths:
        raise FileNotFoundError("No *_img.b2frame under " + meta_dir)

    records = []
    for fp in img_paths:
        tile_id = Path(fp).stem.replace("_img", "")   # e.g. "48_72"
        records.append({
            "meta_patch": sub_region,   # same field names as create_splits.csv
            "tile_id":    tile_id,
            "split":      "test"        # dummy value; not used by dataloader
        })

    eval_df = pd.DataFrame(records)      # ↓ dataloader consumes this DataFrame
    eval_dataloader = get_california_dataloader(
        eval_df,                         # <-- instead of “paths=…”
        root_dir=input_dir,              # top-level CA_2022_bframe2
        split='test',
        transform=PASTIS_segmentation_transform(cfg['MODEL'], is_training=False),
        batch_size=cfg['DATASETS']['eval']['batch_size'],
        shuffle=False,
        num_workers=cfg['DATASETS']['eval']['num_workers'],
        return_paths=True,  # return tile IDs
    )

    net = get_model(cfg, device)
    ckpt = cfg['CHECKPOINT']['load_from_checkpoint']
    if ckpt: load_from_checkpoint(net, ckpt, partial_restore=False, device=device)

    mosaic = stitch_patches(net, eval_dataloader, device)
    # -------- ground‑truth CDL (compact indices) -----------
    cdl_mosaic = build_cdl_mosaic(meta_dir)

    # -------- side‑by‑side visual --------------------------
    rgb_pred = label_to_rgb(mosaic)
    rgb_cdl  = label_to_rgb(cdl_mosaic)

    fig, ax = plt.subplots(1, 3, figsize=(15, 6),
                           gridspec_kw={'width_ratios':[1,1,1]})
    ax[0].imshow(rgb_pred); ax[0].axis('off'); ax[0].set_title("Prediction")
    ax[1].imshow(rgb_cdl);  ax[1].axis('off'); ax[1].set_title("CDL ground‑truth")
    draw_legend(ax[2]); ax[2].set_title("Legend")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
    print("🖼  saved", out_png)


# -------------------------- CLI ---------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--input_dir",   required=True)
    ap.add_argument("--sub_region",  required=True)
    ap.add_argument("--output_dir",  default="./output")
    args = ap.parse_args()

    run_inference(args.config, args.input_dir, args.sub_region, args.output_dir)
