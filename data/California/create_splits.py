#!/usr/bin/env python3
import os, json, random
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import blosc2
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from functools import partial
import multiprocessing as mp

# -------------------------------------------------------------------- #
#  Constants                                                           #
# -------------------------------------------------------------------- #
REMOVED_CLASSES        = [1, 2, 4, 23]
SEASONAL_MAJOR_CLASSES = [12, 13, 3, 5, 51, 14, 37, 15, 16, 44]
SEASONAL_MINOR_CLASSES = [27, 38, 20, 33, 31, 40, 49, 17, 29,
                          46, 54, 41, 42]
ALL_CLASSES = SEASONAL_MAJOR_CLASSES + SEASONAL_MINOR_CLASSES

class_names = {  # abbreviated for brevity
    1:"Unknown", 12:"Corn/Sorghum", 13:"Alfalfa", 3:"Mixed pasture",
    5:"Misc grain", 51:"Rice", 14:"Wheat", 37:"Tomato", 15:"Misc grasses",
    16:"Native pasture", 44:"Cotton", 27:"Leafy greens", 38:"Onions",
    20:"Melons", 33:"Safflower", 31:"Strawberries", 40:"Carrots",
    49:"Sunflower", 17:"Potatoes", 29:"Bush berries", 46:"Sweet potatoes",
    54:"Sugar beets", 41:"Dry beans", 42:"Peppers",
}

IDX_LOOKUP = {cid: i + 1 for i, cid in enumerate(ALL_CLASSES)}


# -------------------------------------------------------------------- #
#  Remapping helpers                                                   #
# -------------------------------------------------------------------- #
def load_remap_table(fp: str) -> dict[int, int]:
    """Original CDL code → remapped code (as in your JSON file)."""
    with open(fp) as fh:
        raw = json.load(fh)
    return {int(k): int(v["remapped_id"]) for k, v in raw.items()}

def remap_first(arr: np.ndarray, table: dict[int, int]) -> np.ndarray:
    """Apply class_map (unknown→0)."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for orig, new in table.items():
        out[arr == orig] = new
    return out

def remap_second(arr: np.ndarray) -> np.ndarray:
    """Compact to 0…N using IDX_LOOKUP."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for cid, idx in IDX_LOOKUP.items():
        out[arr == cid] = idx
    return out


# -------------------------------------------------------------------- #
#  I/O helper                                                          #
# -------------------------------------------------------------------- #
def read_b2frame(fp: str, dtype=np.uint8) -> np.ndarray:
    sch   = blosc2.open(fp, mode="r")
    shape = np.frombuffer(sch.vlmeta["shape"], np.int32)
    arr   = np.empty(shape, dtype=dtype)
    sch.get_slice(out=arr)
    return arr

# Pick the best shuffle filter constant that exists in *this* py‑blosc2 build.
try:
    # Newer API (v2.0+)
    _BITSHUFF = blosc2.Filter.BITSHUFFLE  # type: ignore[attr-defined]
    _SHUFF = blosc2.Filter.SHUFFLE        # type: ignore[attr-defined]
except AttributeError:  # Older builds: top‑level ints
    _BITSHUFF = getattr(blosc2, "BITSHUFFLE", None)
    _SHUFF = getattr(blosc2, "SHUFFLE", None)

if _BITSHUFF is not None:
    _FILTERS = [_BITSHUFF]
elif _SHUFF is not None:
    warnings.warn("BITSHUFFLE unavailable – falling back to byte‑shuffle.")
    _FILTERS = [_SHUFF]
else:
    warnings.warn("No Blosc shuffle filters found – proceeding unshuffled.")
    _FILTERS = []

def make_storage(fp: str) -> blosc2.Storage:
    """Return a Storage object that writes a contiguous frame at *fp*."""
    return blosc2.Storage(contiguous=True, urlpath=fp, mode="w")

def make_cparams(dtype: np.dtype) -> blosc2.CParams:
    return blosc2.CParams(
        codec=blosc2.Codec.ZSTD,
        clevel=5,
        filters=_FILTERS,
        typesize=dtype.itemsize,
    )

def write_b2frame(arr: np.ndarray, fp: str):
    """Write *arr* to *fp* as a contiguous Blosc2 frame (tutorial style)."""
    # 1) Configure compression params & storage
    cparams = make_cparams(arr.dtype)
    storage = make_storage(fp)

    # 2) Create SChunk on disk
    schunk = blosc2.SChunk(chunksize=arr.nbytes, cparams=cparams, storage=storage)

    # 3) Append *one* chunk (the whole sample)
    schunk.append_data(arr.tobytes())

    # 4) Persist shape metadata so readers know how to reshape
    schunk.vlmeta["shape"] = np.asarray(arr.shape, dtype=np.int32).tobytes()

    # No explicit close needed; object going out of scope flushes buffers.


# -------------------------------------------------------------------- #
#  1. find & filter tiles                                              #
# -------------------------------------------------------------------- #
def collect_kept_tiles(label_root: str, remap_tbl: dict[int, int]):
    kept = []  # list of (meta_patch, tile_id, path_to_label)
    label_files = sorted(glob(os.path.join(label_root, "*", "label", "*_label.b2frame")))

    for fp in tqdm(label_files, desc="Filtering tiles"):
        arr = read_b2frame(fp)                  # uint8
        arr = remap_first(arr, remap_tbl)       # FIRST remapping

        unique, counts = np.unique(arr, return_counts=True)
        counts = {k:v for k,v in zip(unique, counts) if k not in REMOVED_CLASSES}
        if not counts:
            continue

        seasonal_pixels = sum(v for k,v in counts.items() if k in ALL_CLASSES)
        if seasonal_pixels / arr.size <= 0.5:    # reject
            continue

        xy_patch = Path(fp).parent.parent.name          # X_Y
        tile_id  = Path(fp).stem.replace("_label","")   # iy_ix
        kept.append((xy_patch, tile_id, fp))

    return kept

# -------------------------------------------------------------------- #
#  2. split meta-patches 80/20                                         #
# -------------------------------------------------------------------- #
def split_meta_patches(kept_tiles, train_ratio=.8, seed=42):
    patches = sorted({xy for xy,_,_ in kept_tiles})
    rnd = random.Random(seed); rnd.shuffle(patches)

    n_train = int(len(patches) * train_ratio)
    train_set = set(patches[:n_train])
    val_set   = set(patches[n_train:])

    rows, lookup = [], {}
    for xy, tile, path in kept_tiles:
        split = "train" if xy in train_set else "val"
        rows.append({"meta_patch":xy, "tile_id":tile, "split":split})
        lookup[path] = split
    return pd.DataFrame(rows), lookup

# -------------------------------------------------------------------- #
#  3. pixel tallies per split                                           #
# -------------------------------------------------------------------- #
def pixel_counts_by_split(df: pd.DataFrame, label_root: str = "/mnt/ssd_8tb/CA_2022_bframe2"):
    counts = { "train": defaultdict(int), "val": defaultdict(int) }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pixel count"):
        fp = Path(label_root) / row["meta_patch"] / "label_remap" / f"{row['tile_id']}_label.b2frame"
        arr = read_b2frame(fp)
        u, c = np.unique(arr, return_counts=True)
        for cid, cnt in zip(u, c):
            counts[row["split"]][int(cid)] += int(cnt)
    return counts

# -------------------------------------------------------------------- #
#  4. bar-chart                                                         #
# -------------------------------------------------------------------- #
def plot_distribution(counts, out_png: str | None = None):
    """
    Bars:  Other | 12-Corn | 13-Alfalfa | …   (% of pixels per split)
    """
    all_classes = ["Others"] + [class_names.get(i, "?") for i in ALL_CLASSES]

    # total pixels in each split
    tot_train = sum(counts["train"].values()) or 1
    tot_val   = sum(counts["val"].values())   or 1

    def pct_list(split_totals, total_pixels):
        # convert to percentage
        lst = [split_totals.get(i, 0) for i in range(len(all_classes))]
        return [v * 100 / total_pixels for v in lst]

    train_pct = pct_list(counts["train"], tot_train)
    val_pct   = pct_list(counts["val"],   tot_val)

    # ── plotting ──
    idx, width = np.arange(len(all_classes)), 0.4
    plt.figure(figsize=(15, 6))
    plt.bar(idx - width/2, train_pct, width, label="train")
    plt.bar(idx + width/2, val_pct,   width, label="val")

    xlabels = all_classes
    plt.xticks(idx, xlabels, rotation=90)
    plt.ylabel("% of pixels within split")
    plt.title("Class distribution (percentage share, with Other)")
    plt.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"📊 percentage plot saved → {out_png}")

    plt.close()


# -------------------------------------------------------------------- #
#  Main                                                                
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    label_root = "/mnt/ssd_8tb/CA_2022_bframe2"
    out_csv    = "tiles_train_val.csv"
    class_map = "classnames_indent.json"

    remap_tbl = load_remap_table(class_map)
    all_label_files = sorted(
        glob(os.path.join(label_root, "*", "label", "*_label.b2frame"))
    )

    def _init_pool(shared_tbl):
        """Runs **once** in every worker → store table in global var (avoids pickling)."""
        global REMAP_TBL
        REMAP_TBL = shared_tbl          # each worker gets read‑only copy

    def _remap_and_save(fp: str, label_root: str):
        """Worker: read → remap twice → write sibling label_remapped/ file."""
        arr = read_b2frame(fp)                      # original labels (uint8)
        arr = remap_second(remap_first(arr, REMAP_TBL))

        meta_patch_dir = Path(fp).parent.parent     # …/10_15
        out_dir = meta_patch_dir / "label_remap"
        out_dir.mkdir(exist_ok=True)

        out_fp = out_dir / Path(fp).name            # same filename
        write_b2frame(arr, str(out_fp))
        return fp                                   # for progress bar
    
    print(f"🔄  Writing remapped labels ({len(all_label_files):,} files)…")
    n_workers = os.cpu_count()

    with mp.Pool(n_workers, initializer=_init_pool, initargs=(remap_tbl,)) as pool:
        for _ in tqdm(pool.imap_unordered(
                partial(_remap_and_save, label_root=label_root),
                all_label_files),
                total=len(all_label_files),
                desc="label_remapped"):
            pass
    print("✅  All remapped labels written.\n")

    kept_tiles = collect_kept_tiles(label_root, remap_tbl)
    df, _ = split_meta_patches(kept_tiles)

    df.to_csv(out_csv, index=False)
    print(f"✅  Saved {len(df)} rows to {out_csv}")

    # print high-level stats
    df = pd.read_csv(out_csv)  # load existing CSV if available
    total_tiles = len(glob(os.path.join(label_root,"*","label","*_label.b2frame")))
    print(f"\nKept tiles {len(df)} / {total_tiles} = {len(df)/total_tiles:.2%}")
    kept_patches = df["meta_patch"].nunique()
    total_patches = len([d for d in Path(label_root).iterdir() if (d/"label").exists()])
    print(f"Kept patches {kept_patches} / {total_patches} = {kept_patches/total_patches:.2%}")

    # pixel distribution & plot
    pixel_counts = pixel_counts_by_split(df, label_root)
    plot_distribution(pixel_counts, 'pixel_class_distribution.png')
