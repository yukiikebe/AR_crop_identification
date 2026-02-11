#!/usr/bin/env python3
# ar_split_from_yaml.py — Arkansas: crop-only splits from YAML (optional single remap)

import os, json, random, argparse, warnings
from glob import glob
from pathlib import Path
from collections import defaultdict
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import blosc2
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

# ─────────────────────────── I/O helpers ────────────────────────────
def read_b2frame(fp: str | Path, dtype=np.uint8) -> np.ndarray:
    sch   = blosc2.open(str(fp), mode="r")
    shape = np.frombuffer(sch.vlmeta["shape"], np.int32)
    arr   = np.empty(shape, dtype=dtype)
    sch.get_slice(out=arr)
    return arr

def write_b2frame(arr: np.ndarray, fp: str | Path):
    # one-shot contiguous frame writer
    cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, typesize=arr.dtype.itemsize)
    storage = blosc2.Storage(contiguous=True, urlpath=str(fp), mode="w")
    schunk  = blosc2.SChunk(chunksize=arr.nbytes, cparams=cparams, storage=storage)
    schunk.append_data(arr.tobytes())
    schunk.vlmeta["shape"] = np.asarray(arr.shape, dtype=np.int32).tobytes()

# ─────────────────────────── Config / LUTs ───────────────────────────
def load_yaml_classes(yaml_path: Path) -> tuple[dict[int, str], set[int]]:
    """
    Expecting arkansas_data.yaml with a `classes` section.
    We read **classes[1]** = crop-only.
    """
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    classes = cfg.get("classes", {})
    crop_map_raw = classes.get(3, {})  # crop-only bucket
    crop_map = {int(k): str(v) for k, v in crop_map_raw.items()}
    crop_ids = set(crop_map.keys())
    if not crop_ids:
        raise ValueError("No crop classes found in YAML under classes[1].")
    return crop_map, crop_ids

def parse_remap(remap_json: Path | None) -> dict[int, int]:
    """
    Optional **single** remap table: original_id -> new_id.
    If absent, labels are used as-is.
    """
    if remap_json is None:
        return {}
    raw = json.loads(Path(remap_json).read_text())
    out: dict[int, int] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "remapped_id" in v:
            out[int(k)] = int(v["remapped_id"])
        else:
            out[int(k)] = int(v)
    return out

def make_lut(remap: dict[int, int]) -> np.ndarray | None:
    if not remap:
        return None
    lut = np.arange(256, dtype=np.uint8)
    for k, v in remap.items():
        lut[int(k) & 0xFF] = int(v) & 0xFF
    return lut

def make_compaction_lut(crop_ids: set[int], crop_map: dict[int, str]) -> tuple[np.ndarray, dict[int, str], dict[int, int]]:
    """Return (lut, id_to_name_after, fwd_old_to_new) mapping crop IDs → 1..N and background → 0."""
    lut = np.zeros(256, dtype=np.uint8)  # default background 0
    id_to_name_after: dict[int, str] = {}
    fwd: dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(crop_ids), start=1):
        lut[old_id & 0xFF] = np.uint8(new_id)
        id_to_name_after[new_id] = crop_map.get(old_id, str(old_id))
        fwd[old_id] = new_id
    return lut, id_to_name_after, fwd

# ─────────────────────────── Discovery ───────────────────────────────
def find_label_files(out_root: Path) -> list[str]:
    # Expect structure: <out_root>/<X_Y>/label/*_label.b2frame
    return sorted(glob(str(out_root / "*" / "label" / "*_label.b2frame")))

# ────────────────────────── Filter / keep ────────────────────────────
def collect_kept_tiles(label_files: list[str],
                       id_filter: set[int],
                       remap_lut: np.ndarray | None,
                       min_crop_ratio: float) -> list[tuple[str, str, str]]:
    """
    Return list of (meta_patch, tile_id, path_to_label) that pass crop threshold.
    - Keep a tile if (crop-pixels / total) >= min_crop_ratio
    - If min_crop_ratio == 0, keep any tile that has at least one crop pixel.
    """
    kept: list[tuple[str, str, str]] = []
    for fp in tqdm(label_files, desc="Filtering tiles"):
        arr = read_b2frame(fp, np.uint8)
        if remap_lut is not None:
            arr = remap_lut[arr]
        u, c = np.unique(arr, return_counts=True)
        counts = {int(k): int(v) for k, v in zip(u, c)}
        total = arr.size
        crop_pix = sum(v for k, v in counts.items() if k in id_filter)

        keep = (crop_idx > 0) if (min_crop_ratio <= 0.0) else ((crop_pix / total) >= min_crop_ratio)
        if not keep:
            continue

        meta_patch = Path(fp).parent.parent.name  # "<X>_<Y>"
        tile_id    = Path(fp).stem.replace("_label", "")  # "<row>_<col>"
        kept.append((meta_patch, tile_id, fp))
    return kept

# ────────────────────────── Split meta-patches ───────────────────────
def split_meta_patches(kept_tiles, train_ratio=0.8, seed=42):
    patches = sorted({xy for xy, _, _ in kept_tiles})
    rnd = random.Random(seed); rnd.shuffle(patches)
    n_train = int(len(patches) * train_ratio)
    train_set = set(patches[:n_train]); val_set = set(patches[n_train:])
    rows, lookup = [], {}
    for xy, tile, path in kept_tiles:
        split = "train" if xy in train_set else "val"
        rows.append({"meta_patch": xy, "tile_id": tile, "split": split})
        lookup[path] = split
    return pd.DataFrame(rows), lookup

# ────────────────────────── Pixel tallies ────────────────────────────
def pixel_counts_by_split(df: pd.DataFrame,
                          out_root: Path,
                          id_filter: set[int],
                          remap_lut: np.ndarray | None) -> dict[str, dict[int, int]]:
    counts = {"train": defaultdict(int), "val": defaultdict(int)}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pixel count"):
        fp = out_root / row["meta_patch"] / "label_remap" / f"{row['tile_id']}_label.b2frame"
        arr = read_b2frame(fp, np.uint8)
        if remap_lut is not None:
            arr = remap_lut[arr]
        u, c = np.unique(arr, return_counts=True)
        for cid, cnt in zip(u, c):
            if int(cid) in id_filter:
                counts[row["split"]][int(cid)] += int(cnt)
    return counts

# ────────────────────────── Plot distribution ────────────────────────
def plot_distribution(counts: dict[str, dict[int, int]],
                      id_to_name: dict[int, str],
                      out_png: str | None = None):
    ids_sorted = sorted(id_to_name.keys())
    tot_train = sum(counts["train"].values()) or 1
    tot_val   = sum(counts["val"].values()) or 1

    def pct(bucket, tot): return [bucket.get(i, 0) * 100.0 / tot for i in ids_sorted]
    train_pct = pct(counts["train"], tot_train)
    val_pct   = pct(counts["val"],   tot_val)

    idx, width = np.arange(len(ids_sorted)), 0.45
    plt.figure(figsize=(max(12, len(idx) * 0.35), 6))
    plt.bar(idx - width/2, train_pct, width, label="train")
    plt.bar(idx + width/2, val_pct,   width, label="val")
    xlabels = [f"{i}:{id_to_name.get(i, str(i))}" for i in ids_sorted]
    plt.xticks(idx, xlabels, rotation=90)
    plt.ylabel("% of crop pixels within split")
    plt.title("Arkansas crop-only class distribution (train vs val)")
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"📊 percentage plot saved → {out_png}")
    plt.close()

def _init_pool(shared_lut):
    global LUT
    LUT = shared_lut

def _remap_and_save(fp: str):
    arr = read_b2frame(fp, np.uint8)
    arr = LUT[arr]
    out_dir = Path(fp).parent.parent / "label_remap"
    out_dir.mkdir(exist_ok=True)
    out_fp = out_dir / Path(fp).name
    write_b2frame(arr, out_fp)
    return True

# ───────────────────────────── Main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Arkansas: create train/val splits (crop-only) from YAML")
    ap.add_argument("--out_root", type=Path, required=True,
                    help="Root with <X_Y>/label/*_label.b2frame")
    ap.add_argument("--yaml", type=Path, required=True,
                    help="Arkansas class YAML (uses classes[1] = crop-only)")
    ap.add_argument("--class_map", type=Path, default=None,
                    help="Optional single remap JSON (orig_id->new_id or {remapped_id:...})")
    ap.add_argument("--out_csv", type=Path, default=Path("tiles_train_val_AR.csv"))
    ap.add_argument("--plot_png", type=Path, default=Path("pixel_class_distribution_AR.png"))
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_crop_ratio", type=float, default=0.0,
                    help="Min fraction of crop pixels to keep a tile (0 keeps any tile with ≥1 crop pixel)")
    ap.add_argument("--write_remap", action="store_true",
                    help="If --class_map is given, also write remapped labels to <X_Y>/label_remap/")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    crop_map, crop_ids = load_yaml_classes(args.yaml)
    remap_tbl = parse_remap(args.class_map)

    if remap_tbl:
        remap_lut = make_lut(remap_tbl)
        id_to_name = {}
        for old_id, name in crop_map.items():
            new_id = remap_tbl.get(old_id, old_id)
            if new_id != 0:
                id_to_name[int(new_id)] = name
        use_lut = remap_lut
        id_filter = set(id_to_name.keys())
    else:
        compact_lut, id_to_name, fwd = make_compaction_lut(crop_ids, crop_map)
        use_lut = compact_lut
        id_filter = set(id_to_name.keys())

    label_files = find_label_files(args.out_root)
    if not label_files:
        raise SystemExit(f"No *_label.b2frame under {args.out_root}/<X_Y>/label/")

    # Optional: write a single-remap copy to label_remap/ for faster re-use
    if args.write_remap:
        print(f"🔄  Writing remapped labels ({len(label_files):,} files)…")
        with mp.Pool(args.workers, initializer=_init_pool, initargs=(use_lut,)) as pool:
            for _ in tqdm(pool.imap_unordered(_remap_and_save, label_files, chunksize=64),
                          total=len(label_files), desc="label_remap"):
                pass
        print("✅  All remapped labels written.\n")
        label_files = sorted(glob(str(args.out_root / "*" / "label_remap" / "*_label.b2frame")))
        runtime_lut = None
    else:
        runtime_lut = use_lut

    kept_tiles = collect_kept_tiles(label_files, id_filter, runtime_lut, args.min_crop_ratio)
    if not kept_tiles:
        warnings.warn("No tiles met the crop threshold. Try lowering --min_crop_ratio.")
        kept_tiles = []

    df, _ = split_meta_patches(kept_tiles, train_ratio=args.train_ratio, seed=args.seed)
    df.to_csv(args.out_csv, index=False)
    print(f"✅  Saved {len(df)} rows to {args.out_csv}")
    #df = pd.read_csv(args.out_csv)

    # High-level stats
    total_tiles = len(find_label_files(args.out_root))
    print(f"\nKept tiles {len(df)} / {total_tiles} = {len(df)/max(total_tiles,1):.2%}")
    kept_patches = df["meta_patch"].nunique() if len(df) else 0
    total_patches = len([d for d in Path(args.out_root).iterdir() if (d / "label").exists()])
    print(f"Kept patches {kept_patches} / {total_patches} = {kept_patches/max(total_patches,1):.2%}")

    # Pixel distribution (crop-only) & plot
    if len(df):
        pixel_counts = pixel_counts_by_split(df, args.out_root, id_filter, runtime_lut)
        plot_distribution(pixel_counts, id_to_name, str(args.plot_png))

        # Also dump per-class counts to CSV for imbalance inspection
        recs = []
        for split in ("train", "val"):
            for cid, cnt in pixel_counts[split].items():
                recs.append({"split": split, "class_id": cid,
                             "class_name": id_to_name.get(cid, str(cid)), "pixels": cnt})
        pd.DataFrame(recs).sort_values(["split", "pixels"], ascending=[True, False]) \
            .to_csv("per_class_pixel_counts_AR.csv", index=False)
        print("🧮 wrote per_class_pixel_counts_AR.csv")
    else:
        print("⚠️ No kept tiles → skipped pixel distribution.")

if __name__ == "__main__":
    main()

