#!/usr/bin/env python3
import os
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import blosc2
import yaml
from tqdm import tqdm


# ───────────────────── Blosc2 helpers ─────────────────────

def read_b2frame(fp: str, dtype=np.uint8, shape=None) -> np.ndarray:
    """Read a single-chunk Blosc2 frame written like your writer does."""
    schunk = blosc2.open(fp)  # SChunk
    raw = schunk.decompress_chunk(0)
    if shape is None:
        if "shape" not in schunk.vlmeta:
            raise ValueError(f"{fp} missing shape vlmeta")
        shape = tuple(np.frombuffer(schunk.vlmeta["shape"], dtype=np.int32).tolist())
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ───────────────────── YAML helpers ─────────────────────

def load_class_sets(yaml_path: str):
    """
    Returns:
      ignore_ids: set[int] from classes[0]
      keep_map: dict[int,str] from classes[1]
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    classes = cfg.get("classes", {})
    # YAML keys might parse as ints already; coerce just in case.
    def k2i(d):
        return {int(k): v for k, v in d.items()} if isinstance(d, dict) else {}
    ignore_ids = set(k2i(classes.get(0, {})).keys())
    keep_map   = {int(k): str(v) for k, v in k2i(classes.get(1, {})).items()}
    return ignore_ids, keep_map


# ───────────────────── Core computation ─────────────────────

def scan_label_files(root: str) -> list[str]:
    # out_root/<subregion>/label/*.b2frame
    return sorted(glob(os.path.join(root, "*", "label", "*.b2frame")))

def class_hist_from_file(fp: str) -> np.ndarray:
    lab = read_b2frame(fp, dtype=np.uint8)  # (24,24)
    return np.bincount(lab.ravel(), minlength=256).astype(np.int64)

def compute_distributions(root: str, ignore_ids: set[int], keep_ids: set[int], per_subregion: bool):
    files = scan_label_files(root)
    if not files:
        raise FileNotFoundError(f"No label frames found under {root}")

    global_hist = np.zeros(256, dtype=np.int64)
    per_region_hist: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(256, dtype=np.int64))

    for fp in tqdm(files, desc="labels"):
        h = class_hist_from_file(fp)
        # Remove ignores right away to keep memory paths clean
        if ignore_ids:
            h[list(ignore_ids)] = 0
        # Zero out everything *not* in keep_ids (so we only accumulate crop classes)
        not_keep = np.ones(256, dtype=bool)
        not_keep[list(keep_ids)] = False
        h[not_keep] = 0

        global_hist += h
        if per_subregion:
            subregion = fp.split(os.sep)[-3]
            per_region_hist[subregion] += h

    return global_hist, (per_region_hist if per_subregion else None)

def hist_to_df_keeponly(hist: np.ndarray, keep_map: dict[int, str]) -> pd.DataFrame:
    # Keep only classes that exist in the map and have count > 0
    ids = np.array(sorted(keep_map.keys()), dtype=int)
    counts = hist[ids]
    total = int(counts.sum())
    df = pd.DataFrame({
        "class_id": ids,
        "class_name": [keep_map[i] for i in ids],
        "count": counts,
        "pct": (counts / total * 100.0) if total > 0 else np.zeros_like(counts, dtype=float),
    })
    df = df[df["count"] > 0].sort_values("count", ascending=False).reset_index(drop=True)
    return df


# ───────────────────── CLI ─────────────────────

def main():
    p = argparse.ArgumentParser("Compute crop-class distributions using arkansas_data.yaml")
    p.add_argument("--data", required=True, help="out_root produced by your preprocessor")
    p.add_argument("--yaml", required=True, help="configs/Arkansas/arkansas_data.yaml")
    p.add_argument("--per-subregion", action="store_true", help="also emit per-subregion CSVs")
    p.add_argument("--out", default="crop_distribution.csv", help="CSV for global crop distribution")
    p.add_argument("--out-dir", default="crop_distribution_by_region", help="dir for per-subregion CSVs")
    args = p.parse_args()

    ignore_ids, keep_map = load_class_sets(args.yaml)
    keep_ids = set(keep_map.keys())

    # Sanity: warn if an id appears in both sets (config error)
    overlap = ignore_ids & keep_ids
    if overlap:
        print(f"[warn] IDs present in both ignore and keep sets: {sorted(overlap)}")

    global_hist, per_region = compute_distributions(
        root=args.data,
        ignore_ids=ignore_ids,
        keep_ids=keep_ids,
        per_subregion=args.per_subregion,
    )

    # Save global (only keep classes[1])
    df_global = hist_to_df_keeponly(global_hist, keep_map)
    df_global.to_csv(args.out, index=False)
    print(f"[saved] global crop distribution → {args.out}")
    print(df_global.head(20).to_string(index=False))

    # Save per-subregion
    if per_region:
        os.makedirs(args.out_dir, exist_ok=True)
        for region, hist in per_region.items():
            df = hist_to_df_keeponly(hist, keep_map)
            fp = os.path.join(args.out_dir, f"{region}.csv")
            df.to_csv(fp, index=False)
            print(f"[saved] {region} → {fp}\n{df.head(5).to_string(index=False)}\n")


if __name__ == "__main__":
    main()

