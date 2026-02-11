import os
from glob import glob
from datetime import datetime
import multiprocessing as mp
from functools import partial
import warnings

import rasterio
import numpy as np
import cv2
import pandas as pd
import blosc2
from tqdm import tqdm

# ───────────────────────── CONFIGURATION ──────────────────────────

sample_requirements: dict[int, int] = {
    1: 1,  2: 1,  3: 1,
    4: 2,  5: 2,  6: 2,
    7: 2,  8: 2,  9: 2,
    10: 1, 11: 1, 12: 1,
}

selected_bands: dict[str, list[str]] = {
    "10m": ["B2", "B3", "B4", "B8"],
    "20m": ["B5", "B6", "B7", "B8A", "B11", "B12"],
    "SCL": ["SCL"],
}

band_coord: list[str] = [b for _, bs in selected_bands.items() for b in bs]

# ─────────────────── BLOSC2 PARAMETER HELPERS ────────────────────

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

# Convenience factory so we don’t repeat ourselves

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

# ─────────────────────────── HELPERS ─────────────────────────────

def resample_dates(dates: list[str]) -> list[str]:
    if not dates:
        return []
    df = pd.DataFrame(sorted(dates), columns=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(inplace=True)
    if df.empty:
        return []
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    keep: list[str] = []
    for m, grp in df.groupby("month"):
        k = sample_requirements.get(m, 0)
        if k == 0 or grp.empty:
            continue
        grp = grp.sort_values("date").reset_index(drop=True)
        if k == 1:
            keep.append(grp.loc[(grp["day"] - 15).abs().idxmin(), "date"].strftime("%Y-%m-%d"))
        else:
            keep.extend([
                grp.iloc[0]["date"].strftime("%Y-%m-%d"),
                grp.iloc[-1]["date"].strftime("%Y-%m-%d"),
            ])
    return keep


def create_tiles(arr: np.ndarray, tile=(24, 24)):
    pad_y = (-arr.shape[1]) % tile[0]
    pad_x = (-arr.shape[2]) % tile[1]
    arr = np.pad(arr, [(0, 0), (0, pad_y), (0, pad_x), (0, 0)], "reflect")
    tiles, ids = [], []
    for iy in range(0, arr.shape[1], tile[0]):
        for ix in range(0, arr.shape[2], tile[1]):
            tiles.append(arr[:, iy : iy + tile[0], ix : ix + tile[1]].copy())
            ids.append((iy, ix))
    return tiles, ids


def stack_bands(sat: dict) -> np.ndarray:
    """Stack selected bands along the last axis, **preserving uint16 dtype**."""
    layers = [sat[res][b] for res, bs in selected_bands.items() for b in bs]
    return np.dstack(layers).astype(np.uint16)


def read_satellite_image(root: str, date: str):
    sat: dict[str, dict[str, np.ndarray]] = {k: {} for k in selected_bands}
    for res, bands in selected_bands.items():
        for b in bands:
            fp = os.path.join(root, date, f"{b}_{date}.tif")
            if not os.path.exists(fp):
                return None
            with rasterio.open(fp) as src:
                data = src.read(1)
            if res == "SCL" and (data > 7).sum() / data.size > 0.10:
                return None  # too cloudy
            sat[res][b] = data

    # Upsample 20 m + SCL to 10 m grid
    h, w = sat["10m"]["B2"].shape
    for b in selected_bands["20m"]:
        sat["20m"][b] = cv2.resize(sat["20m"][b], (w, h), interpolation=cv2.INTER_NEAREST)
    scl_raw = sat["SCL"]["SCL"]
    if scl_raw.shape != (h, w):
        sat["SCL"]["SCL"] = cv2.resize(scl_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return sat


def load_meta_patch(meta_dir: str):
    dates = resample_dates([d for d in os.listdir(meta_dir) if os.path.isdir(os.path.join(meta_dir, d))])
    if not dates:
        return [], [], []

    cdl_fp = os.path.join(meta_dir, "cdl.tif")
    if not os.path.exists(cdl_fp):
        return [], [], []
    with rasterio.open(cdl_fp) as src:
        cdl = src.read(1)
    cdl_exp = cdl[..., None]

    stacks, doys = [], []
    for d in sorted(dates):
        sat = read_satellite_image(meta_dir, d)
        if sat is None:
            continue
        stacks.append(np.concatenate([stack_bands(sat), cdl_exp], axis=-1))
        doys.append(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday)
    if not stacks:
        return [], [], []

    arr = np.stack(stacks, 0)
    return *create_tiles(arr), doys  # tiles, ids, doys

# ─────────────────────── Blosc2 FRAME WRITER ────────────────────────

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

# ────────────────────────── CORE WORKER ────────────────────────────

def preprocess_sub_region(sub_dir: str, out_root: str):
    name = os.path.basename(sub_dir)
    out_dir = os.path.join(out_root, name)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "label"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "doy"), exist_ok=True)

    tiles, ids, doys = load_meta_patch(sub_dir)
    if not tiles:
        return  # all‑cloudy or missing

    doy_arr = np.asarray(doys, dtype=np.int16)

    for tile, (iy, ix) in tqdm(zip(tiles, ids), total=len(tiles), desc=f"{name} tiles"):
        # Preserve uint16 for spectral bands
        img_np = tile[..., :-1].astype(np.uint16)  # (time,24,24,11)
        label_np = tile[0, ..., -1].astype(np.uint8)  # (24,24)

        write_b2frame(img_np, os.path.join(out_dir, "img", f"{iy}_{ix}_img.b2frame"))
        write_b2frame(label_np, os.path.join(out_dir, "label", f"{iy}_{ix}_label.b2frame"))
        write_b2frame(doy_arr, os.path.join(out_dir, "doy", f"{iy}_{ix}_doy.b2frame"))


# ──────────────────────── HIGH‑LEVEL DRIVER ────────────────────────

def preprocess_satellite(raw_root: str, out_root: str, n_cpus: int = 8):
    os.makedirs(out_root, exist_ok=True)
    meta_dirs = glob(os.path.join(raw_root, "*"))
    task = partial(preprocess_sub_region, out_root=out_root)
    if n_cpus == 1:
        for d in tqdm(meta_dirs, desc="meta-patches"):
            task(d)
    else:
        with mp.Pool(n_cpus) as pool:
            list(tqdm(pool.imap_unordered(task, meta_dirs), total=len(meta_dirs), desc="meta-patches"))


# ───────────────────────────── CLI ────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Arkansas S2 → Blosc2 frame converter (uint16)")
    p.add_argument("--raw", required=True, help="raw meta-patch directory")
    p.add_argument("--out", required=True, help="output directory for *.b2frame files")
    p.add_argument("--n-cpus", type=int, default=8, help="number of CPU cores to use")
    args = p.parse_args()

    preprocess_satellite(args.raw, args.out, n_cpus=args.n_cpus)
