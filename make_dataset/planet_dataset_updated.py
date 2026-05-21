# maybe use this for making planet dataset
import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
from tqdm import tqdm
import rasterio
# import cv2
from rasterio.enums import Resampling
from tqdm import tqdm
import json
from pathlib import Path
from typing import List, Union

TIF_FILES = []

def save_band_minmax(band_minmax: dict, json_path: str):
    p = Path(json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): [float(v[0]), float(v[1])] for k, v in band_minmax.items()}
    with p.open("w") as f:
        json.dump(payload, f, indent=2)

def load_band_minmax(json_path: str) -> dict:
    p = Path(json_path)
    with p.open("r") as f:
        raw = json.load(f)
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}

def _gather_tifs(input_dirs: Union[str, List[str]]) -> List[str]:
    """Recursively collect all matching Planet TIFFs from one or many roots."""
    if isinstance(input_dirs, (str, Path)):
        roots = [str(input_dirs)]
    else:
        roots = [str(p) for p in input_dirs]

    matches = []
    exts = (".tif", ".tiff")
    suffixes = ("sr_8b_clip.tif", "sr_8b_clip.tiff")
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fn_l = fn.lower()
                if fn_l.endswith(exts) and fn_l.endswith(suffixes):
                    matches.append(os.path.join(dirpath, fn))
    matches = sorted(set(matches))
    print(f"Found {len(matches)} matching TIFFs across {len(roots)} root(s).")
    return matches

def robust_band_stats(band_indices, nodata_value=0, q_low=2, q_high=98, sample_step=8):
    """
    Compute robust per-band percentiles across the dataset.
    sample_step: read every Nth row/col to speed up.
    Returns: dict {band_idx: (p_low, p_high)}
    """
    stats = {b: [] for b in band_indices}
    # for fname in tqdm(sorted(matches), desc="Scanning percentiles"):
    #     path = os.path.join(input_dir, fname)
    #     with rasterio.open(path) as src:
    for name in tqdm(TIF_FILES):
        with rasterio.open(name) as src:
            for b in band_indices:
                arr = src.read(b)  # (H,W)
                # Subsample to speed up (optional)
                arr = arr[::sample_step, ::sample_step]
                if nodata_value is not None:
                    mask = arr != nodata_value
                    arr = arr[mask]
                if arr.size == 0:
                    continue
                p1 = np.percentile(arr, q_low)
                p2 = np.percentile(arr, q_high)
                stats[b].append((p1, p2))
    # aggregate (median of per-image percentiles)
    out = {}
    for b, lst in stats.items():
        if len(lst) == 0:
            out[b] = (0.0, 1.0)
        else:
            lows = [x[0] for x in lst]
            highs = [x[1] for x in lst]
            out[b] = (float(np.median(lows)), float(np.median(highs)))
    return out


def to_uint8_rgb(patch, band_indices, band_minmax):
    """
    Convert multi-band patch to uint8 RGB using per-band robust min/max.
    band_indices are the 1-based band numbers used when reading.
    """
    H, W, C = patch.shape
    # pick bands for RGB, in your order (e.g., [6,4,2]):
    assert len(band_indices) == C, "band_indices must match patch's channels"
    out = np.zeros((H, W, 3), dtype=np.uint8)
    # Map first 3 channels of patch onto RGB
    for ch in range(min(3, C)):
        b = band_indices[ch]
        lo, hi = band_minmax[b]
        arr = patch[:, :, ch].astype(np.float32)
        if hi > lo:
            arr = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        out[:, :, ch] = arr.astype(np.uint8)
    return out

def rescale_img(img_in, scale: float):
    w, h = img_in.size
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img_in.resize(new_size, resample=Image.BICUBIC)

def create_resized_tiff_patches_from_tiff(input_dirs: Union[str, List[str]], output_dir, patch_size=256, scale=4,
                                          overlap=0.5, black_threshold=50, max_black_ratio=0.5,
                                          sub_dir="AR", image_type="RGB"):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    hr_dir = os.path.join(output_dir, "HR")
    os.makedirs(os.path.join(hr_dir, sub_dir), exist_ok=True)
    lr_dir = os.path.join(output_dir, "LR")
    os.makedirs(os.path.join(lr_dir, sub_dir), exist_ok=True)

    hr_dir_farmland = os.path.join(hr_dir, sub_dir)
    lr_dir_farmland = os.path.join(lr_dir, sub_dir)

    all_tifs = _gather_tifs(input_dirs)
    print(f"Found {len(all_tifs)} SR_8b_clip.tif files")

    # Choose which bands to read (1-based indices)
    band_indices = [6, 4, 2] if image_type == "RGB" else None

    # First pass: compute robust per-band percentiles for PNG visualization (skip if already uint8)
    with rasterio.open(all_tifs[0]) as tmp_src:
        dtype_is_uint8 = (tmp_src.dtypes[0] == 'uint8')

    tif_files_dir = "/home/yikebe/research/FastDiffSR/use_train_whole_season"
    
    for png_file in os.listdir(tif_files_dir):
        if png_file.lower().endswith(".png"):
            tif_name = png_file.replace(".png", ".tif")
            for tif_path in all_tifs:
                if os.path.basename(tif_path) == tif_name:
                    TIF_FILES.append(tif_path)
            
    band_minmax = None
    if (image_type == "RGB") and (not dtype_is_uint8):
        band_minmax = robust_band_stats(band_indices, nodata_value=0, q_low=2, q_high=98)
        save_band_minmax(band_minmax, os.path.join(output_dir, "band_minmax.json"))
        # print("Computed band percentiles:", band_minmax)
        # band_minmax = load_band_minmax(os.path.join(output_dir, "band_minmax.json"))

    patch_count = 0
            
    # for file_name in sorted(matches):
    #     img_path = os.path.join(input_dir, file_name)
    for i in range(len(TIF_FILES)):
        img_path = TIF_FILES[i]
        with rasterio.open(img_path) as src:
            if image_type == "RGB":
                arr = src.read(band_indices)  # (C,H,W) with selected bands
            else:
                arr = src.read()  # (C,H,W) all bands
            arr = np.moveaxis(arr, 0, -1)  # -> (H,W,C)

            # Keep original dtype for TIFF patches:
            num_bands = arr.shape[-1]
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=patch_size,
                width=patch_size,
                count=num_bands,
                dtype=arr.dtype
            )

            height, width, _ = arr.shape
            step = int(patch_size * (1 - overlap)) if overlap < 1 else patch_size

            for y in range(0, height, step):
                for x in range(0, width, step):
                    if y + patch_size > height or x + patch_size > width:
                        patch = np.zeros((patch_size, patch_size, num_bands), dtype=arr.dtype)
                        patch[:min(height - y, patch_size), :min(width - x, patch_size), :] = \
                            arr[y:min(y+patch_size, height), x:min(x+patch_size, width), :]
                    else:
                        patch = arr[y:y + patch_size, x:x + patch_size, :]

                    # black ratio test (use first 3 channels or luminance proxy)
                    # Define "black" on original dtype:
                    # If uint16, a threshold like 64 or 128 is usually better than 50
                    thr = black_threshold
                    if patch.dtype == np.uint16:
                        thr = 128  
                    black_pixels = np.sum(np.all(patch[:, :, :min(3,num_bands)] <= thr, axis=-1))
                    total_pixels = patch_size * patch_size
                    black_ratio = black_pixels / total_pixels
                    if black_ratio > max_black_ratio:
                        continue

                    # Make PNG preview (uint8) with consistent contrast
                    png_path = os.path.join(hr_dir_farmland, f"{patch_count:04d}.png")
                    if dtype_is_uint8:
                        # Already 8-bit → just take first 3 channels
                        png_rgb = patch[:, :, :3].astype(np.uint8)
                    else:
                        if image_type == "RGB":
                            png_rgb = to_uint8_rgb(patch[:, :, :3], band_indices, band_minmax)
                        else:
                            # If not RGB, pick first 3 bands for a pseudo-RGB preview with per-band stretch
                            # Build dummy band_indices 1..3 and compute simple min/max from patch
                            local_minmax = {}
                            for ch in range(min(3, num_bands)):
                                pmin = np.percentile(patch[:, :, ch], 2)
                                pmax = np.percentile(patch[:, :, ch], 98)
                                local_minmax[ch+1] = (pmin, pmax)
                            png_rgb = to_uint8_rgb(patch[:, :, :3], [1,2,3], local_minmax)

                    Image.fromarray(png_rgb, 'RGB').save(png_path)
                    
                    # Write TIFF patch preserving dtype
                    tif_path = os.path.join(hr_dir_farmland, f"{patch_count:04d}.tif")
                    with rasterio.open(tif_path, 'w', **profile) as dst:
                        
                        for b in range(num_bands):
                            dst.write(patch[:, :, b], b + 1)

                    # LR PNG (downscale from PNG preview to keep display range consistent)
                    # lr = rescale_img(Image.fromarray(png_rgb, 'RGB'), scale=1/scale)
                    # lr.save(os.path.join(lr_dir_farmland, f"{patch_count:04d}.png"), format='PNG')

                    patch_count += 1

    print(f"Extracted {patch_count} patches to {output_dir}")


def main():
    # input_dir = "/home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene"
    input_dirs = ["/home/thanyu/data/analytic_8b_sr_udm2/Lower/2020_03_cf5b8a65-366f-4b70-946f-44166d008371/cf5b8a65-366f-4b70-946f-44166d008371/PSScene/",
                  "/home/thanyu/data/analytic_8b_sr_udm2/Lower/2020_07_fa1621d8-017b-4bb1-9d9f-030250057dab/fa1621d8-017b-4bb1-9d9f-030250057dab/PSScene",
                  "/home/thanyu/data/analytic_8b_sr_udm2/Lower/2020_11_b0121492-7902-41de-9b36-dc9af277c715/b0121492-7902-41de-9b36-dc9af277c715/PSScene/"
                ]
    # input_dir = "/home/yikebe/research/FastDiffSR/use_inference_Dec"
    output_dir = "../planet_dataset_whole_season"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    create_resized_tiff_patches_from_tiff(input_dirs, output_dir, scale=4, overlap=0, black_threshold=50, max_black_ratio=0.5)

main()    