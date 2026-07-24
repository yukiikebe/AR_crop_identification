# io_utils.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as cc_label


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_and_resample_band(
    band_path: Path,
    ref_src: rasterio.io.DatasetReader,
    *,
    scl: bool = False,
    resampling: Optional[Resampling] = None,
) -> np.ndarray:
    """
    Read a band and resample/reproject to match ref_src grid.

    - Continuous reflectance bands -> default bilinear
    - Discrete labels (SCL) -> pass resampling=Resampling.nearest
    """
    if resampling is None:
        resampling = Resampling.nearest if scl else Resampling.bilinear

    with rasterio.open(band_path) as src:
        if not scl:
            data = src.read(1).astype("float32") / 10000.0
        else:
            data = src.read(1).astype("uint8")

        if src.shape == (ref_src.height, ref_src.width):
            return data

        out_dtype = "float32" if not scl else "uint8"
        out = np.empty((ref_src.height, ref_src.width), dtype=out_dtype)

        rasterio.warp.reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            resampling=resampling,
        )
        return out


def save_resampled_plot_mask_once(
    mask_path: Path,
    sample_path: Path,
    out_npy: Path,
    *,
    logger=None,
) -> Path:
    """
    Create an aligned (resampled) CDL mask once and cache it as .npy.
    Chooses one timestamp folder as reference grid.
    """
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    if out_npy.exists():
        if logger:
            logger.info(f"Aligned mask exists: {out_npy}")
        return out_npy

    # Find any timestamp folder with a B4 band to use as reference
    for ts_folder in sorted(sample_path.iterdir()):
        if not ts_folder.is_dir():
            continue
        try:
            files = [p.name for p in ts_folder.iterdir() if p.is_file()]
            red_file = next(f for f in files if "B4" in f)
            red_path = ts_folder / red_file

            with rasterio.open(red_path) as ref_src:
                with rasterio.open(mask_path) as mask_src:
                    mask_data = mask_src.read(1)

                    out = np.empty((ref_src.height, ref_src.width), dtype=np.float32)
                    rasterio.warp.reproject(
                        source=mask_data,
                        destination=out,
                        src_transform=mask_src.transform,
                        src_crs=mask_src.crs,
                        dst_transform=ref_src.transform,
                        dst_crs=ref_src.crs,
                        resampling=Resampling.nearest,
                    )

                np.save(out_npy, out.astype(int))
                if logger:
                    logger.info(f"Saved aligned mask: {out_npy}")
                return out_npy
        except Exception as e:
            if logger:
                logger.warning(f"Mask align skip {ts_folder.name}: {e}")
            continue

    raise RuntimeError("Failed to find a valid timestamp folder to align the mask.")


def load_saved_index_images(
    index_dir: Path, timestamps: List[datetime]
) -> List[Optional[np.ndarray]]:
    """
    Load cached index .npy files in timestamp order.
    If missing, returns None for that timestamp.
    """
    out: List[Optional[np.ndarray]] = []
    for ts in timestamps:
        f_path = index_dir / f"{ts:%Y%m%d}.npy"
        if not f_path.exists():
            out.append(None)
            continue
        try:
            data = np.load(f_path)
        except (OSError, ValueError):
            out.append(None)
            continue
        if np.nanmax(data) > 1.5:
            data = data / 10000.0
        out.append(data)
    return out


def reconstruct_mask(coords: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """coords: array of (y,x) indices."""
    mask = np.zeros(shape, dtype=bool)
    if coords.size > 0:
        mask[tuple(coords.T)] = True
    return mask


def get_farms_in_crop_mask(
    crop_mask: np.ndarray,
    crop_val: int,
    transform,
    *,
    min_pixels: int = 15,
) -> Dict[str, np.ndarray]:
    """
    Same flow as original: connected components inside (crop_mask == crop_val),
    create farm_id from geographic center, store coords (y,x) indices.
    """
    binary = (crop_mask == crop_val).astype(np.uint8)
    labeled, num = cc_label(binary)
    farms: Dict[str, np.ndarray] = {}

    for i in range(1, num + 1):
        farm_pix = labeled == i
        if int(farm_pix.sum()) < int(min_pixels):
            continue

        cy, cx = center_of_mass(farm_pix)
        x_geo, y_geo = rasterio.transform.xy(transform, cy, cx, offset="center")
        coords = np.argwhere(farm_pix)  # (y,x)

        farm_id = f"{int(x_geo)}_{int(y_geo)}"
        farms[farm_id] = coords

    return farms


def cleanup_index_cache(crop_dir: Path):
    for idx in ["NDVI", "NDWI", "EVI"]:
        d = crop_dir / idx
        if d.exists():
            for f in d.glob("*.npy"):
                f.unlink()
