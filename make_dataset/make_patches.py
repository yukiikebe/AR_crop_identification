import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    from .dataset import get_patch, rescale_img
except ImportError:
    from dataset import get_patch, rescale_img


def normalize_input_dirs(input_dir):
    if isinstance(input_dir, (list, tuple)):
        return list(input_dir)
    return [input_dir]


def build_unique_output_path(output_dir, file_name):
    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        return output_path

    stem, ext = os.path.splitext(file_name)
    suffix = 1
    while True:
        candidate = os.path.join(output_dir, f"{stem}_{suffix}{ext}")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def is_sea_patch(
    patch,
    min_blue_ratio=0.35,
    blue_red_diff=12.0,
    blue_green_diff=6.0,
    max_std=18.0,
):
    """Heuristic filter for water-dominated patches using the first 3 channels."""
    if patch.shape[-1] < 3:
        return False

    patch_rgb = patch[:, :, :3].astype(np.float32)
    blue = patch_rgb[:, :, 2]
    green = patch_rgb[:, :, 1]
    red = patch_rgb[:, :, 0]

    water_mask = (
        (blue - red >= blue_red_diff)
        & (blue - green >= blue_green_diff)
        & (np.std(patch_rgb, axis=-1) <= max_std)
    )
    water_ratio = water_mask.mean()
    return water_ratio >= min_blue_ratio


def normalize_to_uint8(img_array):
    if img_array.dtype == np.uint8:
        return img_array

    img_array = img_array.astype(np.float32, copy=False)
    img_min = float(img_array.min())
    img_max = float(img_array.max())
    if img_max <= img_min:
        return np.zeros(img_array.shape, dtype=np.uint8)

    scaled = (img_array - img_min) / (img_max - img_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def resolve_patch_step(patch_size, overlap):
    step = int(patch_size * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap must be less than 1.0")
    return step


def iter_complete_windows(width, height, patch_size, step):
    max_x = int(width) - int(patch_size)
    max_y = int(height) - int(patch_size)
    if max_x < 0 or max_y < 0:
        return
    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            yield Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
            x += step
        y += step


def resolve_resize_device(device):
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if device != "cpu" and torch is None:
        raise ImportError("PyTorch is required when using --device other than cpu.")

    if device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        raise ValueError(f"Requested device '{device}' but CUDA is not available.")

    return device


def matches_name_filters(file_name, include_name=None, exclude_name=None):
    include_name = include_name or []
    exclude_name = exclude_name or []

    if include_name and not all(token in file_name for token in include_name):
        return False
    if exclude_name and any(token in file_name for token in exclude_name):
        return False
    return True


def collect_tiff_sources(input_dirs, recursive=False, include_name=None, exclude_name=None):
    sources = []
    for source_index, current_input_dir in enumerate(input_dirs):
        if recursive:
            for root, _, file_names in os.walk(current_input_dir):
                for file_name in sorted(file_names):
                    img_path = os.path.join(root, file_name)
                    if file_name.lower().endswith((".tif", ".tiff")) and matches_name_filters(
                        file_name,
                        include_name=include_name,
                        exclude_name=exclude_name,
                    ):
                        sources.append(
                            {
                                "source_index": source_index,
                                "file_name": file_name,
                                "img_path": img_path,
                            }
                        )
        else:
            for file_name in sorted(os.listdir(current_input_dir)):
                img_path = os.path.join(current_input_dir, file_name)
                if file_name.lower().endswith((".tif", ".tiff")) and matches_name_filters(
                    file_name,
                    include_name=include_name,
                    exclude_name=exclude_name,
                ):
                    sources.append(
                        {
                            "source_index": source_index,
                            "file_name": file_name,
                            "img_path": img_path,
                        }
                    )

    return sources


def pad_image_for_patch_grid(img_array, patch_size, step):
    height, width = img_array.shape[:2]
    num_rows = max(1, (height + step - 1) // step)
    num_cols = max(1, (width + step - 1) // step)
    target_height = (num_rows - 1) * step + patch_size
    target_width = (num_cols - 1) * step + patch_size
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)

    return np.pad(
        img_array,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
    )


def iter_patch_blocks(
    img_array,
    patch_size,
    step,
    patch_rows_per_block=8,
    patch_cols_per_block=8,
):
    padded = pad_image_for_patch_grid(img_array, patch_size, step)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (patch_size, patch_size),
        axis=(0, 1),
    )
    sampled_windows = windows[::step, ::step]
    num_rows, num_cols, num_channels = sampled_windows.shape[:3]

    for row_start in range(0, num_rows, patch_rows_per_block):
        for col_start in range(0, num_cols, patch_cols_per_block):
            block = sampled_windows[
                row_start : row_start + patch_rows_per_block,
                col_start : col_start + patch_cols_per_block,
            ]
            patches = block.transpose(0, 1, 3, 4, 2).reshape(-1, patch_size, patch_size, num_channels)
            yield np.ascontiguousarray(patches)


def build_valid_patch_mask(
    patches,
    black_threshold,
    max_black_ratio,
    skip_sea,
    min_sea_ratio,
    sea_blue_red_diff,
    sea_blue_green_diff,
    sea_max_std,
):
    black_ratio = np.all(patches <= black_threshold, axis=-1).mean(axis=(1, 2))
    keep_mask = black_ratio <= max_black_ratio

    if skip_sea and np.any(keep_mask):
        rgb_patches = patches[keep_mask, :, :, :3].astype(np.float32, copy=False)
        blue = rgb_patches[:, :, :, 2]
        green = rgb_patches[:, :, :, 1]
        red = rgb_patches[:, :, :, 0]
        std = np.std(rgb_patches, axis=-1)
        water_mask = (
            (blue - red >= sea_blue_red_diff)
            & (blue - green >= sea_blue_green_diff)
            & (std <= sea_max_std)
        )
        sea_ratio = water_mask.mean(axis=(1, 2))
        keep_mask[np.where(keep_mask)[0]] = sea_ratio < min_sea_ratio

    return keep_mask


def resize_patch_chunk_with_torch(hr_chunk, scale, device, output_dtype=None):
    if torch is None or F is None:
        raise ImportError("PyTorch is required for torch-based patch resizing.")

    output_dtype = np.dtype(output_dtype or hr_chunk.dtype)
    hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_chunk)).permute(0, 3, 1, 2).to(dtype=torch.float32)
    hr_tensor = hr_tensor.to(device, non_blocking=device != "cpu")

    try:
        lr_tensor = F.interpolate(
            hr_tensor,
            scale_factor=1 / scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        lr_tensor = F.interpolate(
            hr_tensor,
            scale_factor=1 / scale,
            mode="bicubic",
            align_corners=False,
        )

    lr_chunk = lr_tensor.permute(0, 2, 3, 1).cpu().numpy()

    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
        lr_chunk = np.clip(np.round(lr_chunk), info.min, info.max).astype(output_dtype)
    else:
        lr_chunk = lr_chunk.astype(output_dtype, copy=False)

    return lr_chunk


def resize_tiff_patch_with_rasterio(patch, scale, profile):
    new_height = max(1, patch.shape[0] // scale)
    new_width = max(1, patch.shape[1] // scale)
    mem_profile = dict(profile)
    mem_profile.update(
        driver="GTiff",
        height=patch.shape[0],
        width=patch.shape[1],
        count=patch.shape[-1],
        dtype=patch.dtype,
    )

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**mem_profile) as dataset:
            dataset.write(np.moveaxis(np.ascontiguousarray(patch), -1, 0))
            data = dataset.read(
                out_shape=(dataset.count, new_height, new_width),
                resampling=Resampling.cubic,
            )

    return np.moveaxis(data, 0, -1)


def resolve_tiff_band_selection(src, band_mode):
    if band_mode == "all":
        return list(range(1, src.count + 1))

    if src.count < 3:
        raise ValueError(f"RGB mode requires at least 3 bands, but {src.name} has {src.count}.")

    descriptions = [desc.lower() if isinstance(desc, str) else "" for desc in src.descriptions]
    desired_names = ("red", "green", "blue")
    band_indices = []

    for desired_name in desired_names:
        matched_index = next(
            (idx + 1 for idx, desc in enumerate(descriptions) if desc == desired_name),
            None,
        )
        if matched_index is None:
            matched_index = next(
                (idx + 1 for idx, desc in enumerate(descriptions) if desired_name in desc),
                None,
            )
        band_indices.append(matched_index)

    if all(index is not None for index in band_indices):
        return band_indices

    # Fall back to the PlanetScope SuperDove RGB convention: red, green, blue = 6, 4, 2.
    if src.count >= 6:
        return [6, 4, 2]

    return [1, 2, 3]


def save_png_patch_pair(hr_patch, hr_path, lr_path=None, lr_patch=None, scale=4):
    hr_patch = np.ascontiguousarray(hr_patch.astype(np.uint8, copy=False))
    hr_img = Image.fromarray(hr_patch)
    hr_img.save(hr_path, format="PNG")

    if lr_path is None:
        return

    if lr_patch is None:
        lr_patch = rescale_img(hr_img, scale=1 / scale)
        lr_patch.save(lr_path, format="PNG")
        return

    lr_patch = np.ascontiguousarray(lr_patch.astype(np.uint8, copy=False))
    Image.fromarray(lr_patch).save(lr_path, format="PNG")


def save_tiff_patch(patch, path, profile):
    patch = np.ascontiguousarray(patch)
    patch_profile = dict(profile)
    patch_profile.update(
        driver="GTiff",
        height=patch.shape[0],
        width=patch.shape[1],
        count=patch.shape[-1],
        dtype=patch.dtype,
    )
    with rasterio.open(path, "w", **patch_profile) as dst:
        dst.write(np.moveaxis(patch, -1, 0))


def save_rgb_geotiff(path, img_uint8, *, crs, transform):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(img_uint8.shape[0]),
        "width": int(img_uint8.shape[1]),
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(np.ascontiguousarray(img_uint8), -1, 0))


def parse_sentinel_date_dir(name):
    try:
        return datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        return None


def iter_sentinel_date_dirs(sentinel_roots):
    for root in sentinel_roots:
        raw_dir = Path(root)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Sentinel root not found: {raw_dir}")
        for grid_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
            if "_" not in grid_dir.name:
                continue
            for date_dir in sorted(path for path in grid_dir.iterdir() if path.is_dir()):
                parsed_date = parse_sentinel_date_dir(date_dir.name)
                if parsed_date is None:
                    continue
                band_paths = {band: date_dir / f"{band}_{date_dir.name}.tif" for band in ("B2", "B3", "B4", "SCL")}
                if all(path.exists() for path in band_paths.values()):
                    yield raw_dir, grid_dir.name, parsed_date, date_dir, band_paths


def compute_rgb_stats_from_tile(rgb, valid_mask, q_low=2.0, q_high=98.0):
    stats = []
    valid = valid_mask.astype(bool, copy=False)
    for channel in range(3):
        values = rgb[..., channel][valid]
        values = values[np.isfinite(values)]
        if values.size == 0:
            stats.append((0.0, 255.0))
            continue
        lo = float(np.percentile(values, q_low))
        hi = float(np.percentile(values, q_high))
        if hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        stats.append((lo, hi))
    return tuple(stats)


def normalize_rgb_with_stats(rgb, stats):
    out = np.zeros(rgb.shape[:2] + (3,), dtype=np.uint8)
    for channel, (lo, hi) in enumerate(stats):
        if hi <= lo:
            continue
        scaled = (np.clip(rgb[..., channel].astype(np.float32, copy=False), lo, hi) - lo) / (hi - lo)
        out[..., channel] = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    return out


def brightness_stats_uint8(img_uint8, bright_threshold=245):
    brightness = img_uint8.astype(np.float32, copy=False).mean(axis=-1)
    return float(brightness.mean()), float((brightness >= int(bright_threshold)).mean())


def build_sr_patch_uint8(lr_patch_uint8, sr_patch_size):
    return np.asarray(
        Image.fromarray(lr_patch_uint8, mode="RGB").resize(
            (int(sr_patch_size), int(sr_patch_size)),
            resample=Image.BICUBIC,
        ),
        dtype=np.uint8,
    )


def create_sentinel_geopatches(
    *,
    sentinel_roots,
    out,
    sentinel_patch_size=64,
    sr_patch_size=256,
    scale=4,
    patch_overlap=0.0,
    max_scl_cloud_ratio=0.0,
    black_threshold=30,
    max_black_ratio=0.0,
    skip_sea=False,
    min_sea_ratio=0.35,
    q_low=2.0,
    q_high=98.0,
    limit_patches=None,
    limit_dates=None,
    debug_png_count=10,
):
    if sr_patch_size % sentinel_patch_size != 0:
        raise ValueError("--sr-patch-size must be divisible by --sentinel-patch-size.")
    expected_scale = sr_patch_size // sentinel_patch_size
    if int(scale) != int(expected_scale):
        raise ValueError(
            f"--scale={scale} does not match {sr_patch_size}/{sentinel_patch_size}={expected_scale}."
        )

    out = Path(out)
    lr_dir = out / f"lr_{sentinel_patch_size}"
    sr_dir = out / f"sr_{sentinel_patch_size}_{sr_patch_size}"
    hr_dir = out / f"hr_{sr_patch_size}"
    for path in (lr_dir, sr_dir, hr_dir):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "patch_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")
    debug_dirs = {}
    if int(debug_png_count) > 0:
        debug_root = out / "debug_png"
        debug_dirs = {
            "lr": debug_root / f"lr_{sentinel_patch_size}",
            "sr": debug_root / f"sr_{sentinel_patch_size}_{sr_patch_size}",
            "hr": debug_root / f"hr_{sr_patch_size}",
        }
        for path in debug_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    step = resolve_patch_step(int(sentinel_patch_size), float(patch_overlap))
    sample_count = 0
    processed_dates = 0

    with manifest_path.open("a", encoding="utf-8") as manifest_fp:
        date_iter = iter_sentinel_date_dirs(sentinel_roots)
        for raw_dir, grid_id, sentinel_date, date_dir, band_paths in tqdm(date_iter, desc="Sentinel date tiles"):
            if limit_dates is not None and processed_dates >= int(limit_dates):
                break
            processed_dates += 1

            try:
                with rasterio.open(band_paths["B2"]) as b2_ds:
                    blue = b2_ds.read(1)
                    crs = b2_ds.crs
                    transform = b2_ds.transform
                    width = b2_ds.width
                    height = b2_ds.height
                    profile_bounds = tuple(float(v) for v in b2_ds.bounds)
                with rasterio.open(band_paths["B3"]) as b3_ds:
                    green = b3_ds.read(1, out_shape=(height, width), resampling=Resampling.bilinear)
                with rasterio.open(band_paths["B4"]) as b4_ds:
                    red = b4_ds.read(1, out_shape=(height, width), resampling=Resampling.bilinear)
                with rasterio.open(band_paths["SCL"]) as scl_ds:
                    scl = scl_ds.read(1, out_shape=(height, width), resampling=Resampling.nearest)
            except Exception as exc:
                manifest_fp.write(
                    json.dumps(
                        {
                            "kind": "sentinel_tile",
                            "status": "skipped",
                            "reason": f"open_failed:{exc}",
                            "source_root": str(raw_dir),
                            "grid_id": grid_id,
                            "date": sentinel_date.isoformat(),
                            "date_dir": str(date_dir),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue

            rgb_raw = np.dstack([red, green, blue]).astype(np.float32, copy=False)
            valid_mask = np.any(rgb_raw != 0, axis=-1)
            rgb_stats = compute_rgb_stats_from_tile(rgb_raw, valid_mask, q_low=q_low, q_high=q_high)
            rgb_uint8 = normalize_rgb_with_stats(rgb_raw, rgb_stats)

            for window in iter_complete_windows(width, height, int(sentinel_patch_size), step):
                if limit_patches is not None and sample_count >= int(limit_patches):
                    print(f"[done] wrote {sample_count} Sentinel patches to {out}")
                    return

                row0 = int(window.row_off)
                col0 = int(window.col_off)
                row1 = row0 + int(window.height)
                col1 = col0 + int(window.width)
                lr_patch = rgb_uint8[row0:row1, col0:col1, :]
                scl_patch = scl[row0:row1, col0:col1]
                valid_patch = valid_mask[row0:row1, col0:col1]
                if lr_patch.shape[:2] != (sentinel_patch_size, sentinel_patch_size):
                    continue

                valid_ratio = float(valid_patch.mean())
                if valid_ratio <= 0.0:
                    reason = "empty_patch"
                else:
                    reason = None
                cloud_ratio = float(((scl_patch > 7) & valid_patch).sum() / max(1, valid_patch.sum()))
                scl_water_ratio = float(((scl_patch == 6) & valid_patch).sum() / max(1, valid_patch.sum()))
                black = float(np.all(lr_patch <= int(black_threshold), axis=-1).mean())
                mean_brightness, bright_ratio = brightness_stats_uint8(lr_patch)

                if reason is None and cloud_ratio > float(max_scl_cloud_ratio):
                    reason = "too_cloudy"
                if reason is None and black > float(max_black_ratio):
                    reason = "black_ratio"
                if reason is None and bool(skip_sea) and scl_water_ratio >= float(min_sea_ratio):
                    reason = "water_ratio"

                patch_bounds = tuple(float(v) for v in window_bounds(window, transform))
                bounds_epsg4326 = None
                if crs is not None:
                    try:
                        bounds_epsg4326 = tuple(float(v) for v in transform_bounds(crs, "EPSG:4326", *patch_bounds))
                    except Exception:
                        bounds_epsg4326 = None

                if reason is not None:
                    manifest_fp.write(
                        json.dumps(
                            {
                                "kind": "sentinel_patch",
                                "status": "skipped",
                                "reason": reason,
                                "source_root": str(raw_dir),
                                "grid_id": grid_id,
                                "date": sentinel_date.isoformat(),
                                "date_dir": str(date_dir),
                                "bounds": patch_bounds,
                                "bounds_epsg4326": bounds_epsg4326,
                                "crs": str(crs) if crs is not None else None,
                                "valid_ratio": valid_ratio,
                                "scl_cloud_ratio": cloud_ratio,
                                "scl_water_ratio": scl_water_ratio,
                                "black_ratio": black,
                                "mean_brightness": mean_brightness,
                                "bright_ratio": bright_ratio,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                sample_id = f"{sample_count:06d}"
                file_name = f"{sample_id}.tif"
                sr_patch = build_sr_patch_uint8(lr_patch, int(sr_patch_size))
                lr_transform = from_bounds(*patch_bounds, int(sentinel_patch_size), int(sentinel_patch_size))
                sr_transform = from_bounds(*patch_bounds, int(sr_patch_size), int(sr_patch_size))
                lr_path = lr_dir / file_name
                sr_path = sr_dir / file_name
                hr_path = hr_dir / file_name
                save_rgb_geotiff(lr_path, lr_patch, crs=crs, transform=lr_transform)
                save_rgb_geotiff(sr_path, sr_patch, crs=crs, transform=sr_transform)
                save_rgb_geotiff(hr_path, sr_patch, crs=crs, transform=sr_transform)
                if sample_count < int(debug_png_count):
                    png_name = f"{sample_id}.png"
                    Image.fromarray(lr_patch, mode="RGB").save(debug_dirs["lr"] / png_name)
                    Image.fromarray(sr_patch, mode="RGB").save(debug_dirs["sr"] / png_name)
                    Image.fromarray(sr_patch, mode="RGB").save(debug_dirs["hr"] / png_name)

                manifest_fp.write(
                    json.dumps(
                        {
                            "kind": "sentinel_patch",
                            "status": "kept",
                            "reason": "kept",
                            "sample_id": sample_id,
                            "source_root": str(raw_dir),
                            "grid_id": grid_id,
                            "date": sentinel_date.isoformat(),
                            "date_dir": str(date_dir),
                            "bounds": patch_bounds,
                            "source_tile_bounds": profile_bounds,
                            "bounds_epsg4326": bounds_epsg4326,
                            "crs": str(crs) if crs is not None else None,
                            "transform": tuple(float(v) for v in lr_transform)[:6],
                            "sr_transform": tuple(float(v) for v in sr_transform)[:6],
                            "valid_ratio": valid_ratio,
                            "scl_cloud_ratio": cloud_ratio,
                            "scl_water_ratio": scl_water_ratio,
                            "black_ratio": black,
                            "mean_brightness": mean_brightness,
                            "bright_ratio": bright_ratio,
                            "lr_path": str(lr_path),
                            "sr_path": str(sr_path),
                            "hr_path": str(hr_path),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                sample_count += 1

    print(f"[done] wrote {sample_count} Sentinel patches to {out}. Manifest: {manifest_path}")


def process_png_tiff_source(
    source,
    hr_dir,
    lr_dir,
    patch_size,
    scale,
    overlap,
    black_threshold,
    max_black_ratio,
    hr_only,
    skip_sea,
    min_sea_ratio,
    sea_blue_red_diff,
    sea_blue_green_diff,
    sea_max_std,
    device,
    resize_batch_size,
):
    source_index = source["source_index"]
    file_name = source["file_name"]
    img_path = source["img_path"]
    step = resolve_patch_step(patch_size, overlap)

    hr_dir_group = os.path.join(hr_dir, str(source_index))
    os.makedirs(hr_dir_group, exist_ok=True)
    lr_dir_group = None
    if not hr_only:
        lr_dir_group = os.path.join(lr_dir, str(source_index))
        os.makedirs(lr_dir_group, exist_ok=True)

    try:
        with rasterio.open(img_path) as src:
            img_array = src.read([1, 2, 3])
            img_array = np.moveaxis(img_array, 0, -1)
            img_array = normalize_to_uint8(img_array)
    except Exception as exc:
        print(f"Error loading {file_name}: {exc}")
        return 0

    local_patch_count = 0
    use_torch_resize = torch is not None

    for patch_block in iter_patch_blocks(img_array, patch_size, step):
        keep_mask = build_valid_patch_mask(
            patch_block,
            black_threshold=black_threshold,
            max_black_ratio=max_black_ratio,
            skip_sea=skip_sea,
            min_sea_ratio=min_sea_ratio,
            sea_blue_red_diff=sea_blue_red_diff,
            sea_blue_green_diff=sea_blue_green_diff,
            sea_max_std=sea_max_std,
        )
        kept_patches = patch_block[keep_mask]
        if kept_patches.size == 0:
            continue

        if hr_only:
            for offset, patch in enumerate(kept_patches):
                patch_name = f"{local_patch_count + offset:04d}.png"
                hr_path = os.path.join(hr_dir_group, patch_name)
                save_png_patch_pair(patch, hr_path)
            local_patch_count += len(kept_patches)
            continue

        if use_torch_resize:
            for batch_start in range(0, len(kept_patches), resize_batch_size):
                hr_chunk = kept_patches[batch_start : batch_start + resize_batch_size]
                lr_chunk = resize_patch_chunk_with_torch(hr_chunk, scale=scale, device=device, output_dtype=np.uint8)
                for offset, (hr_patch, lr_patch) in enumerate(zip(hr_chunk, lr_chunk)):
                    patch_name = f"{local_patch_count + batch_start + offset:04d}.png"
                    hr_path = os.path.join(hr_dir_group, patch_name)
                    lr_path = os.path.join(lr_dir_group, patch_name)
                    save_png_patch_pair(hr_patch, hr_path, lr_path=lr_path, lr_patch=lr_patch)
        else:
            for offset, patch in enumerate(kept_patches):
                patch_name = f"{local_patch_count + offset:04d}.png"
                hr_path = os.path.join(hr_dir_group, patch_name)
                lr_path = os.path.join(lr_dir_group, patch_name)
                save_png_patch_pair(patch, hr_path, lr_path=lr_path, scale=scale)

        local_patch_count += len(kept_patches)

    return local_patch_count


def process_tiff_source(
    source,
    hr_dir,
    lr_dir,
    patch_size,
    scale,
    overlap,
    black_threshold,
    max_black_ratio,
    hr_only,
    skip_sea,
    min_sea_ratio,
    sea_blue_red_diff,
    sea_blue_green_diff,
    sea_max_std,
    device,
    resize_batch_size,
    start_index,
    patch_name_prefix=None,
    band_mode="all",
):
    file_name = source["file_name"]
    img_path = source["img_path"]
    step = resolve_patch_step(patch_size, overlap)

    try:
        with rasterio.open(img_path) as src:
            band_indices = resolve_tiff_band_selection(src, band_mode)
            img_array = np.moveaxis(src.read(band_indices), 0, -1)
            profile = src.profile.copy()
    except Exception as exc:
        print(f"Error loading {file_name}: {exc}")
        return 0

    local_patch_count = 0
    use_torch_resize = torch is not None and not hr_only

    for patch_block in iter_patch_blocks(img_array, patch_size, step):
        keep_mask = build_valid_patch_mask(
            patch_block,
            black_threshold=black_threshold,
            max_black_ratio=max_black_ratio,
            skip_sea=skip_sea,
            min_sea_ratio=min_sea_ratio,
            sea_blue_red_diff=sea_blue_red_diff,
            sea_blue_green_diff=sea_blue_green_diff,
            sea_max_std=sea_max_std,
        )
        kept_patches = patch_block[keep_mask]
        if kept_patches.size == 0:
            continue

        lr_chunks = None
        if use_torch_resize:
            lr_chunks = []
            for batch_start in range(0, len(kept_patches), resize_batch_size):
                hr_chunk = kept_patches[batch_start : batch_start + resize_batch_size]
                lr_chunks.append(
                    resize_patch_chunk_with_torch(
                        hr_chunk,
                        scale=scale,
                        device=device,
                        output_dtype=kept_patches.dtype,
                    )
                )
            lr_chunks = np.concatenate(lr_chunks, axis=0) if lr_chunks else np.empty((0,), dtype=kept_patches.dtype)

        for offset, hr_patch in enumerate(kept_patches):
            patch_index = start_index + local_patch_count + offset
            patch_name = (
                f"{patch_index:04d}.tif"
                if patch_name_prefix is None
                else f"{patch_name_prefix}_{patch_index:06d}.tif"
            )
            hr_path = os.path.join(hr_dir, patch_name)
            save_tiff_patch(hr_patch, hr_path, profile)

            if hr_only:
                continue

            lr_path = os.path.join(lr_dir, patch_name)
            if use_torch_resize:
                lr_patch = lr_chunks[offset]
            else:
                lr_patch = resize_tiff_patch_with_rasterio(hr_patch, scale=scale, profile=profile)
            save_tiff_patch(lr_patch, lr_path, profile)

        local_patch_count += len(kept_patches)

    return local_patch_count


def build_tiff_patch_prefix(source):
    stem = os.path.splitext(source["file_name"])[0]
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_")
    if not safe_stem:
        safe_stem = "patch"
    source_hash = hashlib.sha1(source["img_path"].encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{source_hash}"


def create_resized_patches(input_dir, hr_dir, lr_dir=None, patch_size=256, scale=4, hr_only=False):
    """Create HR PNG patches and optionally LR PNG patches from standard RGB image folders."""
    input_dirs = normalize_input_dirs(input_dir)
    os.makedirs(hr_dir, exist_ok=True)
    if not hr_only:
        if lr_dir is None:
            raise ValueError("lr_dir is required unless hr_only=True")
        os.makedirs(lr_dir, exist_ok=True)

    for current_input_dir in input_dirs:
        for category in os.listdir(current_input_dir):
            category_path = os.path.join(current_input_dir, category)
            if not os.path.isdir(category_path):
                continue

            output_hr_category_path = os.path.join(hr_dir, category)
            os.makedirs(output_hr_category_path, exist_ok=True)
            output_lr_category_path = None
            if not hr_only:
                output_lr_category_path = os.path.join(lr_dir, category)
                os.makedirs(output_lr_category_path, exist_ok=True)

            for file_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
                if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(category_path, file_name)
                img = Image.open(img_path).convert("RGB")
                img_width, img_height = img.size
                if img_width < patch_size or img_height < patch_size:
                    continue

                hr_patch, _ = get_patch(img, img, patch_size, scale)
                hr_patch.save(build_unique_output_path(output_hr_category_path, file_name), format="PNG")
                if not hr_only:
                    lr_patch = rescale_img(hr_patch, scale=1 / scale)
                    lr_patch.save(build_unique_output_path(output_lr_category_path, file_name), format="PNG")


def create_resized_patches_from_tiff(
    input_dir,
    output_dir,
    patch_size=256,
    scale=4,
    overlap=0.5,
    black_threshold=50,
    max_black_ratio=0.5,
    hr_only=False,
    skip_sea=False,
    min_sea_ratio=0.35,
    sea_blue_red_diff=12.0,
    sea_blue_green_diff=6.0,
    sea_max_std=18.0,
    workers=1,
    device="cpu",
    resize_batch_size=64,
    recursive=False,
    include_name=None,
    exclude_name=None,
):
    """Create HR PNG patches and optionally LR PNG patches from TIFF images."""
    input_dirs = normalize_input_dirs(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    hr_dir = os.path.join(output_dir, "HR")
    os.makedirs(hr_dir, exist_ok=True)
    lr_dir = None
    if not hr_only:
        lr_dir = os.path.join(output_dir, "LR")
        os.makedirs(lr_dir, exist_ok=True)

    device = resolve_resize_device(device)
    if device != "cpu" and workers > 1:
        print("CUDA resize requested; forcing workers=1 to avoid GPU contention.")
        workers = 1

    sources = collect_tiff_sources(
        input_dirs,
        recursive=recursive,
        include_name=include_name,
        exclude_name=exclude_name,
    )
    if not sources:
        print(f"No TIFF files found in {input_dirs}")
        return

    process_kwargs = dict(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        patch_size=patch_size,
        scale=scale,
        overlap=overlap,
        black_threshold=black_threshold,
        max_black_ratio=max_black_ratio,
        hr_only=hr_only,
        skip_sea=skip_sea,
        min_sea_ratio=min_sea_ratio,
        sea_blue_red_diff=sea_blue_red_diff,
        sea_blue_green_diff=sea_blue_green_diff,
        sea_max_std=sea_max_std,
        device=device,
        resize_batch_size=resize_batch_size,
    )

    patch_count = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_png_tiff_source, source, **process_kwargs) for source in sources]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TIFF files"):
                patch_count += future.result()
    else:
        for source in tqdm(sources, desc="Processing TIFF files"):
            patch_count += process_png_tiff_source(source, **process_kwargs)

    print(f"Extracted {patch_count} patches to {output_dir}")


def create_resized_tiff_patches_from_tiff(
    input_dir,
    output_dir,
    patch_size=256,
    scale=4,
    overlap=0.5,
    black_threshold=50,
    max_black_ratio=0.5,
    sub_dir="Farmland",
    image_type="RGB",
    hr_only=False,
    skip_sea=False,
    min_sea_ratio=0.35,
    sea_blue_red_diff=12.0,
    sea_blue_green_diff=6.0,
    sea_max_std=18.0,
    device="cpu",
    resize_batch_size=32,
    workers=1,
    recursive=False,
    include_name=None,
    exclude_name=None,
    band_mode="all",
):
    """Create HR TIFF patches and optionally LR TIFF patches from TIFF images."""
    input_dirs = normalize_input_dirs(input_dir)
    del sub_dir
    del image_type
    os.makedirs(output_dir, exist_ok=True)
    hr_dir = os.path.join(output_dir, "HR")
    os.makedirs(hr_dir, exist_ok=True)
    lr_dir = None
    if not hr_only:
        lr_dir = os.path.join(output_dir, "LR")
        os.makedirs(lr_dir, exist_ok=True)

    device = resolve_resize_device(device)
    workers = max(1, int(workers))
    if device != "cpu" and workers > 1:
        print(f"Using device {device!r} for TIFF resize. Falling back to workers=1 to avoid device contention.")
        workers = 1

    sources = collect_tiff_sources(
        input_dirs,
        recursive=recursive,
        include_name=include_name,
        exclude_name=exclude_name,
    )
    if not sources:
        print(f"No TIFF files found in {input_dirs}")
        return

    process_kwargs = dict(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        patch_size=patch_size,
        scale=scale,
        overlap=overlap,
        black_threshold=black_threshold,
        max_black_ratio=max_black_ratio,
        hr_only=hr_only,
        skip_sea=skip_sea,
        min_sea_ratio=min_sea_ratio,
        sea_blue_red_diff=sea_blue_red_diff,
        sea_blue_green_diff=sea_blue_green_diff,
        sea_max_std=sea_max_std,
        device=device,
        resize_batch_size=resize_batch_size,
        band_mode=band_mode,
    )

    patch_count = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_tiff_source,
                    source,
                    start_index=0,
                    patch_name_prefix=build_tiff_patch_prefix(source),
                    **process_kwargs,
                )
                for source in sources
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TIFF files"):
                patch_count += future.result()
    else:
        for source in tqdm(sources, desc="Processing TIFF files"):
            patch_count += process_tiff_source(
                source,
                start_index=patch_count,
                **process_kwargs,
            )

    print(f"Extracted {patch_count} patches to {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Create PNG/TIFF image patches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rgb_parser = subparsers.add_parser("extract-rgb-patches", help="Create HR/LR PNG patches from RGB images.")
    rgb_parser.add_argument("--input-dir", nargs="+", required=True, help="One or more input directories with category subfolders.")
    rgb_parser.add_argument("--hr-dir", required=True, help="Output directory for HR patches.")
    rgb_parser.add_argument("--lr-dir", help="Output directory for LR patches.")
    rgb_parser.add_argument("--patch-size", type=int, default=256, help="HR patch size.")
    rgb_parser.add_argument("--scale", type=int, default=4, help="Downscaling factor for LR patches.")
    rgb_parser.add_argument("--hr-only", action="store_true", help="Only generate HR patches and skip LR output.")

    png_parser = subparsers.add_parser("extract-png-patches", help="Create HR/LR PNG patches from TIFF images.")
    png_parser.add_argument("--input-dir", nargs="+", required=True, help="One or more directories containing TIFF images.")
    png_parser.add_argument("--output-dir", required=True, help="Output directory containing HR/LR folders.")
    png_parser.add_argument("--patch-size", type=int, default=256, help="HR patch size.")
    png_parser.add_argument("--scale", type=int, default=4, help="Downscaling factor for LR patches.")
    png_parser.add_argument("--overlap", type=float, default=0.5, help="Patch overlap ratio in [0, 1).")
    png_parser.add_argument("--black-threshold", type=int, default=50, help="Threshold used to treat a pixel as black.")
    png_parser.add_argument("--max-black-ratio", type=float, default=0.5, help="Skip patches above this black ratio.")
    png_parser.add_argument("--hr-only", action="store_true", help="Only generate HR patches and skip LR output.")
    png_parser.add_argument("--skip-sea", action="store_true", help="Skip patches that look water-dominated.")
    png_parser.add_argument("--min-sea-ratio", type=float, default=0.35, help="Minimum water-like pixel ratio to skip a patch.")
    png_parser.add_argument("--sea-blue-red-diff", type=float, default=12.0, help="Minimum blue-red dominance for water-like pixels.")
    png_parser.add_argument("--sea-blue-green-diff", type=float, default=6.0, help="Minimum blue-green dominance for water-like pixels.")
    png_parser.add_argument("--sea-max-std", type=float, default=18.0, help="Maximum per-pixel RGB std for water-like pixels.")
    png_parser.add_argument("--workers", type=int, default=1, help="Number of TIFF files to process in parallel.")
    png_parser.add_argument("--device", default="cpu", help="Resize device: cpu, cuda, cuda:0, or auto.")
    png_parser.add_argument("--resize-batch-size", type=int, default=64, help="Batch size for LR resize when using torch-based resizing.")
    png_parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories for TIFF files.")
    png_parser.add_argument("--include-name", action="append", default=[], help="Only include TIFFs whose filenames contain this substring. Repeatable.")
    png_parser.add_argument("--exclude-name", action="append", default=[], help="Exclude TIFFs whose filenames contain this substring. Repeatable.")

    tiff_parser = subparsers.add_parser("extract-tiff-patches", help="Create HR/LR TIFF patches from TIFF images.")
    tiff_parser.add_argument("--input-dir", nargs="+", required=True, help="One or more directories containing TIFF images.")
    tiff_parser.add_argument("--output-dir", required=True, help="Output directory containing HR/LR folders.")
    tiff_parser.add_argument("--patch-size", type=int, default=256, help="HR patch size.")
    tiff_parser.add_argument("--scale", type=int, default=4, help="Downscaling factor for LR patches.")
    tiff_parser.add_argument("--overlap", type=float, default=0.5, help="Patch overlap ratio in [0, 1).")
    tiff_parser.add_argument("--black-threshold", type=int, default=50, help="Threshold used to treat a pixel as black.")
    tiff_parser.add_argument("--max-black-ratio", type=float, default=0.5, help="Skip patches above this black ratio.")
    tiff_parser.add_argument("--sub-dir", default="Farmland", help="Legacy label kept for compatibility.")
    tiff_parser.add_argument("--image-type", default="RGB", help="Image type passed through to the TIFF extractor.")
    tiff_parser.add_argument("--hr-only", action="store_true", help="Only generate HR patches and skip LR output.")
    tiff_parser.add_argument("--skip-sea", action="store_true", help="Skip patches that look water-dominated.")
    tiff_parser.add_argument("--min-sea-ratio", type=float, default=0.35, help="Minimum water-like pixel ratio to skip a patch.")
    tiff_parser.add_argument("--sea-blue-red-diff", type=float, default=12.0, help="Minimum blue-red dominance for water-like pixels.")
    tiff_parser.add_argument("--sea-blue-green-diff", type=float, default=6.0, help="Minimum blue-green dominance for water-like pixels.")
    tiff_parser.add_argument("--sea-max-std", type=float, default=18.0, help="Maximum per-pixel RGB std for water-like pixels.")
    tiff_parser.add_argument("--device", default="cpu", help="Resize device: cpu, cuda, cuda:0, or auto.")
    tiff_parser.add_argument("--resize-batch-size", type=int, default=32, help="Batch size for LR resize when using torch-based resizing.")
    tiff_parser.add_argument("--workers", type=int, default=1, help="Number of TIFF files to process in parallel.")
    tiff_parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories for TIFF files.")
    tiff_parser.add_argument("--include-name", action="append", default=[], help="Only include TIFFs whose filenames contain this substring. Repeatable.")
    tiff_parser.add_argument("--exclude-name", action="append", default=[], help="Exclude TIFFs whose filenames contain this substring. Repeatable.")
    tiff_parser.add_argument("--band-mode", choices=["all", "rgb"], default="all", help="Keep all TIFF bands or extract RGB-only TIFF patches.")

    sentinel_parser = subparsers.add_parser(
        "sentinel-geopatches",
        help="Create georeferenced Sentinel-2 FastDiffSR inference patches with a location manifest.",
    )
    sentinel_parser.add_argument("--sentinel-root", type=Path, action="append", required=True, help="Sentinel raw root such as MD_2018_raw. Repeatable.")
    sentinel_parser.add_argument("--out", type=Path, required=True, help="Output FastDiffSR-style dataset root.")
    sentinel_parser.add_argument("--sentinel-patch-size", type=int, default=64)
    sentinel_parser.add_argument("--sr-patch-size", type=int, default=256)
    sentinel_parser.add_argument("--scale", type=int, default=4)
    sentinel_parser.add_argument("--patch-overlap", type=float, default=0.0)
    sentinel_parser.add_argument("--max-scl-cloud-ratio", type=float, default=0.0)
    sentinel_parser.add_argument("--black-threshold", type=int, default=30)
    sentinel_parser.add_argument("--max-black-ratio", type=float, default=0.0)
    sentinel_parser.add_argument("--skip-sea", action="store_true", help="Skip water-dominated Sentinel patches using SCL water class.")
    sentinel_parser.add_argument("--min-sea-ratio", type=float, default=0.35, help="Minimum water-like pixel ratio to skip a Sentinel patch.")
    sentinel_parser.add_argument("--q-low", type=float, default=2.0)
    sentinel_parser.add_argument("--q-high", type=float, default=98.0)
    sentinel_parser.add_argument("--limit-patches", type=int, default=None, help="Stop after N kept patches for dry runs.")
    sentinel_parser.add_argument("--limit-dates", type=int, default=None, help="Stop after N date/grid folders for dry runs.")
    sentinel_parser.add_argument("--debug-png-count", type=int, default=10, help="Save the first N kept patches as PNG previews per output kind.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract-rgb-patches":
        create_resized_patches(
            input_dir=args.input_dir,
            hr_dir=args.hr_dir,
            lr_dir=args.lr_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            hr_only=args.hr_only,
        )
    elif args.command == "extract-png-patches":
        create_resized_patches_from_tiff(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            overlap=args.overlap,
            black_threshold=args.black_threshold,
            max_black_ratio=args.max_black_ratio,
            hr_only=args.hr_only,
            skip_sea=args.skip_sea,
            min_sea_ratio=args.min_sea_ratio,
            sea_blue_red_diff=args.sea_blue_red_diff,
            sea_blue_green_diff=args.sea_blue_green_diff,
            sea_max_std=args.sea_max_std,
            workers=args.workers,
            device=args.device,
            resize_batch_size=args.resize_batch_size,
            recursive=args.recursive,
            include_name=args.include_name,
            exclude_name=args.exclude_name,
        )
    elif args.command == "extract-tiff-patches":
        create_resized_tiff_patches_from_tiff(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            overlap=args.overlap,
            black_threshold=args.black_threshold,
            max_black_ratio=args.max_black_ratio,
            sub_dir=args.sub_dir,
            image_type=args.image_type,
            hr_only=args.hr_only,
            skip_sea=args.skip_sea,
            min_sea_ratio=args.min_sea_ratio,
            sea_blue_red_diff=args.sea_blue_red_diff,
            sea_blue_green_diff=args.sea_blue_green_diff,
            sea_max_std=args.sea_max_std,
            device=args.device,
            resize_batch_size=args.resize_batch_size,
            workers=args.workers,
            recursive=args.recursive,
            include_name=args.include_name,
            exclude_name=args.exclude_name,
            band_mode=args.band_mode,
        )
    elif args.command == "sentinel-geopatches":
        create_sentinel_geopatches(
            sentinel_roots=args.sentinel_root,
            out=args.out,
            sentinel_patch_size=args.sentinel_patch_size,
            sr_patch_size=args.sr_patch_size,
            scale=args.scale,
            patch_overlap=args.patch_overlap,
            max_scl_cloud_ratio=args.max_scl_cloud_ratio,
            black_threshold=args.black_threshold,
            max_black_ratio=args.max_black_ratio,
            skip_sea=args.skip_sea,
            min_sea_ratio=args.min_sea_ratio,
            q_low=args.q_low,
            q_high=args.q_high,
            limit_patches=args.limit_patches,
            limit_dates=args.limit_dates,
            debug_png_count=args.debug_png_count,
        )


if __name__ == "__main__":
    main()
