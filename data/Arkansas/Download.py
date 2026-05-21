import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
import math
import os
import multiprocessing
from functools import partial
import random
import shutil
import time

import ee
import geemap
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.merge import merge
import matplotlib.pyplot as plt


DEFAULT_EE_PROJECT = "satelite-430703"
DEFAULT_ROIG = [
    [-94.7610, 36.6652],
    [-94.7610, 32.8376],
    [-89.5522, 36.6652],
    [-89.5522, 32.8376],
]
DEFAULT_START_DAY = "2023-01-01"
DEFAULT_END_DAY = "2023-12-31"
DEFAULT_MONTH_TAG_DAY = 15
DEFAULT_GRID_TARGET_WIDTH_DEG = 0.25
DEFAULT_GRID_TARGET_HEIGHT_DEG = 0.25
DEFAULT_GRID_MAX_SPLITS_PER_AXIS = 20

DEFAULT_DATA_ROOT = "/mnt/vhvkhoa_ssd/datasets"
LEGACY_DATA_DIR = "/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/"

ALL_REQUIRED_BANDS: dict[str, tuple[int, str]] = {
    # band -> (scale_meters, ext)
    "B2": (10, "tif"),
    "B3": (10, "tif"),
    "B4": (10, "tif"),
    "B8": (10, "tif"),
    "B5": (20, "tif"),
    "B6": (20, "tif"),
    "B7": (20, "tif"),
    "B8A": (20, "tif"),
    "B11": (20, "tif"),
    "B12": (20, "tif"),
    "SCL": (20, "tif"),
    "TCI_R": (10, "tif"),
    "TCI_G": (10, "tif"),
    "TCI_B": (10, "tif"),
    # Derived locally from TCI_*.tif
    "TCI": (0, "jpg"),
}
MODEL1_RGB_SCL_BANDS = ("B2", "B3", "B4", "SCL")
BAND_PRESETS: dict[str, tuple[str, ...]] = {
    "all": tuple(ALL_REQUIRED_BANDS.keys()),
    "rgb_scl": MODEL1_RGB_SCL_BANDS,
}
ACTIVE_BAND_PRESET = "all"
REQUIRED_BANDS: dict[str, tuple[int, str]] = dict(ALL_REQUIRED_BANDS)

STATEWIDE_META_NAME = "_statewide_meta.json"
RATE_LIMIT_PATTERNS = (
    "429",
    "too many requests",
    "rate limit",
    "rate exceeded",
    "quota exceeded",
    "user rate limit exceeded",
    "resource exhausted",
    "throttl",
)

REMOTE_DOWNLOAD_PATTERNS = (
    "an error occurred while downloading",
    "export completed without producing a non-empty file",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
    "temporarily unavailable",
)


def init_ee(project: str, *, authenticate: bool) -> None:
    if authenticate:
        ee.Authenticate()
    ee.Initialize(project=project)


def configure_band_preset(band_preset: str) -> None:
    global ACTIVE_BAND_PRESET
    global REQUIRED_BANDS

    if band_preset not in BAND_PRESETS:
        raise ValueError(f"Unsupported band preset: {band_preset}")
    ACTIVE_BAND_PRESET = str(band_preset)
    REQUIRED_BANDS = {
        band: ALL_REQUIRED_BANDS[band]
        for band in BAND_PRESETS[ACTIVE_BAND_PRESET]
    }


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def month_range(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def previous_month_range(today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    first_this = date(today.year, today.month, 1)
    last_prev = first_this - timedelta(days=1)
    start_prev = date(last_prev.year, last_prev.month, 1)
    return start_prev.strftime("%Y-%m-%d"), last_prev.strftime("%Y-%m-%d")


def _bbox_from_roig(roig: list[list[float]]) -> tuple[float, float, float, float]:
    lon_min = min(roig[0][0], roig[1][0])
    lon_max = max(roig[2][0], roig[3][0])
    lat_min = min(roig[1][1], roig[3][1])
    lat_max = max(roig[0][1], roig[2][1])
    return float(lon_min), float(lat_min), float(lon_max), float(lat_max)


def _roig_from_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[list[float]]:
    return [
        [float(min_lon), float(max_lat)],
        [float(min_lon), float(min_lat)],
        [float(max_lon), float(max_lat)],
        [float(max_lon), float(min_lat)],
    ]


def _parse_roi_bbox(value: str | None) -> list[list[float]] | None:
    if value is None:
        return None

    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 4:
        raise SystemExit(
            "--roi-bbox must be four comma-separated numbers: min_lon,min_lat,max_lon,max_lat"
        )

    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(
            "--roi-bbox must be four comma-separated numbers: min_lon,min_lat,max_lon,max_lat"
        ) from exc

    if min_lon >= max_lon:
        raise SystemExit(f"--roi-bbox invalid longitude range: {min_lon} >= {max_lon}")
    if min_lat >= max_lat:
        raise SystemExit(f"--roi-bbox invalid latitude range: {min_lat} >= {max_lat}")
    if min_lon < -180 or max_lon > 180:
        raise SystemExit(f"--roi-bbox longitudes must be within [-180, 180], got {min_lon},{max_lon}")
    if min_lat < -90 or max_lat > 90:
        raise SystemExit(f"--roi-bbox latitudes must be within [-90, 90], got {min_lat},{max_lat}")

    return _roig_from_bbox(min_lon, min_lat, max_lon, max_lat)


def build_arkansas_bbox_geometry(roig: list[list[float]] | None = None) -> ee.Geometry:
    return ee.Geometry.Polygon(roig or DEFAULT_ROIG)


def _month_tag_from_day(day_str: str) -> str:
    dt = _parse_ymd(day_str)
    return f"{dt.year:04d}-{dt.month:02d}"


def _month_pseudo_date(day_str: str, month_tag_day: int) -> str:
    dt = _parse_ymd(day_str)
    max_day = calendar.monthrange(dt.year, dt.month)[1]
    day = max(1, min(int(month_tag_day), max_day))
    return f"{dt.year:04d}-{dt.month:02d}-{day:02d}"


def _iter_month_ranges(start_day: str, end_day: str) -> list[tuple[str, str]]:
    start = _parse_ymd(start_day)
    end = _parse_ymd(end_day)
    if end < start:
        raise SystemExit(f"end_day must be >= start_day, got {start_day}..{end_day}")

    out: list[tuple[str, str]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        month_last = calendar.monthrange(cur.year, cur.month)[1]
        cur_end = date(cur.year, cur.month, month_last)
        part_start = max(start, cur)
        part_end = min(end, cur_end)
        out.append((part_start.strftime("%Y-%m-%d"), part_end.strftime("%Y-%m-%d")))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out

def _run_tag(*, start_day: str, end_day: str, cloud_thresh: float, mask_water: bool = False) -> str:
    ct = f"{float(cloud_thresh):g}".replace(".", "p")
    mw = "mw1" if bool(mask_water) else "mw0"
    return f"{start_day}_{end_day}_ct{ct}_{mw}"

def _run_state_path(grid_dir: str, *, start_day: str, end_day: str, cloud_thresh: float, mask_water: bool = False) -> str:
    return os.path.join(
        grid_dir,
        f"_download_{_run_tag(start_day=start_day, end_day=end_day, cloud_thresh=cloud_thresh, mask_water=mask_water)}.json",
    )


def _exc_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = _exc_message(exc).lower()
    return any(pat in msg for pat in RATE_LIMIT_PATTERNS)


def _is_remote_download_error(exc: Exception) -> bool:
    msg = _exc_message(exc).lower()
    return any(pat in msg for pat in REMOTE_DOWNLOAD_PATTERNS)


def _compute_retry_sleep_s(
    *,
    is_rate_limit: bool,
    attempt: int,
    retry_sleep_s: float,
    backoff_base_s: float,
    backoff_max_s: float,
    retry_jitter_s: float,
) -> float:
    if not is_rate_limit:
        return max(0.0, float(retry_sleep_s))

    base = max(0.0, float(backoff_base_s))
    capped = max(base, float(backoff_max_s))
    expo = base * (2 ** max(0, int(attempt) - 1))
    sleep_s = min(expo, capped)
    jitter = random.uniform(0.0, max(0.0, float(retry_jitter_s)))
    return sleep_s + jitter

def _load_run_state(fp: str) -> dict | None:
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_run_state(fp: str, data: dict) -> None:
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, fp)


def _grid_layout_meta_path(data_dir: str) -> str:
    return os.path.join(data_dir, "_grid_layout.json")


def _load_grid_layout_meta(data_dir: str) -> dict | None:
    return _load_run_state(_grid_layout_meta_path(data_dir))


def _write_grid_layout_meta(data_dir: str, data: dict) -> None:
    _write_run_state(_grid_layout_meta_path(data_dir), data)


def _compute_grid_split_counts(
    *,
    roig: list[list[float]],
    target_width_deg: float,
    target_height_deg: float,
    max_splits_per_axis: int,
) -> tuple[int, int]:
    lon_min, lat_min, lon_max, lat_max = _bbox_from_roig(roig)
    width_deg = max(0.0, float(lon_max) - float(lon_min))
    height_deg = max(0.0, float(lat_max) - float(lat_min))
    cols = max(1, min(int(max_splits_per_axis), int(math.ceil(width_deg / float(target_width_deg)))))
    rows = max(1, min(int(max_splits_per_axis), int(math.ceil(height_deg / float(target_height_deg)))))
    return cols, rows


def _existing_grid_dirs(data_dir: str) -> list[str]:
    out: list[str] = []
    if not os.path.isdir(data_dir):
        return out
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        parts = name.split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            continue
        out.append(name)
    return sorted(out)


def _ensure_grid_layout_meta(
    *,
    data_dir: str,
    roig: list[list[float]],
    cols: int,
    rows: int,
    target_width_deg: float,
    target_height_deg: float,
    max_splits_per_axis: int,
) -> None:
    bbox = [round(v, 6) for v in _bbox_from_roig(roig)]
    desired = {
        "roi_bbox": bbox,
        "grid_cols": int(cols),
        "grid_rows": int(rows),
        "target_width_deg": float(target_width_deg),
        "target_height_deg": float(target_height_deg),
        "max_splits_per_axis": int(max_splits_per_axis),
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    existing = _load_grid_layout_meta(data_dir)
    if existing is not None:
        if (
            int(existing.get("grid_cols", -1)) != int(cols)
            or int(existing.get("grid_rows", -1)) != int(rows)
            or list(existing.get("roi_bbox", [])) != bbox
        ):
            raise SystemExit(
                "Grid layout mismatch for existing download directory. "
                f"Existing={existing.get('grid_cols')}x{existing.get('grid_rows')} "
                f"Requested={cols}x{rows}. Use a new --data-root/--data-dir or clean {data_dir} first."
            )
        desired["created_at"] = existing.get("created_at") or desired["updated_at"]
        _write_grid_layout_meta(data_dir, desired)
        return

    legacy_grid_dirs = _existing_grid_dirs(data_dir)
    if legacy_grid_dirs and (int(cols) != 20 or int(rows) != 20):
        raise SystemExit(
            "Existing grid tiles were found without layout metadata, and the new adaptive layout "
            f"would use {cols}x{rows} instead of the legacy 20x20. "
            f"Use a new --data-root/--data-dir or clean {data_dir} before rerunning."
        )
    desired["created_at"] = desired["updated_at"]
    _write_grid_layout_meta(data_dir, desired)

def _next_day_str(day: str) -> str:
    dt = datetime.strptime(day, "%Y-%m-%d").date() + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def _expected_filenames_for_date(date_str: str) -> list[str]:
    out: list[str] = []
    for band, (_, ext) in REQUIRED_BANDS.items():
        if band == "TCI":
            out.append(f"TCI_{date_str}.{ext}")
        else:
            out.append(f"{band}_{date_str}.{ext}")
    return out


def _expected_filenames_for_month(month_tag: str) -> list[str]:
    out: list[str] = []
    for band, (_, ext) in REQUIRED_BANDS.items():
        if band == "TCI":
            out.append(f"TCI_{month_tag}.{ext}")
        else:
            out.append(f"{band}_{month_tag}.{ext}")
    return out

def _missing_files_for_date(date_dir: str, date_str: str) -> list[str]:
    missing: list[str] = []
    for fn in _expected_filenames_for_date(date_str):
        fp = os.path.join(date_dir, fn)
        if not os.path.isfile(fp) or os.path.getsize(fp) <= 0:
            missing.append(fp)
    return missing


def _missing_files_for_month(month_dir: str, month_tag: str) -> list[str]:
    missing: list[str] = []
    for fn in _expected_filenames_for_month(month_tag):
        fp = os.path.join(month_dir, fn)
        if not os.path.isfile(fp) or os.path.getsize(fp) <= 0:
            missing.append(fp)
    return missing


def _remove_dir_if_exists(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _compute_scl_cloud_ratio(scl_path: str) -> float | None:
    if not os.path.isfile(scl_path) or os.path.getsize(scl_path) <= 0:
        return None
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    valid = scl > 0
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return 1.0
    cloudy = (scl > 7) & valid
    return float(cloudy.sum()) / float(valid_count)


def _compute_rgb_black_ratio(
    *,
    blue_path: str,
    green_path: str,
    red_path: str,
    black_threshold: int,
) -> float | None:
    rgb_paths = [blue_path, green_path, red_path]
    if any((not os.path.isfile(fp)) or os.path.getsize(fp) <= 0 for fp in rgb_paths):
        return None
    with rasterio.open(blue_path) as src:
        blue = src.read(1)
    with rasterio.open(green_path) as src:
        green = src.read(1)
    with rasterio.open(red_path) as src:
        red = src.read(1)
    black = (blue <= black_threshold) & (green <= black_threshold) & (red <= black_threshold)
    return float(black.sum()) / float(black.size)


def _evaluate_date_quality_filters(
    *,
    date_dir: str,
    date_str: str,
    max_scl_cloud_ratio: float | None,
    max_black_ratio: float | None,
    black_threshold: int,
) -> tuple[bool, dict[str, float]]:
    metrics: dict[str, float] = {}

    if max_scl_cloud_ratio is not None:
        scl_ratio = _compute_scl_cloud_ratio(os.path.join(date_dir, f"SCL_{date_str}.tif"))
        if scl_ratio is not None:
            metrics["scl_cloud_ratio"] = float(scl_ratio)
            if scl_ratio > float(max_scl_cloud_ratio):
                return False, metrics

    if max_black_ratio is not None:
        black_ratio = _compute_rgb_black_ratio(
            blue_path=os.path.join(date_dir, f"B2_{date_str}.tif"),
            green_path=os.path.join(date_dir, f"B3_{date_str}.tif"),
            red_path=os.path.join(date_dir, f"B4_{date_str}.tif"),
            black_threshold=int(black_threshold),
        )
        if black_ratio is not None:
            metrics["rgb_black_ratio"] = float(black_ratio)
            if black_ratio > float(max_black_ratio):
                return False, metrics

    return True, metrics

def _ensure_tci_jpg(date_dir: str, date_str: str, *, overwrite: bool = False) -> None:
    """
    Create TCI_<date>.jpg if the three TCI_*.tif bands exist.
    """
    tci_rgb_path = os.path.join(date_dir, f"TCI_{date_str}.jpg")
    if not overwrite and os.path.isfile(tci_rgb_path) and os.path.getsize(tci_rgb_path) > 0:
        return

    tci_r_path = os.path.join(date_dir, f"TCI_R_{date_str}.tif")
    tci_g_path = os.path.join(date_dir, f"TCI_G_{date_str}.tif")
    tci_b_path = os.path.join(date_dir, f"TCI_B_{date_str}.tif")
    if not (os.path.isfile(tci_r_path) and os.path.isfile(tci_g_path) and os.path.isfile(tci_b_path)):
        return

    tmp = tci_rgb_path + ".tmp.jpg"
    save_tci_image(tci_b_path, tci_g_path, tci_r_path, tmp)
    if os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, tci_rgb_path)
    elif os.path.isfile(tmp):
        os.remove(tmp)

def _maybe_write_first_rgb(date_dir: str, date_str: str, *, overwrite: bool = False) -> bool:
    """
    Preserve legacy behavior: write one 10m_rgb_<date>.tif for the *first* usable date.
    Returns True if the RGB now exists.
    """
    out_fp = os.path.join(date_dir, f"10m_rgb_{date_str}.tif")
    if not overwrite and os.path.isfile(out_fp) and os.path.getsize(out_fp) > 0:
        return True
    save_rgb_image(
        red_path=os.path.join(date_dir, f"B4_{date_str}.tif"),
        green_path=os.path.join(date_dir, f"B3_{date_str}.tif"),
        blue_path=os.path.join(date_dir, f"B2_{date_str}.tif"),
        output_path=out_fp,
    )
    return os.path.isfile(out_fp) and os.path.getsize(out_fp) > 0

def _export_with_retries(
    *,
    image: ee.Image,
    band: str,
    out_fp: str,
    scale: int,
    region: ee.Geometry,
    crs: str,
    retries: int,
    sleep_s: float,
    rate_limit_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    retry_jitter_s: float,
    overwrite: bool = False,
) -> None:
    if not overwrite and os.path.isfile(out_fp) and os.path.getsize(out_fp) > 0:
        return

    base, ext = os.path.splitext(out_fp)
    tmp = f"{base}.tmp{ext}"

    last_exc: Exception | None = None
    generic_attempt = 0
    rate_limit_attempt = 0
    overall_attempt = 0
    while True:
        overall_attempt += 1
        is_rate_limit = False
        use_backoff = False
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
            geemap.ee_export_image(image.select(band), filename=tmp, scale=scale, crs=crs, region=region)
            if os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, out_fp)
                return
            raise RuntimeError(f"Export completed without producing a non-empty file: {tmp}")
        except Exception as exc:
            last_exc = exc
            is_rate_limit = _is_rate_limit_error(exc)
            is_remote_download = _is_remote_download_error(exc)
            use_backoff = is_rate_limit or is_remote_download
            if is_rate_limit or is_remote_download:
                rate_limit_attempt += 1
                attempt = int(rate_limit_attempt)
                budget = max(1, int(rate_limit_retries))
                reason = "rate_limit" if is_rate_limit else "download"
            else:
                generic_attempt += 1
                attempt = int(generic_attempt)
                budget = max(1, int(retries))
                reason = "transient"
        finally:
            if os.path.isfile(tmp) and (not os.path.isfile(out_fp)):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        if attempt >= budget:
            break

        delay_s = _compute_retry_sleep_s(
            is_rate_limit=bool(use_backoff),
            attempt=int(attempt),
            retry_sleep_s=float(sleep_s),
            backoff_base_s=float(backoff_base_s),
            backoff_max_s=float(backoff_max_s),
            retry_jitter_s=float(retry_jitter_s),
        )
        print(
            f"[download-retry] band={band} attempt={overall_attempt} kind={reason} "
            f"budget={attempt}/{budget} sleep_s={delay_s:.2f} file={out_fp} "
            f"error={_exc_message(last_exc)}",
            flush=True,
        )
        if delay_s > 0:
            time.sleep(float(delay_s))

    if last_exc is not None:
        if _is_rate_limit_error(last_exc):
            kind = "rate-limit"
        elif _is_remote_download_error(last_exc):
            kind = "download"
        else:
            kind = "transient"
        raise RuntimeError(
            f"Failed to export {band} -> {out_fp} after {overall_attempt} attempts "
            f"({kind} budget exhausted): {last_exc}"
        )
    raise RuntimeError(f"Failed to export {band} -> {out_fp} after {overall_attempt} attempts.")

def _list_available_dates(collection: ee.ImageCollection) -> list[str]:
    """
    Return unique YYYY-MM-DD dates in the collection using a single server round-trip.
    """
    times = collection.aggregate_array("system:time_start").getInfo() or []
    dates = [datetime.utcfromtimestamp(int(t) / 1000).strftime("%Y-%m-%d") for t in times]
    return sorted(set(dates))


def _list_available_timestamps(collection: ee.ImageCollection) -> list[int]:
    times = collection.aggregate_array("system:time_start").getInfo() or []
    out: list[int] = []
    for t in times:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out


def mask_s2_clouds_scl(image: ee.Image, *, mask_water: bool = False) -> ee.Image:
    scl = image.select("SCL")
    keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(7))
    if not bool(mask_water):
        keep = keep.Or(scl.eq(6))
    return image.updateMask(keep)


def mask_s2_water_scl(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    return image.updateMask(scl.neq(6))


def _statewide_collection(*, start_day: str, end_day: str, roi: ee.Geometry, cloud_thresh: float) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_day, _next_day_str(end_day))
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )


def build_monthly_statewide_composite(
    start_day: str,
    end_day: str,
    roi: ee.Geometry,
    cloud_thresh: float,
    mask_water: bool = False,
) -> tuple[ee.Image | None, list[str], int]:
    collection = _statewide_collection(
        start_day=start_day,
        end_day=end_day,
        roi=roi,
        cloud_thresh=float(cloud_thresh),
    )
    source_dates = _list_available_dates(collection)
    source_image_count = len(_list_available_timestamps(collection))
    if source_image_count <= 0:
        return None, source_dates, 0

    def _mask(image: ee.Image) -> ee.Image:
        return mask_s2_clouds_scl(image, mask_water=bool(mask_water))

    masked = collection.map(_mask)
    pieces: list[ee.Image] = []
    for band in [b for b in REQUIRED_BANDS.keys() if b != "TCI"]:
        reduced = masked.select([band]).median().rename(band)
        if band == "SCL" or band.startswith("TCI_"):
            reduced = reduced.round().toUint8()
        else:
            reduced = reduced.round().toUint16()
        pieces.append(reduced)

    composite = pieces[0]
    for piece in pieces[1:]:
        composite = composite.addBands(piece)
    return composite.clip(roi), source_dates, source_image_count


def save_tci_image(blue_band_path, green_band_path, red_band_path, output_path):
    # Open the TIF files
    if not (os.path.isfile(blue_band_path) and os.path.isfile(green_band_path) and os.path.isfile(red_band_path)):
        return

    with rasterio.open(blue_band_path) as blue_band:
        blue = blue_band.read(1)
    with rasterio.open(green_band_path) as green_band:
        green = green_band.read(1)
    with rasterio.open(red_band_path) as red_band:
        red = red_band.read(1)

    # Stack the bands
    stacked_data = np.stack((red, green, blue), axis=-1)
    plt.imsave(output_path, stacked_data)


def save_rgb_image(red_path, green_path, blue_path, output_path):
    if not all([
        os.path.exists(red_path),
        os.path.exists(green_path),
        os.path.exists(blue_path),
    ]):
        return

    with rasterio.open(red_path) as red_file:
        red = red_file.read(1)
        profile = red_file.profile

    with rasterio.open(green_path) as green_file:
        green = green_file.read(1)

    with rasterio.open(blue_path) as blue_file:
        blue = blue_file.read(1)
    
    stacked_data = np.stack((red, green, blue))
    profile.update(count=3)

    with rasterio.open(output_path, 'w', **profile) as dst_file:
        dst_file.write(stacked_data)


def download_dataset(
    roig,
    start_day,
    end_day,
    save_dir,
    *,
    cloud_thresh: float = 20,
    retries: int = 3,
    retry_sleep_s: float = 1.0,
    rate_limit_retries: int = 8,
    backoff_base_s: float = 5.0,
    backoff_max_s: float = 120.0,
    retry_jitter_s: float = 1.0,
    verify: bool = True,
    mask_water: bool = False,
    max_scl_cloud_ratio: float | None = None,
    max_black_ratio: float | None = None,
    black_threshold: int = 0,
):
    """
    Download all required bands for the requested period.

    Speed improvements:
    - Avoid per-image getInfo() calls; fetch all timestamps in one call.
    - Skip any day whose files are already complete.
    Reliability improvements:
    - Export to a temp file then atomically rename.
    - Retry exports; optionally verify completeness at the end.
    """
    roi = ee.Geometry.Polygon(roig)
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    collection = (
        collection.filterDate(start_day, end_day)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )

    dates = _list_available_dates(collection)
    kept_dates: list[str] = []
    filtered_dates: list[dict[str, float | str]] = []
    wrote_rgb = False
    overwrite_outputs = bool(mask_water)

    for date_str in dates:
        date_dir = os.path.join(save_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)

        _ensure_tci_jpg(date_dir, date_str, overwrite=overwrite_outputs)
        if (not overwrite_outputs) and (not _missing_files_for_date(date_dir, date_str)):
            if not wrote_rgb:
                wrote_rgb = _maybe_write_first_rgb(date_dir, date_str, overwrite=overwrite_outputs) or wrote_rgb
            continue

        next_day = _next_day_str(date_str)
        image = ee.Image(collection.filterDate(date_str, next_day).sort("CLOUDY_PIXEL_PERCENTAGE").first())
        if bool(mask_water):
            image = mask_s2_water_scl(image)

        for band, (scale, ext) in REQUIRED_BANDS.items():
            if band == "TCI":
                continue
            out_fp = os.path.join(date_dir, f"{band}_{date_str}.{ext}")
            _export_with_retries(
                image=image,
                band=band,
                out_fp=out_fp,
                scale=int(scale),
                region=roi,
                crs="EPSG:3857",
                retries=int(retries),
                sleep_s=float(retry_sleep_s),
                rate_limit_retries=int(rate_limit_retries),
                backoff_base_s=float(backoff_base_s),
                backoff_max_s=float(backoff_max_s),
                retry_jitter_s=float(retry_jitter_s),
                overwrite=overwrite_outputs,
            )

        keep_date, metrics = _evaluate_date_quality_filters(
            date_dir=date_dir,
            date_str=date_str,
            max_scl_cloud_ratio=max_scl_cloud_ratio,
            max_black_ratio=max_black_ratio,
            black_threshold=int(black_threshold),
        )
        if not keep_date:
            _remove_dir_if_exists(date_dir)
            filtered_dates.append({"date": date_str, **metrics})
            continue

        _ensure_tci_jpg(date_dir, date_str, overwrite=overwrite_outputs)
        if not wrote_rgb:
            wrote_rgb = _maybe_write_first_rgb(date_dir, date_str, overwrite=overwrite_outputs) or wrote_rgb
        kept_dates.append(date_str)

    missing: list[str] = []
    if verify:
        for d in kept_dates:
            ddir = os.path.join(save_dir, d)
            _ensure_tci_jpg(ddir, d, overwrite=overwrite_outputs)
            missing.extend(_missing_files_for_date(ddir, d))
    return kept_dates, missing, filtered_dates


def init_worker(project: str, band_preset: str) -> None:
    """
    Ensure Earth Engine is initialized inside each worker process.
    This avoids issues on platforms/start-methods that don't inherit parent state.
    """
    configure_band_preset(str(band_preset))
    ee.Initialize(project=project)


def download_grid(
    grid_data: tuple[str, list[list[float]]],
    *,
    start_day: str,
    end_day: str,
    cloud_thresh: float,
    retries: int,
    retry_sleep_s: float,
    rate_limit_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    retry_jitter_s: float,
    verify: bool,
    strict: bool,
    fast_skip: bool,
    mask_water: bool,
    max_scl_cloud_ratio: float | None,
    max_black_ratio: float | None,
    black_threshold: int,
) -> None:
    grid_dir, grid_corners = grid_data
    os.makedirs(grid_dir, exist_ok=True)
    run_fp = _run_state_path(
        grid_dir,
        start_day=start_day,
        end_day=end_day,
        cloud_thresh=float(cloud_thresh),
        mask_water=bool(mask_water),
    )
    state = _load_run_state(run_fp) or {}

    if (
        fast_skip
        and state.get("complete") is True
        and state.get("dates")
        and bool(state.get("mask_water", False)) == bool(mask_water)
        and state.get("max_scl_cloud_ratio") == (
            None if max_scl_cloud_ratio is None else float(max_scl_cloud_ratio)
        )
        and state.get("max_black_ratio") == (
            None if max_black_ratio is None else float(max_black_ratio)
        )
        and int(state.get("black_threshold", 0)) == int(black_threshold)
    ):
        dates = [str(d) for d in state.get("dates") or []]
        missing: list[str] = []
        for d in dates:
            date_dir = os.path.join(grid_dir, d)
            _ensure_tci_jpg(date_dir, d)
            missing.extend(_missing_files_for_date(date_dir, d))
        if not missing:
            return
        state["complete"] = False
        state["missing"] = missing
        state["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_run_state(run_fp, state)

    dates, missing, filtered_dates = download_dataset(
        grid_corners,
        start_day,
        end_day,
        grid_dir,
        cloud_thresh=float(cloud_thresh),
        retries=int(retries),
        retry_sleep_s=float(retry_sleep_s),
        rate_limit_retries=int(rate_limit_retries),
        backoff_base_s=float(backoff_base_s),
        backoff_max_s=float(backoff_max_s),
        retry_jitter_s=float(retry_jitter_s),
        verify=bool(verify),
        mask_water=bool(mask_water),
        max_scl_cloud_ratio=None if max_scl_cloud_ratio is None else float(max_scl_cloud_ratio),
        max_black_ratio=None if max_black_ratio is None else float(max_black_ratio),
        black_threshold=int(black_threshold),
    )

    state = {
        "start_day": start_day,
        "end_day": end_day,
        "band_preset": ACTIVE_BAND_PRESET,
        "bands": [b for b in REQUIRED_BANDS.keys() if b != "TCI"],
        "cloud_thresh": float(cloud_thresh),
        "mask_water": bool(mask_water),
        "max_scl_cloud_ratio": None if max_scl_cloud_ratio is None else float(max_scl_cloud_ratio),
        "max_black_ratio": None if max_black_ratio is None else float(max_black_ratio),
        "black_threshold": int(black_threshold),
        "retry_sleep_s": float(retry_sleep_s),
        "rate_limit_retries": int(rate_limit_retries),
        "backoff_base_s": float(backoff_base_s),
        "backoff_max_s": float(backoff_max_s),
        "retry_jitter_s": float(retry_jitter_s),
        "dates": dates,
        "filtered_dates": filtered_dates,
        "complete": len(missing) == 0,
        "missing": missing,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_run_state(run_fp, state)

    if strict and missing:
        raise RuntimeError(f"Grid {os.path.basename(grid_dir)} incomplete: {len(missing)} missing files (see {run_fp}).")


def _compute_data_dir(*, data_dir: str | None, data_root: str | None, start_day: str) -> str:
    if data_dir:
        return data_dir

    start_year = _parse_ymd(start_day).year
    # Honor an explicitly provided data_root even if it does not exist yet;
    # main() creates the output directory later.
    if data_root:
        return os.path.join(data_root, f"AR_{start_year}_raw")

    # Fallback for legacy paths (keeps old behavior if /mnt/... is not available)
    return LEGACY_DATA_DIR


def _compute_statewide_dir(*, statewide_dir: str | None, data_root: str | None, start_day: str) -> str:
    if statewide_dir:
        return statewide_dir

    start_year = _parse_ymd(start_day).year
    if data_root:
        return os.path.join(data_root, f"AR_{start_year}_statewide")

    return os.path.join(LEGACY_DATA_DIR, "..", f"AR_{start_year}_statewide")


def export_statewide_month(
    *,
    start_day: str,
    end_day: str,
    statewide_root: str,
    roig: list[list[float]] | None,
    cloud_thresh: float,
    retries: int,
    retry_sleep_s: float,
    rate_limit_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    retry_jitter_s: float,
    verify: bool,
    strict: bool,
    fast_skip: bool,
    month_tag_day: int,
    statewide_split: int,
    workers: int,
    mask_water: bool,
) -> dict:
    month_tag = _month_tag_from_day(start_day)
    out_dir = os.path.join(statewide_root, month_tag)
    os.makedirs(out_dir, exist_ok=True)
    meta_fp = os.path.join(out_dir, STATEWIDE_META_NAME)
    existing = _load_run_state(meta_fp) or {}
    overwrite_outputs = bool(mask_water)
    roi_type = "custom_bbox" if roig is not None else "arkansas_bbox"
    roi_corners = roig or DEFAULT_ROIG
    roi_bbox = [round(v, 6) for v in _bbox_from_roig(roi_corners)]

    if fast_skip and existing.get("complete") is True and bool(existing.get("mask_water", False)) == bool(mask_water):
        missing = _missing_files_for_month(out_dir, month_tag)
        if not missing:
            return existing
        existing["complete"] = False
        existing["missing_files"] = missing
        existing["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_run_state(meta_fp, existing)

    roi = build_arkansas_bbox_geometry(roi_corners)
    composite, source_dates, source_image_count = build_monthly_statewide_composite(
        start_day=start_day,
        end_day=end_day,
        roi=roi,
        cloud_thresh=float(cloud_thresh),
        mask_water=bool(mask_water),
    )

    missing: list[str]
    if composite is None or source_image_count <= 0:
        missing = [os.path.join(out_dir, fn) for fn in _expected_filenames_for_month(month_tag)]
        state = {
            "layout": "statewide",
            "composite": "monthly_median",
            "start_day": start_day,
            "end_day": end_day,
            "month": int(_parse_ymd(start_day).month),
            "year": int(_parse_ymd(start_day).year),
            "roi_type": roi_type,
            "roi_bbox": roi_bbox,
            "cloud_thresh": float(cloud_thresh),
            "mask_water": bool(mask_water),
            "retry_sleep_s": float(retry_sleep_s),
            "rate_limit_retries": int(rate_limit_retries),
            "backoff_base_s": float(backoff_base_s),
            "backoff_max_s": float(backoff_max_s),
            "retry_jitter_s": float(retry_jitter_s),
            "source_image_count": 0,
            "source_dates": source_dates,
            "band_preset": ACTIVE_BAND_PRESET,
            "bands": [b for b in REQUIRED_BANDS.keys() if b != "TCI"],
            "crs": "EPSG:3857",
            "missing_files": missing,
            "complete": False,
            "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scl_note": "SCL is approximated by monthly masked median and rounded to uint8.",
            "month_tag_day": int(month_tag_day),
            "retile_pseudo_date": _month_pseudo_date(start_day, int(month_tag_day)),
            "statewide_split": int(statewide_split),
            "export_workers": int(max(1, workers)),
        }
        _write_run_state(meta_fp, state)
        if strict:
            raise RuntimeError(f"No statewide Sentinel-2 imagery found for {month_tag} (see {meta_fp}).")
        return state

    subregions = _statewide_subregions(int(statewide_split), roig=roi_corners)
    tile_root = os.path.join(out_dir, "_statewide_tiles")
    export_workers = int(max(1, workers))
    for band, (scale, ext) in REQUIRED_BANDS.items():
        if band == "TCI":
            continue
        out_fp = os.path.join(out_dir, f"{band}_{month_tag}.{ext}")
        if (not overwrite_outputs) and os.path.isfile(out_fp) and os.path.getsize(out_fp) > 0:
            continue
        if len(subregions) == 1:
            _export_with_retries(
                image=composite,
                band=band,
                out_fp=out_fp,
                scale=int(scale),
                region=roi,
                crs="EPSG:3857",
                retries=int(retries),
                sleep_s=float(retry_sleep_s),
                rate_limit_retries=int(rate_limit_retries),
                backoff_base_s=float(backoff_base_s),
                backoff_max_s=float(backoff_max_s),
                retry_jitter_s=float(retry_jitter_s),
                overwrite=overwrite_outputs,
            )
            continue

        band_tile_dir = os.path.join(tile_root, band)
        os.makedirs(band_tile_dir, exist_ok=True)
        band_tile_files: list[str] = []
        jobs: list[tuple[str, list[list[float]]]] = []
        for idx, sub_roig in enumerate(subregions):
            tile_fp = os.path.join(band_tile_dir, f"{band}_{month_tag}_{idx:03d}.tif")
            band_tile_files.append(tile_fp)
            jobs.append((tile_fp, sub_roig))

        def _run_one(job: tuple[str, list[list[float]]]) -> str:
            tile_fp_i, sub_roig_i = job
            _export_with_retries(
                image=composite,
                band=band,
                out_fp=tile_fp_i,
                scale=int(scale),
                region=ee.Geometry.Polygon(sub_roig_i),
                crs="EPSG:3857",
                retries=int(retries),
                sleep_s=float(retry_sleep_s),
                rate_limit_retries=int(rate_limit_retries),
                backoff_base_s=float(backoff_base_s),
                backoff_max_s=float(backoff_max_s),
                retry_jitter_s=float(retry_jitter_s),
                overwrite=overwrite_outputs,
            )
            return tile_fp_i

        if export_workers <= 1 or len(jobs) <= 1:
            for job in jobs:
                _run_one(job)
        else:
            with ThreadPoolExecutor(max_workers=int(export_workers)) as ex:
                futures = [ex.submit(_run_one, job) for job in jobs]
                for fut in as_completed(futures):
                    fut.result()
        _merge_export_tiles(band_tile_files, out_fp)

    _ensure_tci_jpg(out_dir, month_tag, overwrite=overwrite_outputs)
    missing = _missing_files_for_month(out_dir, month_tag) if verify else []
    state = {
        "layout": "statewide",
        "composite": "monthly_median",
        "start_day": start_day,
        "end_day": end_day,
        "month": int(_parse_ymd(start_day).month),
        "year": int(_parse_ymd(start_day).year),
        "roi_type": roi_type,
        "roi_bbox": roi_bbox,
        "cloud_thresh": float(cloud_thresh),
        "mask_water": bool(mask_water),
        "retry_sleep_s": float(retry_sleep_s),
        "rate_limit_retries": int(rate_limit_retries),
        "backoff_base_s": float(backoff_base_s),
        "backoff_max_s": float(backoff_max_s),
        "retry_jitter_s": float(retry_jitter_s),
        "source_image_count": int(source_image_count),
        "source_dates": source_dates,
        "band_preset": ACTIVE_BAND_PRESET,
        "bands": [b for b in REQUIRED_BANDS.keys() if b != "TCI"],
        "crs": "EPSG:3857",
        "missing_files": missing,
        "complete": len(missing) == 0,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scl_note": "SCL is approximated by monthly masked median and rounded to uint8.",
        "month_tag_day": int(month_tag_day),
        "retile_pseudo_date": _month_pseudo_date(start_day, int(month_tag_day)),
        "statewide_split": int(statewide_split),
        "export_workers": int(max(1, workers)),
    }
    _write_run_state(meta_fp, state)
    if strict and missing:
        raise RuntimeError(f"Statewide month {month_tag} incomplete: {len(missing)} missing files (see {meta_fp}).")
    return state


def _grid_corners(i: int, j: int, lon_range: np.ndarray, lat_range: np.ndarray) -> list[list[float]]:
    lon1, lon2 = lon_range[i], lon_range[i + 1]
    lat1, lat2 = lat_range[j], lat_range[j + 1]
    return [
        [lon1, lat2],
        [lon1, lat1],
        [lon2, lat2],
        [lon2, lat1],
    ]


def _statewide_subregions(split_n: int, *, roig: list[list[float]] | None = None) -> list[list[list[float]]]:
    roi_corners = roig or DEFAULT_ROIG
    if int(split_n) <= 1:
        return [roi_corners]
    lon_min, lat_min, lon_max, lat_max = _bbox_from_roig(roi_corners)
    lon_range = np.linspace(lon_min, lon_max, int(split_n) + 1)
    lat_range = np.linspace(lat_min, lat_max, int(split_n) + 1)
    out: list[list[list[float]]] = []
    for i in range(int(split_n)):
        for j in range(int(split_n)):
            out.append(_grid_corners(i, j, lon_range, lat_range))
    return out


def _merge_export_tiles(tile_files: list[str], out_fp: str) -> None:
    datasets = []
    try:
        for fp in tile_files:
            if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                datasets.append(rasterio.open(fp))
        if not datasets:
            raise RuntimeError(f"No readable export tiles to merge for {out_fp}")

        merged, out_transform = merge(datasets, nodata=0)
        profile = dict(datasets[0].profile)
        profile.update(
            driver="GTiff",
            height=int(merged.shape[1]),
            width=int(merged.shape[2]),
            count=int(merged.shape[0]),
            transform=out_transform,
            compress="DEFLATE",
        )
        tmp = out_fp + ".tmp"
        if os.path.isfile(tmp):
            os.remove(tmp)
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(merged)
        os.replace(tmp, out_fp)
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser("Download Sentinel-2 data from Google Earth Engine (Arkansas grid or statewide)")
    ap.add_argument("--project", default=DEFAULT_EE_PROJECT, help="Earth Engine project id")
    ap.add_argument(
        "--auth",
        action="store_true",
        help="Run ee.Authenticate() before ee.Initialize() (interactive). Not recommended for cron.",
    )

    date_grp = ap.add_mutually_exclusive_group()
    date_grp.add_argument("--previous-month", action="store_true", help="Download the previous calendar month")
    date_grp.add_argument("--year", type=int, help="Year (use with --month or --whole-year)")

    ap.add_argument("--month", type=int, choices=range(1, 13), help="Month (1-12). Requires --year.")
    ap.add_argument("--whole-year", action="store_true", help="If set with --year, downloads Jan..Dec.")
    ap.add_argument("--start-day", default=None, help="Start day YYYY-MM-DD (overrides year/month)")
    ap.add_argument("--end-day", default=None, help="End day YYYY-MM-DD (overrides year/month)")

    ap.add_argument("--layout", choices=["grid", "statewide"], default="grid", help="Output layout.")
    ap.add_argument(
        "--composite",
        choices=["none", "monthly"],
        default="none",
        help="Composite mode. Use 'monthly' with --layout statewide.",
    )
    ap.add_argument("--data-dir", default=None, help="Grid output directory (contains <i>_<j>/<YYYY-MM-DD>/...)")
    ap.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="If --data-dir/--statewide-dir is not set, output below <data-root>.",
    )
    ap.add_argument("--statewide-dir", default=None, help="Statewide output root (contains <YYYY-MM>/...)")
    ap.add_argument(
        "--roi-bbox",
        default=None,
        help="Optional ROI bbox override as min_lon,min_lat,max_lon,max_lat. Defaults to the Arkansas bbox.",
    )
    ap.add_argument(
        "--month-tag-day",
        type=int,
        default=DEFAULT_MONTH_TAG_DAY,
        help="Pseudo day used by downstream statewide->grid retile metadata (default: 15).",
    )
    ap.add_argument(
        "--statewide-split",
        type=int,
        default=2,
        help="Split statewide export into NxN subregions before local merge to avoid EE size limits (default: 2).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 8)),
        help="Parallel download workers. Grid mode uses a multiprocessing pool; statewide mode uses parallel export workers.",
    )
    ap.add_argument(
        "--grid-target-width-deg",
        type=float,
        default=DEFAULT_GRID_TARGET_WIDTH_DEG,
        help="Target per-grid tile width in EPSG:4326 degrees for grid layout. Smaller values increase tile count.",
    )
    ap.add_argument(
        "--grid-target-height-deg",
        type=float,
        default=DEFAULT_GRID_TARGET_HEIGHT_DEG,
        help="Target per-grid tile height in EPSG:4326 degrees for grid layout. Smaller values increase tile count.",
    )
    ap.add_argument(
        "--grid-max-splits-per-axis",
        type=int,
        default=DEFAULT_GRID_MAX_SPLITS_PER_AXIS,
        help="Maximum grid splits per axis for grid layout.",
    )
    ap.add_argument("--cloud-thresh", type=float, default=20, help="Max CLOUDY_PIXEL_PERCENTAGE to keep")
    ap.add_argument(
        "--max-scl-cloud-ratio",
        type=float,
        default=None,
        help="Optional local per-date SCL cloud fraction filter (0..1) applied after export.",
    )
    ap.add_argument(
        "--max-black-ratio",
        type=float,
        default=None,
        help="Optional local per-date RGB black-pixel fraction filter (0..1) applied after export.",
    )
    ap.add_argument(
        "--black-threshold",
        type=int,
        default=0,
        help="Pixels with all RGB bands <= this value count as black for --max-black-ratio.",
    )
    ap.add_argument(
        "--band-preset",
        choices=tuple(sorted(BAND_PRESETS.keys())),
        default="all",
        help=(
            "Band selection preset. "
            "'all' keeps the legacy full Sentinel export. "
            "'rgb_scl' downloads only B2/B3/B4/SCL for the Model 1 FastDiffSR dataset."
        ),
    )
    ap.add_argument(
        "--mask-water",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask SCL water pixels (class 6) so water or sea exports as nodata or black in derived previews.",
    )
    ap.add_argument("--retries", type=int, default=3, help="Max attempts for non-rate-limit export failures.")
    ap.add_argument("--retry-sleep-s", type=float, default=1.0, help="Base sleep between non-rate-limit retries (seconds).")
    ap.add_argument(
        "--rate-limit-retries",
        type=int,
        default=8,
        help="Max attempts for 429/rate-limit export failures before giving up.",
    )
    ap.add_argument(
        "--backoff-base-s",
        type=float,
        default=5.0,
        help="Initial automatic backoff for 429/rate-limit retries (seconds).",
    )
    ap.add_argument(
        "--backoff-max-s",
        type=float,
        default=120.0,
        help="Maximum automatic backoff for 429/rate-limit retries (seconds).",
    )
    ap.add_argument(
        "--retry-jitter-s",
        type=float,
        default=1.0,
        help="Random jitter added to automatic 429/rate-limit backoff (seconds).",
    )
    ap.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After downloads, verify all expected files exist (local check).",
    )
    ap.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If verification finds missing files, raise error (non-zero exit).",
    )
    ap.add_argument(
        "--fast-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a previous run marked the grid complete, skip EE calls and only do local verification.",
    )

    args = ap.parse_args()
    configure_band_preset(str(args.band_preset))
    if args.layout == "grid" and args.composite != "none":
        raise SystemExit("--composite monthly is only supported with --layout statewide.")
    if args.layout == "statewide" and args.composite != "monthly":
        raise SystemExit("--layout statewide requires --composite monthly.")
    if int(args.month_tag_day) < 1 or int(args.month_tag_day) > 31:
        raise SystemExit(f"--month-tag-day must be in 1..31, got {args.month_tag_day}")
    if int(args.statewide_split) < 1:
        raise SystemExit(f"--statewide-split must be >= 1, got {args.statewide_split}")
    if int(args.workers) < 1:
        raise SystemExit(f"--workers must be >= 1, got {args.workers}")
    if float(args.grid_target_width_deg) <= 0:
        raise SystemExit(f"--grid-target-width-deg must be > 0, got {args.grid_target_width_deg}")
    if float(args.grid_target_height_deg) <= 0:
        raise SystemExit(f"--grid-target-height-deg must be > 0, got {args.grid_target_height_deg}")
    if int(args.grid_max_splits_per_axis) < 1:
        raise SystemExit(
            f"--grid-max-splits-per-axis must be >= 1, got {args.grid_max_splits_per_axis}"
        )
    if args.max_scl_cloud_ratio is not None and not (0.0 <= float(args.max_scl_cloud_ratio) <= 1.0):
        raise SystemExit(f"--max-scl-cloud-ratio must be in [0, 1], got {args.max_scl_cloud_ratio}")
    if args.max_black_ratio is not None and not (0.0 <= float(args.max_black_ratio) <= 1.0):
        raise SystemExit(f"--max-black-ratio must be in [0, 1], got {args.max_black_ratio}")
    if int(args.black_threshold) < 0:
        raise SystemExit(f"--black-threshold must be >= 0, got {args.black_threshold}")
    roig = _parse_roi_bbox(args.roi_bbox) or DEFAULT_ROIG

    if args.start_day or args.end_day:
        if not (args.start_day and args.end_day):
            raise SystemExit("Provide both --start-day and --end-day.")
        start_day = args.start_day
        end_day = args.end_day
    elif args.previous_month:
        start_day, end_day = previous_month_range()
    elif args.year and args.month:
        start_day, end_day = month_range(args.year, args.month)
    elif args.year and args.whole_year:
        start_day = f"{args.year}-01-01"
        end_day = f"{args.year}-12-31"
    else:
        start_day = DEFAULT_START_DAY
        end_day = DEFAULT_END_DAY

    try:
        init_ee(args.project, authenticate=args.auth)
    except Exception as exc:
        raise SystemExit(
            f"Earth Engine init failed: {exc}\n"
            f"- If this is your first time, run with --auth once (interactive)\n"
            f"- For cron, authenticate once manually and run without --auth"
        ) from exc

    if args.layout == "statewide":
        statewide_dir = _compute_statewide_dir(
            statewide_dir=args.statewide_dir,
            data_root=args.data_root,
            start_day=start_day,
        )
        os.makedirs(statewide_dir, exist_ok=True)
        month_ranges = _iter_month_ranges(start_day, end_day)
        for month_start, month_end in month_ranges:
            state = export_statewide_month(
                start_day=month_start,
                end_day=month_end,
                statewide_root=statewide_dir,
                roig=roig,
                cloud_thresh=float(args.cloud_thresh),
                retries=int(args.retries),
                retry_sleep_s=float(args.retry_sleep_s),
                rate_limit_retries=int(args.rate_limit_retries),
                backoff_base_s=float(args.backoff_base_s),
                backoff_max_s=float(args.backoff_max_s),
                retry_jitter_s=float(args.retry_jitter_s),
                verify=bool(args.verify),
                strict=bool(args.strict),
                fast_skip=bool(args.fast_skip),
                month_tag_day=int(args.month_tag_day),
                statewide_split=int(args.statewide_split),
                workers=int(args.workers),
                mask_water=bool(args.mask_water),
            )
            print(
                f"Statewide {state['year']:04d}-{state['month']:02d}: "
                f"source_images={state['source_image_count']} complete={state['complete']} "
                f"dir={os.path.join(statewide_dir, _month_tag_from_day(month_start))}"
            )
        return

    data_dir = _compute_data_dir(data_dir=args.data_dir, data_root=args.data_root, start_day=start_day)

    lon_min = min(roig[0][0], roig[1][0])
    lon_max = max(roig[2][0], roig[3][0])
    lat_min = min(roig[1][1], roig[3][1])
    lat_max = max(roig[0][1], roig[2][1])

    grid_cols, grid_rows = _compute_grid_split_counts(
        roig=roig,
        target_width_deg=float(args.grid_target_width_deg),
        target_height_deg=float(args.grid_target_height_deg),
        max_splits_per_axis=int(args.grid_max_splits_per_axis),
    )

    os.makedirs(data_dir, exist_ok=True)
    _ensure_grid_layout_meta(
        data_dir=data_dir,
        roig=roig,
        cols=int(grid_cols),
        rows=int(grid_rows),
        target_width_deg=float(args.grid_target_width_deg),
        target_height_deg=float(args.grid_target_height_deg),
        max_splits_per_axis=int(args.grid_max_splits_per_axis),
    )

    lon_range = np.linspace(lon_min, lon_max, int(grid_cols) + 1)
    lat_range = np.linspace(lat_min, lat_max, int(grid_rows) + 1)
    grids = []
    for i in range(lon_range.shape[0] - 1):
        for j in range(lat_range.shape[0] - 1):
            grid_dir = os.path.join(data_dir, f"{i}_{j}")
            grid_corners = _grid_corners(i, j, lon_range, lat_range)
            grids.append((grid_dir, grid_corners))

    worker = partial(
        download_grid,
        start_day=start_day,
        end_day=end_day,
        cloud_thresh=float(args.cloud_thresh),
        retries=int(args.retries),
        retry_sleep_s=float(args.retry_sleep_s),
        rate_limit_retries=int(args.rate_limit_retries),
        backoff_base_s=float(args.backoff_base_s),
        backoff_max_s=float(args.backoff_max_s),
        retry_jitter_s=float(args.retry_jitter_s),
        verify=bool(args.verify),
        strict=bool(args.strict),
        fast_skip=bool(args.fast_skip),
        mask_water=bool(args.mask_water),
        max_scl_cloud_ratio=None if args.max_scl_cloud_ratio is None else float(args.max_scl_cloud_ratio),
        max_black_ratio=None if args.max_black_ratio is None else float(args.max_black_ratio),
        black_threshold=int(args.black_threshold),
    )
    with multiprocessing.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.project, args.band_preset),
    ) as pool:
        with tqdm(total=len(grids)) as pbar:
            for _ in pool.imap_unordered(worker, grids):
                pbar.update()


if __name__ == "__main__":
    main()
