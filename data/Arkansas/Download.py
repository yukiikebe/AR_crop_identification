import argparse
import calendar
from datetime import date, datetime, timedelta
import json
import os
import multiprocessing
from functools import partial
import time

import ee
import geemap
import numpy as np
from tqdm import tqdm
import rasterio
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

DEFAULT_DATA_ROOT = "/mnt/vhvkhoa_ssd/datasets"
LEGACY_DATA_DIR = "/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/"

REQUIRED_BANDS: dict[str, tuple[int, str]] = {
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


def init_ee(project: str, *, authenticate: bool) -> None:
    if authenticate:
        ee.Authenticate()
    ee.Initialize(project=project)


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

def _run_tag(*, start_day: str, end_day: str, cloud_thresh: float) -> str:
    ct = f"{float(cloud_thresh):g}".replace(".", "p")
    return f"{start_day}_{end_day}_ct{ct}"

def _run_state_path(grid_dir: str, *, start_day: str, end_day: str, cloud_thresh: float) -> str:
    return os.path.join(grid_dir, f"_download_{_run_tag(start_day=start_day, end_day=end_day, cloud_thresh=cloud_thresh)}.json")

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

def _missing_files_for_date(date_dir: str, date_str: str) -> list[str]:
    missing: list[str] = []
    for fn in _expected_filenames_for_date(date_str):
        fp = os.path.join(date_dir, fn)
        if not os.path.isfile(fp) or os.path.getsize(fp) <= 0:
            missing.append(fp)
    return missing

def _ensure_tci_jpg(date_dir: str, date_str: str) -> None:
    """
    Create TCI_<date>.jpg if the three TCI_*.tif bands exist.
    """
    tci_rgb_path = os.path.join(date_dir, f"TCI_{date_str}.jpg")
    if os.path.isfile(tci_rgb_path) and os.path.getsize(tci_rgb_path) > 0:
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

def _maybe_write_first_rgb(date_dir: str, date_str: str) -> bool:
    """
    Preserve legacy behavior: write one 10m_rgb_<date>.tif for the *first* usable date.
    Returns True if the RGB now exists.
    """
    out_fp = os.path.join(date_dir, f"10m_rgb_{date_str}.tif")
    if os.path.isfile(out_fp) and os.path.getsize(out_fp) > 0:
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
) -> None:
    if os.path.isfile(out_fp) and os.path.getsize(out_fp) > 0:
        return

    base, ext = os.path.splitext(out_fp)
    tmp = f"{base}.tmp{ext}"

    last_exc: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
            geemap.ee_export_image(image.select(band), filename=tmp, scale=scale, crs=crs, region=region)
            if os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, out_fp)
                return
        except Exception as exc:
            last_exc = exc
        finally:
            if os.path.isfile(tmp) and (not os.path.isfile(out_fp)):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        if attempt < retries and sleep_s > 0:
            time.sleep(float(sleep_s))

    if last_exc is not None:
        raise RuntimeError(f"Failed to export {band} -> {out_fp} after {retries} attempts: {last_exc}")
    raise RuntimeError(f"Failed to export {band} -> {out_fp} after {retries} attempts.")

def _list_available_dates(collection: ee.ImageCollection) -> list[str]:
    """
    Return unique YYYY-MM-DD dates in the collection using a single server round-trip.
    """
    times = collection.aggregate_array("system:time_start").getInfo() or []
    dates = [datetime.utcfromtimestamp(int(t) / 1000).strftime("%Y-%m-%d") for t in times]
    return sorted(set(dates))


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
    verify: bool = True,
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
    wrote_rgb = False

    for date_str in dates:
        date_dir = os.path.join(save_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)

        _ensure_tci_jpg(date_dir, date_str)
        if not _missing_files_for_date(date_dir, date_str):
            if not wrote_rgb:
                wrote_rgb = _maybe_write_first_rgb(date_dir, date_str) or wrote_rgb
            continue

        next_day = _next_day_str(date_str)
        image = ee.Image(collection.filterDate(date_str, next_day).sort("CLOUDY_PIXEL_PERCENTAGE").first())

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
            )

        _ensure_tci_jpg(date_dir, date_str)
        if not wrote_rgb:
            wrote_rgb = _maybe_write_first_rgb(date_dir, date_str) or wrote_rgb

    missing: list[str] = []
    if verify:
        for d in dates:
            ddir = os.path.join(save_dir, d)
            _ensure_tci_jpg(ddir, d)
            missing.extend(_missing_files_for_date(ddir, d))
    return dates, missing


def init_worker(project: str) -> None:
    """
    Ensure Earth Engine is initialized inside each worker process.
    This avoids issues on platforms/start-methods that don't inherit parent state.
    """
    ee.Initialize(project=project)


def download_grid(
    grid_data: tuple[str, list[list[float]]],
    *,
    start_day: str,
    end_day: str,
    cloud_thresh: float,
    retries: int,
    retry_sleep_s: float,
    verify: bool,
    strict: bool,
    fast_skip: bool,
) -> None:
    grid_dir, grid_corners = grid_data
    os.makedirs(grid_dir, exist_ok=True)
    run_fp = _run_state_path(grid_dir, start_day=start_day, end_day=end_day, cloud_thresh=float(cloud_thresh))
    state = _load_run_state(run_fp) or {}

    if fast_skip and state.get("complete") is True and state.get("dates"):
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

    dates, missing = download_dataset(
        grid_corners,
        start_day,
        end_day,
        grid_dir,
        cloud_thresh=float(cloud_thresh),
        retries=int(retries),
        retry_sleep_s=float(retry_sleep_s),
        verify=bool(verify),
    )

    state = {
        "start_day": start_day,
        "end_day": end_day,
        "cloud_thresh": float(cloud_thresh),
        "dates": dates,
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
    if data_root and os.path.isdir(data_root):
        return os.path.join(data_root, f"AR_{start_year}_raw")

    # Fallback for legacy paths (keeps old behavior if /mnt/... is not available)
    return LEGACY_DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser("Download Sentinel-2 tiles from Google Earth Engine (Arkansas grid)")
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

    ap.add_argument("--data-dir", default=None, help="Output directory (contains <i>_<j>/<YYYY-MM-DD>/...)")
    ap.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="If --data-dir is not set, output to <data-root>/AR_<year>_raw",
    )
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 8)))
    ap.add_argument("--cloud-thresh", type=float, default=20, help="Max CLOUDY_PIXEL_PERCENTAGE to keep")
    ap.add_argument("--retries", type=int, default=3, help="Retries per missing band download")
    ap.add_argument("--retry-sleep-s", type=float, default=1.0, help="Sleep between retries (seconds)")
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

    data_dir = _compute_data_dir(data_dir=args.data_dir, data_root=args.data_root, start_day=start_day)

    try:
        init_ee(args.project, authenticate=args.auth)
    except Exception as exc:
        raise SystemExit(
            f"Earth Engine init failed: {exc}\n"
            f"- If this is your first time, run with --auth once (interactive)\n"
            f"- For cron, authenticate once manually and run without --auth"
        ) from exc

    # Extracting the longitude and latitude of the corners
    roig = DEFAULT_ROIG
    lon_min = min(roig[0][0], roig[1][0])
    lon_max = max(roig[2][0], roig[3][0])
    lat_min = min(roig[1][1], roig[3][1])
    lat_max = max(roig[0][1], roig[2][1])

    # Create linspace for longitude and latitude
    lon_range = np.linspace(lon_min, lon_max, 21)  # 20 intervals, so 21 points
    lat_range = np.linspace(lat_min, lat_max, 21)

    # Function to get the four corners of a grid cell
    def get_grid_corners(i, j, lon_range, lat_range):
        lon1, lon2 = lon_range[i], lon_range[i+1]
        lat1, lat2 = lat_range[j], lat_range[j+1]
        return [
            [lon1, lat2],  # top-left
            [lon1, lat1],  # bottom-left
            [lon2, lat2],  # top-right
            [lon2, lat1],  # bottom-right
        ]

    # Iterate through each grid cell
    grids = []
    for i in range(lon_range.shape[0] - 1):  # 20x20 grid
        for j in range(lat_range.shape[0]-1):
            grid_dir = os.path.join(data_dir, f'{i}_{j}')
            grid_corners = get_grid_corners(i, j, lon_range, lat_range)
            grids.append((grid_dir, grid_corners))

    os.makedirs(data_dir, exist_ok=True)
    worker = partial(
        download_grid,
        start_day=start_day,
        end_day=end_day,
        cloud_thresh=float(args.cloud_thresh),
        retries=int(args.retries),
        retry_sleep_s=float(args.retry_sleep_s),
        verify=bool(args.verify),
        strict=bool(args.strict),
        fast_skip=bool(args.fast_skip),
    )
    with multiprocessing.Pool(processes=args.workers, initializer=init_worker, initargs=(args.project,)) as pool:
        with tqdm(total=len(grids)) as pbar:
            for _ in pool.imap_unordered(worker, grids):
                pbar.update()


if __name__ == "__main__":
    main()
