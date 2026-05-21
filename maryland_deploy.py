import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import ar_deploy as _base


APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _bounds_to_roig(*, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> List[List[float]]:
    lon1 = min(float(lon_min), float(lon_max))
    lon2 = max(float(lon_min), float(lon_max))
    lat1 = min(float(lat_min), float(lat_max))
    lat2 = max(float(lat_min), float(lat_max))
    return [
        [lon1, lat2],
        [lon1, lat1],
        [lon2, lat2],
        [lon2, lat1],
    ]


ROI_PRESETS: Dict[str, List[List[float]]] = {
    "md_focus": _bounds_to_roig(lon_min=-76.78, lat_min=38.07, lon_max=-76.36, lat_max=38.32),
    "md_large": _bounds_to_roig(lon_min=-76.745771, lat_min=38.0970323, lon_max=-76.25462, lat_max=38.616701),
}
DEFAULT_ROI_PRESET = "md_large"
FASTDIFFSR_DEFAULT_CONFIG = "FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_maryland.json"


def _resolve_roig(args) -> List[List[float]]:
    roi_bounds = getattr(args, "roi_bounds", None)
    if roi_bounds:
        return _bounds_to_roig(
            lon_min=float(roi_bounds[0]),
            lat_min=float(roi_bounds[1]),
            lon_max=float(roi_bounds[2]),
            lat_max=float(roi_bounds[3]),
        )
    return ROI_PRESETS[str(getattr(args, "roi_preset", DEFAULT_ROI_PRESET))]


def _apply_region(args) -> List[List[float]]:
    roig = _resolve_roig(args)
    _base.AR_ROIG = roig
    _base.AR_LON_MIN = min(roig[0][0], roig[1][0])
    _base.AR_LON_MAX = max(roig[2][0], roig[3][0])
    _base.AR_LAT_MIN = min(roig[1][1], roig[3][1])
    _base.AR_LAT_MAX = max(roig[0][1], roig[2][1])
    _base.AR_GRID_N = int(getattr(args, "grid_n", 20))
    return roig


def _append_roi_args(cmd: List[str], args) -> None:
    roi_bounds = getattr(args, "roi_bounds", None)
    if roi_bounds:
        cmd += ["--roi-bounds", *(str(float(v)) for v in roi_bounds)]
    else:
        cmd += ["--roi-preset", str(getattr(args, "roi_preset", DEFAULT_ROI_PRESET))]


def cmd_download(args) -> int:
    script = APP_DIR / "data" / "Maryland" / "Download.py"
    cmd = [sys.executable, str(script)]
    _append_roi_args(cmd, args)
    cmd += ["--project", args.project]
    if args.auth:
        cmd += ["--auth"]
    cmd += ["--layout", str(getattr(args, "layout", "grid"))]
    cmd += ["--composite", str(getattr(args, "composite", "none"))]
    if args.previous_month:
        cmd += ["--previous-month"]
    if args.year is not None:
        cmd += ["--year", str(args.year)]
    if args.month is not None:
        cmd += ["--month", str(args.month)]
    if args.whole_year:
        cmd += ["--whole-year"]
    if args.start_day and args.end_day:
        cmd += ["--start-day", args.start_day, "--end-day", args.end_day]
    cmd += ["--data-root", args.data_root]
    statewide_dir = getattr(args, "statewide_dir", None)
    if statewide_dir:
        cmd += ["--statewide-dir", str(statewide_dir)]
    cmd += ["--month-tag-day", str(int(getattr(args, "month_tag_day", 15)))]
    cmd += ["--statewide-split", str(int(getattr(args, "statewide_split", 2)))]
    cmd += ["--workers", str(args.download_workers)]
    cmd += ["--cloud-thresh", str(args.cloud_thresh)]
    cmd += ["--mask-water" if getattr(args, "mask_water", False) else "--no-mask-water"]
    cmd += ["--retries", str(args.download_retries)]
    cmd += ["--retry-sleep-s", str(args.download_retry_sleep_s)]
    cmd += ["--rate-limit-retries", str(args.download_rate_limit_retries)]
    cmd += ["--backoff-base-s", str(args.download_backoff_base_s)]
    cmd += ["--backoff-max-s", str(args.download_backoff_max_s)]
    cmd += ["--retry-jitter-s", str(args.download_retry_jitter_s)]
    cmd += ["--verify" if args.download_verify else "--no-verify"]
    cmd += ["--strict" if args.download_strict else "--no-strict"]
    cmd += ["--fast-skip" if args.download_fast_skip else "--no-fast-skip"]

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_statewide_download(args) -> int:
    dl_args = argparse.Namespace(**vars(args))
    dl_args.layout = "statewide"
    dl_args.composite = "monthly"
    return cmd_download(dl_args)


def cmd_statewide_retile(args) -> int:
    _apply_region(args)
    return _base.cmd_statewide_retile(args)


def cmd_fastdiffsr(args) -> int:
    _apply_region(args)
    return _base.cmd_fastdiffsr(args)


def cmd_fastdiffsr_rerun_incomplete(args) -> int:
    _apply_region(args)
    return _base.cmd_fastdiffsr_rerun_incomplete(args)


def cmd_fastdiffsr_retile(args) -> int:
    _apply_region(args)
    return _base.cmd_fastdiffsr_retile(args)


def cmd_fastdiffsr_mosaic(args) -> int:
    return _base.cmd_fastdiffsr_mosaic(args)


def _add_roi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--roi-preset",
        choices=sorted(ROI_PRESETS.keys()),
        default=DEFAULT_ROI_PRESET,
        help="Maryland ROI preset. Default: md_large.",
    )
    parser.add_argument(
        "--roi-bounds",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=None,
        help="Optional custom bounds in decimal degrees. Overrides --roi-preset.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "Maryland deployment helper (Sentinel-2 download, statewide retile, and FastDiffSR utilities)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download Maryland raw data from Earth Engine")
    _add_roi_args(dl)
    dl.add_argument("--project", default="satelite-430703")
    dl.add_argument("--auth", action="store_true", help="Interactive ee.Authenticate() (not for cron)")
    dl.add_argument("--previous-month", action="store_true")
    dl.add_argument("--year", type=int)
    dl.add_argument("--month", type=int)
    dl.add_argument("--whole-year", action="store_true")
    dl.add_argument("--start-day")
    dl.add_argument("--end-day")
    dl.add_argument("--data-root", default="/mnt/vhvkhoa_ssd/datasets")
    dl.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Parallel download workers. Grid mode uses a multiprocessing pool; statewide mode uses parallel export workers. Lower values may help avoid rate limits on large ROIs.",
    )
    dl.add_argument("--cloud-thresh", type=float, default=20)
    dl.add_argument(
        "--mask-water",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask SCL water pixels (class 6) so sea or water exports as nodata or black in previews.",
    )
    dl.add_argument("--download-retries", type=int, default=3, help="Max attempts for non-rate-limit export failures.")
    dl.add_argument("--download-retry-sleep-s", type=float, default=1.0, help="Base sleep between non-rate-limit retries (seconds).")
    dl.add_argument("--download-rate-limit-retries", type=int, default=8, help="Max attempts for 429/rate-limit export failures.")
    dl.add_argument("--download-backoff-base-s", type=float, default=5.0, help="Initial automatic backoff for 429/rate-limit retries (seconds).")
    dl.add_argument("--download-backoff-max-s", type=float, default=120.0, help="Maximum automatic backoff for 429/rate-limit retries (seconds).")
    dl.add_argument("--download-retry-jitter-s", type=float, default=1.0, help="Random jitter added to automatic 429/rate-limit backoff (seconds).")
    dl.add_argument(
        "--download-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify all expected files exist after download.",
    )
    dl.add_argument(
        "--download-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if verification finds missing files.",
    )
    dl.add_argument(
        "--download-fast-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a previous run marked a grid complete, skip EE calls and only do local verification.",
    )
    dl.set_defaults(func=cmd_download)

    sd = sub.add_parser("statewide_download", help="Download Maryland statewide monthly Sentinel-2 mosaics")
    _add_roi_args(sd)
    sd.add_argument("--project", default="satelite-430703")
    sd.add_argument("--auth", action="store_true", help="Interactive ee.Authenticate() (not for cron)")
    sd.add_argument("--previous-month", action="store_true")
    sd.add_argument("--year", type=int)
    sd.add_argument("--month", type=int)
    sd.add_argument("--whole-year", action="store_true")
    sd.add_argument("--start-day")
    sd.add_argument("--end-day")
    sd.add_argument("--data-root", default="/mnt/vhvkhoa_ssd/datasets")
    sd.add_argument("--statewide-dir", default=None, help="Optional override for MD_<year>_statewide")
    sd.add_argument("--month-tag-day", type=int, default=15, help="Pseudo day recorded for downstream retile (default: 15)")
    sd.add_argument("--statewide-split", type=int, default=2, help="Split statewide export into NxN subregions before merge (default: 2)")
    sd.add_argument(
        "--download-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 8)),
        help="Parallel download workers. Grid mode uses a multiprocessing pool; statewide mode uses parallel export workers. Lower values may help avoid rate limits on large ROIs.",
    )
    sd.add_argument("--cloud-thresh", type=float, default=20)
    sd.add_argument(
        "--mask-water",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask SCL water pixels (class 6) so sea or water exports as nodata or black in previews.",
    )
    sd.add_argument("--download-retries", type=int, default=3, help="Max attempts for non-rate-limit export failures.")
    sd.add_argument("--download-retry-sleep-s", type=float, default=1.0, help="Base sleep between non-rate-limit retries (seconds).")
    sd.add_argument("--download-rate-limit-retries", type=int, default=8, help="Max attempts for 429/rate-limit export failures.")
    sd.add_argument("--download-backoff-base-s", type=float, default=5.0, help="Initial automatic backoff for 429/rate-limit retries (seconds).")
    sd.add_argument("--download-backoff-max-s", type=float, default=120.0, help="Maximum automatic backoff for 429/rate-limit retries (seconds).")
    sd.add_argument("--download-retry-jitter-s", type=float, default=1.0, help="Random jitter added to automatic 429/rate-limit backoff (seconds).")
    sd.add_argument(
        "--download-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify all expected files exist after download.",
    )
    sd.add_argument(
        "--download-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if verification finds missing files.",
    )
    sd.add_argument(
        "--download-fast-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a previous run marked a month complete, skip EE calls and only do local verification.",
    )
    sd.set_defaults(func=cmd_statewide_download)

    fd = sub.add_parser("fastdiffsr", help="Run FastDiffSR super-resolution on selected Maryland meta-patches")
    _add_roi_args(fd)
    fd.add_argument("--year", type=int, required=True)
    fd.add_argument("--month", type=int, required=True, choices=range(1, 13))
    fd.add_argument("--raw-root", required=True, help="Path to MD_<year>_raw")
    fd.add_argument("--sr-root", required=True, help="Output root (writes <MM>mo/<meta_patch>.tif)")
    fd.add_argument("--fastdiffsr-config", default=FASTDIFFSR_DEFAULT_CONFIG)
    fd.add_argument(
        "--fastdiffsr-checkpoint",
        required=True,
        help="Checkpoint dir, *_gen.pth/_opt.pth file, or resume-state prefix path.",
    )
    fd.add_argument("--device", default="0", help="GPU ids, e.g. '0' or '0,1'")
    fd.add_argument("--scale", type=int, default=4, help="SR upscale factor (default: 4)")
    fd.add_argument("--lr-tile-size", type=int, default=64, help="LR tile size for strict tiled inference (default: 64)")
    fd.add_argument("--lr-overlap", type=int, default=16, help="LR overlap between adjacent inference tiles in pixels (default: 16)")
    fd.add_argument("--date-policy", choices=["latest", "earliest", "all", "statewide_anchor"], default="latest")
    fd.add_argument("--cloud-thresh-fastdiffsr", type=float, default=0.10, help="Max cloud fraction by SCL mask (0..1).")
    fd.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for meta-patch inference (default: True).",
    )
    fd.add_argument("--force", action="store_true")
    fd.add_argument("--meta-patches", nargs="*", default=None, help="Optional subset like '0_0 0_1 ...'")
    fd.set_defaults(func=cmd_fastdiffsr)

    fi = sub.add_parser(
        "fastdiffsr_rerun_incomplete",
        help="Scan a month output dir for small or missing FastDiffSR tiles and rerun only those Maryland meta-patches",
    )
    _add_roi_args(fi)
    fi.add_argument("--year", type=int, required=True)
    fi.add_argument("--month", type=int, required=True, choices=range(1, 13))
    fi.add_argument("--raw-root", required=True, help="Path to MD_<year>_raw")
    fi.add_argument("--sr-root", required=True, help="Output root (expects <MM>mo/<meta_patch>.tif)")
    fi.add_argument("--fastdiffsr-config", default=FASTDIFFSR_DEFAULT_CONFIG)
    fi.add_argument(
        "--fastdiffsr-checkpoint",
        required=True,
        help="Checkpoint dir, *_gen.pth/_opt.pth file, or resume-state prefix path.",
    )
    fi.add_argument("--device", default="0", help="GPU ids, e.g. '0' or '0,1'")
    fi.add_argument("--scale", type=int, default=4, help="SR upscale factor (default: 4)")
    fi.add_argument("--lr-tile-size", type=int, default=64, help="LR tile size for strict tiled inference (default: 64)")
    fi.add_argument("--lr-overlap", type=int, default=16, help="LR overlap between adjacent inference tiles in pixels (default: 16)")
    fi.add_argument("--date-policy", choices=["latest", "earliest", "all", "statewide_anchor"], default="latest")
    fi.add_argument("--cloud-thresh-fastdiffsr", type=float, default=0.10, help="Max cloud fraction by SCL mask (0..1).")
    fi.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars while scanning or rerunning tiles (default: True).",
    )
    fi.add_argument(
        "--min-size-mb",
        type=float,
        default=120.0,
        help="Mark tile as incomplete when file size is smaller than this threshold (default: 120MB).",
    )
    fi.add_argument(
        "--include-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include missing output tiles in rerun candidates (default: True).",
    )
    fi.add_argument("--dry-run", action="store_true", help="Only print candidates; do not rerun inference.")
    fi.add_argument("--show-first", type=int, default=20, help="How many candidate IDs to print in dry-run (default: 20).")
    fi.add_argument("--meta-patches", nargs="*", default=None, help="Optional subset like '0_0 0_1 ...'")
    fi.set_defaults(func=cmd_fastdiffsr_rerun_incomplete)

    fr = sub.add_parser(
        "fastdiffsr_retile",
        help="Post-process existing FastDiffSR month tiles: mosaic by georef and retile to the Maryland grid",
    )
    _add_roi_args(fr)
    fr.add_argument("--in-dir", required=True, help="Input month dir containing FastDiffSR tiles like <i>_<j>.tif")
    fr.add_argument("--out-dir", required=True, help="Output month dir for Maryland-grid re-tiled GeoTIFFs")
    fr.add_argument("--grid-n", type=int, default=20, help="Maryland grid size per axis (default: 20)")
    fr.add_argument("--overwrite", action="store_true", help="Overwrite existing output tiles")
    fr.add_argument(
        "--write-empty",
        action="store_true",
        help="Keep empty target tiles (currently skipped if no source overlap or all-zero merge)",
    )
    fr.set_defaults(func=cmd_fastdiffsr_retile)

    fm = sub.add_parser(
        "fastdiffsr_mosaic",
        help="Merge an existing FastDiffSR month directory into one compressed GeoTIFF mosaic",
    )
    fm.add_argument("--in-dir", required=True, help="Input month dir containing FastDiffSR tiles like <i>_<j>.tif")
    fm.add_argument("--out-file", required=True, help="Output mosaic GeoTIFF path")
    fm.add_argument("--nodata", type=int, default=0, help="Nodata value for the merged mosaic (default: 0)")
    fm.add_argument(
        "--build-overviews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build internal overviews for faster windowed reads (default: True)",
    )
    fm.add_argument("--overwrite", action="store_true", help="Overwrite an existing output mosaic")
    fm.set_defaults(func=cmd_fastdiffsr_mosaic)

    sr = sub.add_parser(
        "statewide_retile",
        help="Retile a statewide monthly Maryland mosaic back into the canonical raw grid layout",
    )
    _add_roi_args(sr)
    sr.add_argument("--in-dir", required=True, help="Input statewide month dir containing band files like B2_2025-06.tif")
    sr.add_argument("--out-root", required=True, help="Output MD_<year>_raw root")
    sr.add_argument("--year", type=int, required=True)
    sr.add_argument("--month", type=int, required=True, choices=range(1, 13))
    sr.add_argument("--grid-n", type=int, default=20, help="Maryland grid size per axis (default: 20)")
    sr.add_argument("--month-tag-day", type=int, default=15, help="Pseudo day in output filenames and directories (default: 15)")
    sr.add_argument("--overwrite", action="store_true", help="Overwrite existing output band files")
    sr.set_defaults(func=cmd_statewide_retile)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
