import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_BASE_FP = REPO_ROOT / "data" / "Arkansas" / "Download.py"
_BASE_SPEC = importlib.util.spec_from_file_location("arkansas_download_base", str(_BASE_FP))
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError("Unable to load base Arkansas downloader module.")
_base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _base
_BASE_SPEC.loader.exec_module(_base)


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
LEGACY_DATA_DIR = "/home/khoavo/Desktop/workplace/satelite/raw_maryland/2023_all/"


def _compute_data_dir(*, data_dir: Optional[str], data_root: Optional[str], start_day: str) -> str:
    if data_dir:
        return data_dir

    start_year = _base._parse_ymd(start_day).year
    if data_root:
        return os.path.join(data_root, f"MD_{start_year}_raw")

    return LEGACY_DATA_DIR


def _compute_statewide_dir(*, statewide_dir: Optional[str], data_root: Optional[str], start_day: str) -> str:
    if statewide_dir:
        return statewide_dir

    start_year = _base._parse_ymd(start_day).year
    if data_root:
        return os.path.join(data_root, f"MD_{start_year}_statewide")

    return os.path.join(LEGACY_DATA_DIR, "..", f"MD_{start_year}_statewide")


def _configure_base(*, roi_preset: str, roi_bounds: Optional[List[float]]) -> None:
    if roi_bounds:
        roig = _bounds_to_roig(
            lon_min=float(roi_bounds[0]),
            lat_min=float(roi_bounds[1]),
            lon_max=float(roi_bounds[2]),
            lat_max=float(roi_bounds[3]),
        )
    else:
        roig = ROI_PRESETS[str(roi_preset)]

    _base.DEFAULT_ROIG = roig
    _base.LEGACY_DATA_DIR = LEGACY_DATA_DIR
    _base._compute_data_dir = _compute_data_dir
    _base._compute_statewide_dir = _compute_statewide_dir


def _parse_wrapper_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--roi-preset", choices=sorted(ROI_PRESETS.keys()), default=DEFAULT_ROI_PRESET)
    ap.add_argument(
        "--roi-bounds",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=None,
    )
    return ap.parse_known_args(argv)


def main() -> None:
    wrapper_args, remaining = _parse_wrapper_args(sys.argv[1:])
    _configure_base(roi_preset=str(wrapper_args.roi_preset), roi_bounds=wrapper_args.roi_bounds)

    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        print(
            "Maryland wrapper options: "
            "--roi-preset {md_focus,md_large} or "
            "--roi-bounds LON_MIN LAT_MIN LON_MAX LAT_MAX"
        )

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        _base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
