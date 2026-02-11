import argparse
import calendar
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

# Make relative paths work whether launched from repo root or DeepSatModels/
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint


AR_ROIG = [
    [-94.7610, 36.6652],
    [-94.7610, 32.8376],
    [-89.5522, 36.6652],
    [-89.5522, 32.8376],
]

SAMPLE_REQUIREMENTS: dict[int, int] = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 1,
    11: 1,
    12: 1,
}

SELECTED_BANDS: dict[str, list[str]] = {
    "10m": ["B2", "B3", "B4", "B8"],
    "20m": ["B5", "B6", "B7", "B8A", "B11", "B12"],
    "SCL": ["SCL"],
}


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _month_range(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _previous_month_range(today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    first_this = date(today.year, today.month, 1)
    last_prev = first_this - timedelta(days=1)
    start_prev = date(last_prev.year, last_prev.month, 1)
    return start_prev.strftime("%Y-%m-%d"), last_prev.strftime("%Y-%m-%d")


def _max_doy_for_month(year: int, month: int) -> int:
    _, end = _month_range(year, month)
    return _parse_ymd(end).timetuple().tm_yday


def _clamp_model_month(month: int) -> int:
    return max(6, min(11, int(month)))


def _default_model_paths(model_month: int) -> tuple[str, str]:
    mm = int(model_month)
    saved_cfg = f"models/saved_models/AR23_focal_{mm:02d}mo/config_file.yaml"
    if Path(saved_cfg).exists():
        cfg = saved_cfg
    else:
        cfg = f"configs/Arkansas/TSViT_AR23_{mm:02d}mo_focal.yaml"

    ckpt = f"models/saved_models/AR23_focal_{mm:02d}mo/best.pth"
    if not Path(ckpt).exists() and mm == 11 and Path("models/saved_models/AR23_focal_11mo/best.pth").exists():
        ckpt = "models/saved_models/AR23_focal_11mo/best.pth"
    return cfg, ckpt


def _list_meta_patches(raw_root: Path) -> list[str]:
    out = []
    for d in raw_root.iterdir():
        if d.is_dir() and re.fullmatch(r"\d+_\d+", d.name):
            out.append(d.name)
    return sorted(out, key=lambda s: (int(s.split("_")[0]), int(s.split("_")[1])))


def _list_dates(meta_dir: Path, *, start_day: str, end_day: str) -> list[date]:
    start = _parse_ymd(start_day)
    end = _parse_ymd(end_day)
    out: list[date] = []
    for d in meta_dir.iterdir():
        if not d.is_dir():
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name):
            continue
        try:
            dt = _parse_ymd(d.name)
        except Exception:
            continue
        if start <= dt <= end:
            out.append(dt)
    return sorted(out)


def _resample_dates(dates: list[date]) -> list[date]:
    by_month: dict[int, list[date]] = {}
    for d in dates:
        by_month.setdefault(d.month, []).append(d)

    keep: list[date] = []
    for m in sorted(by_month.keys()):
        k = int(SAMPLE_REQUIREMENTS.get(m, 0))
        if k <= 0:
            continue
        ds = sorted(by_month[m])
        if not ds:
            continue
        if k == 1:
            keep.append(min(ds, key=lambda x: abs(x.day - 15)))
        else:
            keep.append(ds[0])
            if ds[-1] != ds[0]:
                keep.append(ds[-1])
    return sorted(set(keep))


def _read_satellite_image(meta_dir: Path, day: date) -> dict[str, dict[str, np.ndarray]] | None:
    day_s = day.strftime("%Y-%m-%d")
    sat: dict[str, dict[str, np.ndarray]] = {k: {} for k in SELECTED_BANDS}

    for res, bands in SELECTED_BANDS.items():
        for b in bands:
            fp = meta_dir / day_s / f"{b}_{day_s}.tif"
            if not fp.exists():
                return None
            with rasterio.open(fp) as src:
                data = src.read(1)
            if res == "SCL" and (data > 7).sum() / data.size > 0.10:
                return None
            sat[res][b] = data

    # Upsample 20m + SCL to 10m grid
    h, w = sat["10m"]["B2"].shape
    for b in SELECTED_BANDS["20m"]:
        sat["20m"][b] = cv2.resize(sat["20m"][b], (w, h), interpolation=cv2.INTER_NEAREST)
    scl_raw = sat["SCL"]["SCL"]
    if scl_raw.shape != (h, w):
        sat["SCL"]["SCL"] = cv2.resize(scl_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    return sat


def _stack_bands(sat: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    layers = [sat[res][b] for res, bs in SELECTED_BANDS.items() for b in bs]
    return np.dstack(layers).astype(np.uint16)


def _pad_tile_reflect_or_edge(tile: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    h, w = tile.shape[1], tile.shape[2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile

    mode = "reflect" if (h > 1 and w > 1) else "edge"
    return np.pad(tile, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], mode=mode)


@dataclass(frozen=True)
class RunSpec:
    year: int
    available_month: int
    model_month: int
    start_day: str
    end_day: str
    max_doy: int


def _infer_meta_patch(
    *,
    meta_patch: str,
    raw_root: Path,
    out_dir: Path,
    config_path: str,
    checkpoint_path: str,
    device_ids: list[int],
    batch_size: int,
    tile_size: int,
    start_day: str,
    end_day: str,
    max_doy: int,
    force: bool,
):
    out_fp = out_dir / f"{meta_patch}.tif"
    if out_fp.exists() and not force:
        return

    meta_dir = raw_root / meta_patch
    if not meta_dir.exists():
        return

    all_dates = _list_dates(meta_dir, start_day=start_day, end_day=end_day)
    if not all_dates:
        return

    selected_dates = _resample_dates(all_dates)
    if not selected_dates:
        return

    stacks: list[np.ndarray] = []
    doys: list[int] = []
    used_dates: list[str] = []
    for d in selected_dates:
        sat = _read_satellite_image(meta_dir, d)
        if sat is None:
            continue
        stacks.append(_stack_bands(sat))
        doys.append(d.timetuple().tm_yday)
        used_dates.append(d.strftime("%Y-%m-%d"))

    if not stacks:
        return

    # Enforce max_doy (month model) at inference time.
    keep_idx = [i for i, doy in enumerate(doys) if doy <= max_doy]
    stacks = [stacks[i] for i in keep_idx]
    doys = [doys[i] for i in keep_idx]
    used_dates = [used_dates[i] for i in keep_idx]
    if not stacks:
        return

    h, w, _ = stacks[0].shape

    # Use first used date for georeferencing template
    template_fp = meta_dir / used_dates[0] / f"B2_{used_dates[0]}.tif"
    if not template_fp.exists():
        return

    cfg = read_yaml(config_path)
    device = get_device(device_ids, allow_cpu=False)
    cfg["local_device_ids"] = device_ids
    net = get_model(cfg, device)
    if checkpoint_path:
        load_from_checkpoint(net, checkpoint_path, partial_restore=False, device=device)
    net.to(device)
    net.eval()

    transform = PASTIS_segmentation_transform(cfg["MODEL"], is_training=False)
    doy_arr = np.asarray(doys, dtype=np.int16)
    dummy_labels = np.zeros((1, tile_size, tile_size), dtype=np.uint8)

    pred_mosaic = np.zeros((h, w), dtype=np.uint8)

    batch_inputs: list[torch.Tensor] = []
    batch_pos: list[tuple[int, int]] = []

    def flush():
        if not batch_inputs:
            return
        inputs = torch.stack(batch_inputs, 0).to(device)
        with torch.no_grad():
            logits = net(inputs)
            preds = logits.argmax(1).cpu().numpy().astype(np.uint8)  # (B,H,W)
        for i, (iy, ix) in enumerate(batch_pos):
            vh = min(tile_size, h - iy)
            vw = min(tile_size, w - ix)
            pred_mosaic[iy : iy + vh, ix : ix + vw] = preds[i, :vh, :vw]
        batch_inputs.clear()
        batch_pos.clear()

    y_positions = list(range(0, h, tile_size))
    x_positions = list(range(0, w, tile_size))

    for iy in y_positions:
        for ix in x_positions:
            # Build (T, Ht, Wt, 11) then pad to (T,24,24,11)
            tiles_t = [img[iy : iy + tile_size, ix : ix + tile_size, :] for img in stacks]
            tile = np.stack(tiles_t, 0)
            tile = _pad_tile_reflect_or_edge(tile, target_h=tile_size, target_w=tile_size)

            # Drop SCL channel (last) -> 10 channels, match training normalization.
            if tile.shape[-1] == 11:
                tile = tile[..., :-1]

            # (T, H, W, C) -> (T, C, H, W)
            img_np = np.transpose(tile.astype(np.float32), (0, 3, 1, 2))
            sample = {"img": img_np, "labels": dummy_labels, "doy": doy_arr}
            sample_t = transform(sample)

            batch_inputs.append(sample_t["inputs"])
            batch_pos.append((iy, ix))
            if len(batch_inputs) >= batch_size:
                flush()

    flush()

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_fp) as src:
        profile = src.profile
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        nodata=0,
        compress="DEFLATE",
    )
    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(pred_mosaic, 1)


def cmd_download(args) -> int:
    script = APP_DIR / "data" / "Arkansas" / "Download.py"
    cmd = [sys.executable, str(script)]
    cmd += ["--project", args.project]
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
    cmd += ["--workers", str(args.download_workers)]
    cmd += ["--cloud-thresh", str(args.cloud_thresh)]
    cmd += ["--retries", str(args.download_retries)]
    cmd += ["--retry-sleep-s", str(args.download_retry_sleep_s)]
    cmd += ["--verify" if args.download_verify else "--no-verify"]
    cmd += ["--strict" if args.download_strict else "--no-strict"]
    cmd += ["--fast-skip" if args.download_fast_skip else "--no-fast-skip"]
    if args.auth:
        cmd += ["--auth"]

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_predict(args) -> int:
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"raw_root not found: {raw_root}")

    available_month = int(args.available_month)
    model_month = _clamp_model_month(available_month)
    year = int(args.year)

    start_day = f"{year}-01-01"
    end_day = _month_range(year, available_month)[1]
    max_doy = _max_doy_for_month(year, model_month)

    if available_month < 6:
        print(f"available_month={available_month} < 6: skipping predictions (no supported model yet).")
        return 0

    config_path = args.config or _default_model_paths(model_month)[0]
    checkpoint_path = args.checkpoint or _default_model_paths(model_month)[1]
    if not Path(config_path).exists():
        raise SystemExit(f"config not found: {config_path}")
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device_ids = [int(d) for d in args.device.split(",")] if args.device else [0]
    out_dir = Path(args.pred_root) / f"{model_month:02d}mo"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_patches = args.meta_patches or _list_meta_patches(raw_root)
    print(
        f"Predicting year={year} available_month={available_month} model_month={model_month} "
        f"date_range={start_day}..{end_day} max_doy={max_doy} patches={len(meta_patches)}"
    )

    for mp in meta_patches:
        _infer_meta_patch(
            meta_patch=mp,
            raw_root=raw_root,
            out_dir=out_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device_ids=device_ids,
            batch_size=int(args.batch_size),
            tile_size=int(args.tile_size),
            start_day=start_day,
            end_day=end_day,
            max_doy=max_doy,
            force=bool(args.force),
        )

    # Write a small run metadata file for traceability.
    meta_fp = Path(args.pred_root) / f"{model_month:02d}mo" / "_run.txt"
    meta_fp.write_text(
        f"year={year}\n"
        f"available_month={available_month}\n"
        f"model_month={model_month}\n"
        f"start_day={start_day}\n"
        f"end_day={end_day}\n"
        f"max_doy={max_doy}\n"
        f"config={config_path}\n"
        f"checkpoint={checkpoint_path}\n"
    )
    return 0


def cmd_monthly(args) -> int:
    start_day, end_day = _previous_month_range()
    year = _parse_ymd(start_day).year
    month = _parse_ymd(end_day).month

    # 1) Download previous month
    dl_args = argparse.Namespace(
        project=args.project,
        auth=args.auth,
        previous_month=True,
        year=None,
        month=None,
        whole_year=False,
        start_day=None,
        end_day=None,
        data_root=args.data_root,
        download_workers=args.download_workers,
        cloud_thresh=args.cloud_thresh,
        download_retries=args.download_retries,
        download_retry_sleep_s=args.download_retry_sleep_s,
        download_verify=args.download_verify,
        download_strict=args.download_strict,
        download_fast_skip=args.download_fast_skip,
    )
    rc = cmd_download(dl_args)
    if rc != 0:
        return rc

    # 2) Predict using the model matching available months
    raw_root = Path(args.data_root) / f"AR_{year}_raw"
    pred_root = Path(args.data_root) / f"AR_{year}_preds"

    pred_args = argparse.Namespace(
        year=year,
        available_month=month,
        raw_root=str(raw_root),
        pred_root=str(pred_root),
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        force=args.force,
        meta_patches=args.meta_patches,
    )
    return cmd_predict(pred_args)


def cmd_run(args) -> int:
    """
    One-shot run for a specific (year, month):
      1) download that calendar month
      2) if month >= 6, run the corresponding 6..11mo model and save preds
    """
    year = int(args.year)
    month = int(args.month)

    dl_args = argparse.Namespace(
        project=args.project,
        auth=args.auth,
        previous_month=False,
        year=year,
        month=month,
        whole_year=False,
        start_day=None,
        end_day=None,
        data_root=args.data_root,
        download_workers=args.download_workers,
        cloud_thresh=args.cloud_thresh,
        download_retries=args.download_retries,
        download_retry_sleep_s=args.download_retry_sleep_s,
        download_verify=args.download_verify,
        download_strict=args.download_strict,
        download_fast_skip=args.download_fast_skip,
    )
    rc = cmd_download(dl_args)
    if rc != 0:
        return rc

    raw_root = Path(args.data_root) / f"AR_{year}_raw"
    pred_root = Path(args.data_root) / f"AR_{year}_preds"

    pred_args = argparse.Namespace(
        year=year,
        available_month=month,
        raw_root=str(raw_root),
        pred_root=str(pred_root),
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        force=args.force,
        meta_patches=args.meta_patches,
    )
    return cmd_predict(pred_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Arkansas deployment helper (download → monthly model → save preds)")
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download raw data from Earth Engine (delegates to data/Arkansas/Download.py)")
    dl.add_argument("--project", default="satelite-430703")
    dl.add_argument("--auth", action="store_true", help="Interactive ee.Authenticate() (not for cron)")
    dl.add_argument("--previous-month", action="store_true")
    dl.add_argument("--year", type=int)
    dl.add_argument("--month", type=int)
    dl.add_argument("--whole-year", action="store_true")
    dl.add_argument("--start-day")
    dl.add_argument("--end-day")
    dl.add_argument("--data-root", default="/mnt/vhvkhoa_ssd/datasets")
    dl.add_argument("--download-workers", type=int, default=4)
    dl.add_argument("--cloud-thresh", type=float, default=20)
    dl.add_argument("--download-retries", type=int, default=3, help="Retries per missing band download")
    dl.add_argument("--download-retry-sleep-s", type=float, default=1.0, help="Sleep between retries (seconds)")
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

    pr = sub.add_parser("predict", help="Run batch inference and save GeoTIFF preds for one year/month")
    pr.add_argument("--year", type=int, required=True)
    pr.add_argument(
        "--available-month",
        type=int,
        required=True,
        help="How many months of data are available (1-12). Selects model=min(max(m,6),11).",
    )
    pr.add_argument("--raw-root", required=True, help="Path to AR_<year>_raw")
    pr.add_argument("--pred-root", required=True, help="Output dir (will write <MM>mo/<meta_patch>.tif)")
    pr.add_argument("--config", default=None)
    pr.add_argument("--checkpoint", default=None)
    pr.add_argument("--device", default="0", help="GPU ids, e.g. '0' or '0,1'")
    pr.add_argument("--batch-size", type=int, default=64)
    pr.add_argument("--tile-size", type=int, default=24)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--meta-patches", nargs="*", default=None, help="Optional subset like '0_0 0_1 ...'")
    pr.set_defaults(func=cmd_predict)

    mo = sub.add_parser("monthly", help="Run previous-month download + predictions (safe for cron)")
    mo.add_argument("--project", default="satelite-430703")
    mo.add_argument("--auth", action="store_true", help="Interactive ee.Authenticate() (not for cron)")
    mo.add_argument("--data-root", default="/mnt/vhvkhoa_ssd/datasets")
    mo.add_argument("--download-workers", type=int, default=4)
    mo.add_argument("--cloud-thresh", type=float, default=20)
    mo.add_argument("--download-retries", type=int, default=3)
    mo.add_argument("--download-retry-sleep-s", type=float, default=1.0)
    mo.add_argument("--download-verify", action=argparse.BooleanOptionalAction, default=True)
    mo.add_argument("--download-strict", action=argparse.BooleanOptionalAction, default=True)
    mo.add_argument("--download-fast-skip", action=argparse.BooleanOptionalAction, default=True)
    mo.add_argument("--config", default=None)
    mo.add_argument("--checkpoint", default=None)
    mo.add_argument("--device", default="0")
    mo.add_argument("--batch-size", type=int, default=64)
    mo.add_argument("--tile-size", type=int, default=24)
    mo.add_argument("--force", action="store_true")
    mo.add_argument("--meta-patches", nargs="*", default=None)
    mo.set_defaults(func=cmd_monthly)

    ru = sub.add_parser("run", help="Download a specific (year, month) and (if month>=6) run predictions")
    ru.add_argument("--project", default="satelite-430703")
    ru.add_argument("--auth", action="store_true", help="Interactive ee.Authenticate() (not for cron)")
    ru.add_argument("--year", type=int, required=True)
    ru.add_argument("--month", type=int, required=True, choices=range(1, 13))
    ru.add_argument("--data-root", default="/mnt/vhvkhoa_ssd/datasets")
    ru.add_argument("--download-workers", type=int, default=4)
    ru.add_argument("--cloud-thresh", type=float, default=20)
    ru.add_argument("--download-retries", type=int, default=3)
    ru.add_argument("--download-retry-sleep-s", type=float, default=1.0)
    ru.add_argument("--download-verify", action=argparse.BooleanOptionalAction, default=True)
    ru.add_argument("--download-strict", action=argparse.BooleanOptionalAction, default=True)
    ru.add_argument("--download-fast-skip", action=argparse.BooleanOptionalAction, default=True)
    ru.add_argument("--config", default=None)
    ru.add_argument("--checkpoint", default=None)
    ru.add_argument("--device", default="0")
    ru.add_argument("--batch-size", type=int, default=64)
    ru.add_argument("--tile-size", type=int, default=24)
    ru.add_argument("--force", action="store_true")
    ru.add_argument("--meta-patches", nargs="*", default=None)
    ru.set_defaults(func=cmd_run)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
