# main.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harvest_estimation.create_doy_prediction_input.config import Config
from harvest_estimation.create_doy_prediction_input.farm_analysis import (
    FarmLevelAnalyzer,
)
from harvest_estimation.create_doy_prediction_input.global_analysis import (
    GlobalAnalyzer,
)
from harvest_estimation.create_doy_prediction_input.log import PipelineLogger
from harvest_estimation.create_doy_prediction_input.summary import (
    summarize_farm_harvest_dates,
)
from harvest_estimation.create_doy_prediction_input.utils.raster_io import (
    cleanup_index_cache,
)

CHECKPOINT_FILE_NAME = ".pipeline_checkpoint.json"
STAGE_NAMES = ("global", "farm", "summary", "cleanup_npy")
STAGE_STATUSES = {"pending", "in_progress", "completed", "failed"}


def load_gt_windows_yaml(path: Path | None) -> dict[str, tuple[str, str]]:
    """
    Load GT windows mapping:

    Format A (recommended):
      Corn:
        - "09-10"
        - "10-05"

    Format B:
      Corn:
        start: "09-10"
        end: "10-05"
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"GT windows YAML not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    gt: dict[str, tuple[str, str]] = {}
    for crop_name, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt[crop_name] = (str(v[0]), str(v[1]))
        elif isinstance(v, dict) and "start" in v and "end" in v:
            gt[crop_name] = (str(v["start"]), str(v["end"]))
        else:
            raise ValueError(f"Bad GT window format for {crop_name}: {v}")
    return gt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Harvest estimation pipeline (refactored, multi-tile)."
    )

    # Multi-tile root
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing tile subdirectories (e.g. 12_0, 12_1, ...).",
    )

    # Config inputs
    p.add_argument(
        "--cdl-yaml",
        type=Path,
        required=True,
        help="CDL YAML mapping (num2class, crop_type).",
    )
    p.add_argument(
        "--gt-windows-yaml",
        type=Path,
        default=None,
        help="Optional YAML mapping crop_name -> [MM-DD, MM-DD] (or {start,end}).",
    )
    p.add_argument(
        "--seeding-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML mapping crop_name -> {window_start, window_end, offset_days}.",
    )
    p.add_argument(
        "--all-crops",
        action="store_true",
        help="Process every crop type defined in the CDL YAML that appears in each tile. GT windows remain optional.",
    )

    # Outputs
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs"),
        help="Output root directory. Each tile gets its own subfolder here.",
    )

    # Pipeline toggles
    p.add_argument(
        "--no-global", action="store_true", help="Skip global analysis stage."
    )
    p.add_argument(
        "--no-farm", action="store_true", help="Skip farm-level analysis stage."
    )
    p.add_argument(
        "--summarize",
        action="store_true",
        help="After farm stage, write farm_harvest_summary.csv per tile output folder.",
    )

    # Optional: run only selected tiles
    p.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional list of tile folder names to process (e.g. --tiles 12_0 12_1).",
    )

    # Logging
    p.add_argument("--log-dir", type=Path, default=Path("./logs"))
    p.add_argument("--log-name", type=str, default="harvest_pipeline")

    # Resume / checkpoint controls
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume by skipping stages already marked completed in per-tile checkpoint files.",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoint files and rerun requested stages.",
    )
    p.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="Delete per-tile checkpoint files before processing tiles.",
    )

    p.add_argument("--cleanup-npy", action="store_true")
    p.add_argument(
        "--no-index-images",
        action="store_true",
        help="Do not save per-date NDVI/NDWI/EVI NPY and PNG files. Useful for model-inference input generation without farm analysis.",
    )
    p.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Overwrite existing crop/farm CSV and plot outputs instead of skipping them.",
    )

    return p.parse_args()


def list_tile_dirs(dataset_root: Path, tiles: list[str] | None) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"--dataset-root is not a directory: {dataset_root}")

    all_tiles = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if tiles:
        tiles_set = set(tiles)
        selected = [d for d in all_tiles if d.name in tiles_set]
        missing = sorted(list(tiles_set - {d.name for d in selected}))
        if missing:
            raise FileNotFoundError(
                f"Requested tiles not found under {dataset_root}: {missing}"
            )
        return selected

    return all_tiles


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def new_tile_checkpoint(tile_name: str) -> dict:
    now = utc_now_iso()
    return {
        "version": 1,
        "tile": tile_name,
        "tile_complete": False,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "stages": {
            stage: {"status": "pending", "updated_at": now, "error": None}
            for stage in STAGE_NAMES
        },
    }


def load_tile_checkpoint(
    checkpoint_path: Path, tile_name: str, logger: PipelineLogger
) -> dict:
    checkpoint = new_tile_checkpoint(tile_name)

    if not checkpoint_path.exists():
        return checkpoint

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(
            f"⚠️ Failed to read checkpoint {checkpoint_path}: {e}. Starting with empty checkpoint."
        )
        return checkpoint

    checkpoint["version"] = raw.get("version", checkpoint["version"])
    checkpoint["tile"] = str(raw.get("tile", tile_name))
    checkpoint["created_at"] = str(raw.get("created_at", checkpoint["created_at"]))
    checkpoint["updated_at"] = str(raw.get("updated_at", checkpoint["updated_at"]))
    checkpoint["completed_at"] = raw.get("completed_at")
    checkpoint["tile_complete"] = bool(raw.get("tile_complete", False))

    raw_stages = (
        raw.get("stages", {}) if isinstance(raw.get("stages", {}), dict) else {}
    )
    for stage in STAGE_NAMES:
        stage_raw = (
            raw_stages.get(stage, {})
            if isinstance(raw_stages.get(stage, {}), dict)
            else {}
        )
        status = str(stage_raw.get("status", "pending"))
        if status not in STAGE_STATUSES:
            status = "pending"
        checkpoint["stages"][stage]["status"] = status
        checkpoint["stages"][stage]["updated_at"] = str(
            stage_raw.get("updated_at", checkpoint["stages"][stage]["updated_at"])
        )
        checkpoint["stages"][stage]["error"] = stage_raw.get("error")

    return checkpoint


def save_tile_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, sort_keys=True)
    tmp_path.replace(checkpoint_path)


def set_stage_status(
    checkpoint: dict, stage: str, status: str, error: str | None = None
) -> None:
    now = utc_now_iso()
    checkpoint["stages"][stage]["status"] = status
    checkpoint["stages"][stage]["updated_at"] = now
    checkpoint["stages"][stage]["error"] = error
    checkpoint["updated_at"] = now


def update_tile_completion(checkpoint: dict, enabled_stages: list[str]) -> None:
    now = utc_now_iso()
    is_complete = all(
        checkpoint["stages"][stage]["status"] == "completed" for stage in enabled_stages
    )
    checkpoint["tile_complete"] = is_complete
    checkpoint["completed_at"] = now if is_complete else None
    checkpoint["updated_at"] = now


def main() -> None:
    args = parse_args()

    logger = PipelineLogger(name=args.log_name, log_dir=str(args.log_dir))

    gt_windows = load_gt_windows_yaml(args.gt_windows_yaml)

    allowed_crops = None if args.all_crops else set(gt_windows.keys())

    cfg = Config(
        args.cdl_yaml,
        allowed_crops=allowed_crops,
        output_root=str(args.output_root),
        gt_windows=gt_windows,
        seeding_config_yaml=args.seeding_config_yaml,
    )
    cfg.overwrite_outputs = bool(args.overwrite_outputs)
    cfg.save_index_images = not bool(args.no_index_images)

    try:
        tile_dirs = list_tile_dirs(args.dataset_root, args.tiles)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("✅ Pipeline config loaded.")
    logger.info(f"dataset_root  = {args.dataset_root}")
    logger.info(
        f"tiles         = {len(tile_dirs)} ({'selected' if args.tiles else 'all'})"
    )
    logger.info(f"output_root   = {args.output_root}")
    logger.info(f"seeding_yaml  = {args.seeding_config_yaml}")
    logger.info(f"all_crops     = {args.all_crops}")
    logger.info(f"run_global    = {not args.no_global}")
    logger.info(f"run_farm      = {not args.no_farm}")
    logger.info(f"summarize     = {args.summarize}")
    logger.info(f"cleanup_npy   = {args.cleanup_npy}")
    logger.info(f"index_images  = {cfg.save_index_images}")
    logger.info(f"overwrite     = {args.overwrite_outputs}")
    logger.info(f"resume        = {args.resume}")
    logger.info(f"checkpoint    = {CHECKPOINT_FILE_NAME}")

    global_analyzer = GlobalAnalyzer(cfg, logger)
    farm_analyzer = FarmLevelAnalyzer(cfg, logger)

    processed = 0
    skipped = 0
    resumed = 0

    enabled_stages: list[str] = []
    if not args.no_global:
        enabled_stages.append("global")
    if not args.no_farm:
        enabled_stages.append("farm")
    if args.summarize:
        enabled_stages.append("summary")
    if args.cleanup_npy:
        enabled_stages.append("cleanup_npy")

    first_tile = tile_dirs[0]
    first_ts = sorted(d.name for d in first_tile.iterdir() if d.is_dir())[0]
    year = int(first_ts.split("-")[0])

    for tile_dir in tile_dirs:
        sample_path = tile_dir
        plot_mask_path = tile_dir / "cdl.tif"
        tile_output_root = args.output_root / tile_dir.name
        tile_output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tile_output_root / CHECKPOINT_FILE_NAME

        logger.info(f"  Tile: {tile_dir.name}")
        logger.info(f"  sample_path   = {sample_path}")
        logger.info(f"  plot_mask_path= {plot_mask_path}")
        logger.info(f"  out_dir       = {tile_output_root}")

        if not plot_mask_path.exists():
            logger.warning(f"⚠️ Missing cdl.tif for tile {tile_dir.name}. Skipping.")
            skipped += 1
            continue

        checkpoint: dict | None = None
        if args.resume:
            if args.reset_checkpoints and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"🧹 Removed checkpoint: {checkpoint_path}")

            checkpoint = load_tile_checkpoint(checkpoint_path, tile_dir.name, logger)
            update_tile_completion(checkpoint, enabled_stages)
            save_tile_checkpoint(checkpoint_path, checkpoint)

            if checkpoint["tile_complete"]:
                logger.info(
                    f"⏭️ Resume skip tile {tile_dir.name} (all requested stages completed)."
                )
                resumed += 1
                continue

        # Global analysis
        if not args.no_global:
            if (
                args.resume
                and checkpoint
                and checkpoint["stages"]["global"]["status"] == "completed"
            ):
                logger.info(
                    f"↪️  Resume skip global:{tile_dir.name} (checkpoint completed)."
                )
            else:
                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "global", "in_progress")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)

                logger.start(f"global:{tile_dir.name}")
                try:
                    global_analyzer.run(
                        sample_path=sample_path,
                        plot_mask_path=plot_mask_path,
                        output_root=tile_output_root,
                        year=year,
                        run_global=True,
                    )
                except Exception as e:
                    logger.fail(f"global:{tile_dir.name}")
                    if args.resume and checkpoint:
                        set_stage_status(checkpoint, "global", "failed", str(e))
                        update_tile_completion(checkpoint, enabled_stages)
                        save_tile_checkpoint(checkpoint_path, checkpoint)
                    raise
                logger.end(f"global:{tile_dir.name}")

                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "global", "completed")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)
        else:
            logger.info("↪️  Skipping global analysis (--no-global).")

        # Farm-level analysis
        if not args.no_farm:
            if (
                args.resume
                and checkpoint
                and checkpoint["stages"]["farm"]["status"] == "completed"
            ):
                logger.info(
                    f"↪️  Resume skip farm:{tile_dir.name} (checkpoint completed)."
                )
            else:
                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "farm", "in_progress")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)

                logger.start(f"farm:{tile_dir.name}")
                try:
                    farm_analyzer.run(
                        sample_path=sample_path,
                        plot_mask_path=plot_mask_path,
                        output_root=tile_output_root,
                        year=year,
                    )
                except Exception as e:
                    logger.fail(f"farm:{tile_dir.name}")
                    if args.resume and checkpoint:
                        set_stage_status(checkpoint, "farm", "failed", str(e))
                        update_tile_completion(checkpoint, enabled_stages)
                        save_tile_checkpoint(checkpoint_path, checkpoint)
                    raise
                logger.end(f"farm:{tile_dir.name}")

                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "farm", "completed")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)
        else:
            logger.info("↪️  Skipping farm-level analysis (--no-farm).")

        # Per-tile summary
        if args.summarize:
            if (
                args.resume
                and checkpoint
                and checkpoint["stages"]["summary"]["status"] == "completed"
            ):
                logger.info(
                    f"↪️  Resume skip summary:{tile_dir.name} (checkpoint completed)."
                )
            else:
                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "summary", "in_progress")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)

                logger.start(f"summary:{tile_dir.name}")
                try:
                    summarize_farm_harvest_dates(str(tile_output_root))
                except Exception as e:
                    logger.fail(f"summary:{tile_dir.name}")
                    if args.resume and checkpoint:
                        set_stage_status(checkpoint, "summary", "failed", str(e))
                        update_tile_completion(checkpoint, enabled_stages)
                        save_tile_checkpoint(checkpoint_path, checkpoint)
                    raise
                logger.end(f"summary:{tile_dir.name}")

                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "summary", "completed")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)

        if args.cleanup_npy:
            if (
                args.resume
                and checkpoint
                and checkpoint["stages"]["cleanup_npy"]["status"] == "completed"
            ):
                logger.info(
                    f"↪️  Resume skip cleanup_npy:{tile_dir.name} (checkpoint completed)."
                )
            else:
                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "cleanup_npy", "in_progress")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)

                logger.info("🧹 Cleaning up index .npy cache files...")
                try:
                    for crop_name in cfg.allowed_crops:
                        crop_dir = tile_output_root / cfg.crop_dir_name(crop_name)
                        if crop_dir.exists():
                            cleanup_index_cache(crop_dir)
                except Exception as e:
                    logger.fail(f"cleanup_npy:{tile_dir.name}")
                    if args.resume and checkpoint:
                        set_stage_status(checkpoint, "cleanup_npy", "failed", str(e))
                        update_tile_completion(checkpoint, enabled_stages)
                        save_tile_checkpoint(checkpoint_path, checkpoint)
                    raise

                if args.resume and checkpoint:
                    set_stage_status(checkpoint, "cleanup_npy", "completed")
                    update_tile_completion(checkpoint, enabled_stages)
                    save_tile_checkpoint(checkpoint_path, checkpoint)

        if args.resume and checkpoint:
            update_tile_completion(checkpoint, enabled_stages)
            save_tile_checkpoint(checkpoint_path, checkpoint)

        processed += 1

    logger.info(f"🏁 Done. processed={processed}, skipped={skipped}, resumed={resumed}")


if __name__ == "__main__":
    main()

# python -m harvest_estimation.create_doy_prediction_input.main --dataset-root ../AR_sentinel2/2019_AR --cdl-yaml configs/Arkansas/cdl.yaml --gt-windows-yaml configs/Arkansas/gt_windows.yaml --output-root outputs/2019_AR --summarize --cleanup-npy
