from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import transform_bounds

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CODE_ROOT = APP_DIR / "harvest_estimation"
DEFAULT_MODEL_ROOT = APP_DIR / "harvest_estimation" / "models"
DEFAULT_PREDICTIONS_ROOT = APP_DIR / "runtime_data" / "harvest_predictions"
CSV_FIELDS = (
    "year",
    "tile",
    "crop",
    "pred_start_doy",
    "pred_end_doy",
    "num_observations",
    "model_test_mae_days",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute tile-level Arkansas harvest predictions with the deployed direct Hybrid models."
    )
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Exact directory containing per-crop checkpoint directories. Overrides --model-root.",
    )
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT
    )
    parser.add_argument("--model-window", default="1year")
    parser.add_argument(
        "--feature-set", choices=("ndvi_only", "all_indices"), default="all_indices"
    )
    parser.add_argument("--min-points", type=int, default=2)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device")
    device_group.add_argument(
        "--devices",
        nargs="+",
        help="CUDA devices used in parallel, for example: --devices cuda:0 cuda:1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_model_modules(code_root: Path):
    code_root_str = str(code_root.resolve())
    if code_root_str not in sys.path:
        sys.path.insert(0, code_root_str)
    try:
        from doy_prediction.tile_cnn_data import build_tile_crop_records
        from doy_prediction.tile_cnn_model import normalized_to_doy
        from doy_prediction.tile_hybrid_model import TileCNNRNNHybridRegressor
    except Exception as exc:
        raise RuntimeError(
            f"Could not import harvest model code from {code_root}: {exc}"
        ) from exc
    return build_tile_crop_records, normalized_to_doy, TileCNNRNNHybridRegressor


def _load_model(checkpoint_path: Path, device: torch.device, model_class):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "cnn_rnn_hybrid":
        raise RuntimeError(
            f"Unexpected model type in {checkpoint_path}: {checkpoint.get('model_name')}"
        )
    model = model_class(
        in_channels=int(checkpoint["in_channels"]),
        **dict(checkpoint.get("model_config") or {}),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _metrics(checkpoint_path: Path) -> dict:
    path = checkpoint_path.with_name("metrics.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _test_mae(metrics: dict) -> float | None:
    value = (metrics.get("test_metrics") or {}).get("mae_mean")
    return round(float(value), 1) if value is not None else None


def _tile_raster(tile_dir: Path) -> Path | None:
    cdl_path = tile_dir / "cdl.tif"
    if cdl_path.is_file():
        return cdl_path
    return next(iter(sorted(tile_dir.glob("*/B4_*.tif"))), None)


def _tile_bounds_for_year(dataset_root: Path, year: int) -> dict[str, list[float]]:
    year_dir = dataset_root.resolve() / f"{int(year)}_AR"
    if not year_dir.is_dir():
        raise FileNotFoundError(f"Harvest source dataset not found: {year_dir}")

    bounds_by_tile: dict[str, list[float]] = {}
    for tile_dir in sorted(
        (path for path in year_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        raster_path = _tile_raster(tile_dir)
        if raster_path is None:
            continue
        with rasterio.open(raster_path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        bounds_by_tile[tile_dir.name] = [float(value) for value in bounds]
    if not bounds_by_tile:
        raise RuntimeError(f"No source GeoTIFF tile bounds found under {year_dir}")
    return bounds_by_tile


def _supported_region(tile_bounds: dict[str, list[float]]) -> dict[str, float]:
    bounds = list(tile_bounds.values())
    return {
        "lon_min": min(value[0] for value in bounds),
        "lat_min": min(value[1] for value in bounds),
        "lon_max": max(value[2] for value in bounds),
        "lat_max": max(value[3] for value in bounds),
    }


def _artifact_paths(
    *, predictions_root: Path, year: int, model_window: str, feature_set: str
) -> tuple[Path, Path]:
    artifact_dir = (
        predictions_root / f"output_{int(year)}" / model_window / feature_set
    )
    return artifact_dir / "predictions.csv", artifact_dir / "metadata.json"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _serializable_record(record) -> dict[str, object]:
    return {
        "tile": str(record.tile),
        "crop": str(record.crop),
        "x": np.asarray(record.x),
        "num_observations": int(record.num_observations),
    }


def _assign_crop_groups(
    records_by_crop: dict[str, list[dict[str, object]]], devices: list[str]
) -> list[tuple[str, dict[str, list[dict[str, object]]]]]:
    assignments: list[dict[str, list[dict[str, object]]]] = [{} for _ in devices]
    loads = [0 for _ in devices]
    for crop, records in sorted(
        records_by_crop.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        index = min(range(len(devices)), key=lambda value: (loads[value], value))
        assignments[index][crop] = records
        loads[index] += len(records)
    return [
        (device, assignment)
        for device, assignment in zip(devices, assignments)
        if assignment
    ]


def _predict_crop_groups(
    *,
    code_root: str,
    model_dir: str,
    year: int,
    device_name: str,
    records_by_crop: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    _, normalized_to_doy, model_class = _load_model_modules(Path(code_root))
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_rows: list[dict[str, object]] = []
    trained_years: set[int] = set()
    evaluated_years: set[int] = set()
    crops_with_predictions: list[str] = []
    for crop in sorted(records_by_crop):
        checkpoint_path = Path(model_dir) / crop.replace("/", "_") / "best_model.pt"
        if not checkpoint_path.is_file():
            print(
                f"[{device_name}] [skip] Missing checkpoint for crop={crop}: {checkpoint_path}",
                flush=True,
            )
            continue

        crop_records = records_by_crop[crop]
        model = _load_model(checkpoint_path, device, model_class)
        batch = torch.from_numpy(
            np.stack([np.asarray(record["x"]) for record in crop_records])
        ).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            pred_doy = normalized_to_doy(model(batch))

        metrics = _metrics(checkpoint_path)
        trained_years.update(int(value) for value in metrics.get("train_years", []))
        evaluated_years.update(int(value) for value in metrics.get("test_years", []))
        model_test_mae_days = _test_mae(metrics)
        for record, prediction in zip(crop_records, pred_doy):
            output_rows.append(
                {
                    "year": int(year),
                    "tile": str(record["tile"]),
                    "crop": crop,
                    "pred_start_doy": int(prediction[0]),
                    "pred_end_doy": int(prediction[1]),
                    "num_observations": int(record["num_observations"]),
                    "model_test_mae_days": model_test_mae_days
                    if model_test_mae_days is not None
                    else "",
                }
            )
        crops_with_predictions.append(crop)
        print(
            f"[{device_name}] Predicted crop={crop} tiles={len(crop_records)}",
            flush=True,
        )

    return {
        "rows": output_rows,
        "trained_years": sorted(trained_years),
        "evaluated_years": sorted(evaluated_years),
        "crops": crops_with_predictions,
    }


def _run_predictions(
    *,
    args: argparse.Namespace,
    year: int,
    model_dir: Path,
    records_by_crop: dict[str, list[dict[str, object]]],
    devices: list[str],
) -> list[dict[str, object]]:
    assignments = _assign_crop_groups(records_by_crop, devices)
    common = {
        "code_root": str(args.code_root.resolve()),
        "model_dir": str(model_dir),
        "year": int(year),
    }
    if len(assignments) == 1:
        device_name, crop_records = assignments[0]
        return [
            _predict_crop_groups(
                **common,
                device_name=device_name,
                records_by_crop=crop_records,
            )
        ]

    results: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(assignments), mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                _predict_crop_groups,
                **common,
                device_name=device_name,
                records_by_crop=crop_records,
            )
            for device_name, crop_records in assignments
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _precompute_year(
    args: argparse.Namespace, year: int, build_tile_crop_records, devices: list[str]
) -> None:
    year_dir = args.outputs_root.resolve() / f"{int(year)}_AR"
    model_dir = (
        args.model_dir.resolve()
        if args.model_dir is not None
        else args.model_root.resolve() / args.model_window / args.feature_set
    )
    prediction_path, metadata_path = _artifact_paths(
        predictions_root=args.predictions_root.resolve(),
        year=year,
        model_window=args.model_window,
        feature_set=args.feature_set,
    )
    existing = [path for path in (prediction_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Prediction artifact already exists; pass --overwrite to replace it: {existing[0]}"
        )
    if not year_dir.is_dir():
        raise FileNotFoundError(f"Harvest input directory not found: {year_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Hybrid model directory not found: {model_dir}")

    print(f"Building records for year={year} from {year_dir}", flush=True)
    records = build_tile_crop_records(
        year_dir,
        feature_set=args.feature_set,
        min_points=int(args.min_points),
        require_labels=False,
    )
    records_by_crop: dict[str, list] = defaultdict(list)
    for record in records:
        records_by_crop[record.crop].append(_serializable_record(record))

    tile_bounds = _tile_bounds_for_year(args.dataset_root, year)
    record_tiles = {str(record.tile) for record in records}
    missing_bound_tiles = sorted(record_tiles - set(tile_bounds))
    if missing_bound_tiles:
        raise RuntimeError(
            "Source GeoTIFF bounds are missing for input tiles: "
            + ", ".join(missing_bound_tiles)
        )

    output_rows: list[dict[str, object]] = []
    trained_years: set[int] = set()
    evaluated_years: set[int] = set()
    crops_with_predictions: list[str] = []
    for result in _run_predictions(
        args=args,
        year=year,
        model_dir=model_dir,
        records_by_crop=records_by_crop,
        devices=devices,
    ):
        output_rows.extend(result["rows"])
        trained_years.update(int(value) for value in result["trained_years"])
        evaluated_years.update(int(value) for value in result["evaluated_years"])
        crops_with_predictions.extend(str(value) for value in result["crops"])

    if not output_rows:
        raise RuntimeError(f"No harvest predictions were generated for year={year}")
    output_rows.sort(key=lambda row: (str(row["crop"]), str(row["tile"])))
    crops_with_predictions.sort()
    trained_years_text = ", ".join(str(value) for value in sorted(trained_years))
    evaluated_years_text = ", ".join(str(value) for value in sorted(evaluated_years))
    training_note = (
        f"The selected model was trained on {trained_years_text} "
        f"and evaluated on {evaluated_years_text}."
    )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year": int(year),
        "model": "hybrid",
        "model_name": "cnn_rnn_hybrid",
        "model_window": args.model_window,
        "feature_set": args.feature_set,
        "source_outputs_root": str(args.outputs_root.resolve()),
        "source_dataset_root": str(args.dataset_root.resolve()),
        "model_root": str(model_dir),
        "devices": devices,
        "trained_years": sorted(trained_years),
        "evaluated_years": sorted(evaluated_years),
        "training_note": training_note,
        "input_record_count": len(records),
        "prediction_row_count": len(output_rows),
        "tile_count": len({str(row["tile"]) for row in output_rows}),
        "crop_count": len(crops_with_predictions),
        "crops": crops_with_predictions,
        "supported_region": _supported_region(tile_bounds),
        "tile_bounds_wgs84": tile_bounds,
    }
    _write_csv(prediction_path, output_rows)
    _write_json(metadata_path, metadata)
    print(f"Wrote {len(output_rows)} predictions to {prediction_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.min_points < 1:
        raise ValueError("--min-points must be at least 1")
    devices = list(dict.fromkeys(args.devices or [args.device or "cuda"]))
    for device_name in devices:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but CUDA is unavailable: {device_name}"
            )
        if (
            device.type == "cuda"
            and device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"CUDA device index is unavailable: {device_name}; "
                f"visible device count={torch.cuda.device_count()}"
            )

    modules = _load_model_modules(args.code_root)
    build_tile_crop_records = modules[0]
    for year in args.years:
        _precompute_year(args, int(year), build_tile_crop_records, devices)


if __name__ == "__main__":
    main()
