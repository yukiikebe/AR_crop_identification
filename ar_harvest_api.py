from __future__ import annotations

import calendar
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rasterio.warp import transform_bounds


# DeepSatModels_updated/right_bottom_ar.json and the source GeoTIFF footprint.
HARVEST_LON_MIN = -92.28006055531158
HARVEST_LON_MAX = -89.88049449231089
HARVEST_LAT_MIN = 32.90381728267215
HARVEST_LAT_MAX = 34.75315095485688

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CODE_ROOT = APP_DIR / "harvest_estimation"
DEFAULT_MODEL_ROOT = APP_DIR / "harvest_estimation" / "models"
# HPC deployment data shared from the project owner's storage. Environment
# variables still take precedence, so non-HPC deployments can override these.
DEFAULT_OUTPUTS_ROOT = Path("/scrfs/storage/yikebe/home/DeepSatModels_updated/outputs")
DEFAULT_DATASET_ROOT = Path("/scrfs/storage/yikebe/home/AR_sentinel2")


class BBox(BaseModel):
    lon_min: float = Field(..., ge=-180.0, le=180.0)
    lat_min: float = Field(..., ge=-90.0, le=90.0)
    lon_max: float = Field(..., ge=-180.0, le=180.0)
    lat_max: float = Field(..., ge=-90.0, le=90.0)


class HarvestRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100)
    bbox: BBox


app = FastAPI(title="DeepSat Arkansas Harvest Estimation API", version="1.0")


def _expanded_path(env_name: str, default: str | Path) -> Path:
    value = os.environ.get(env_name, "").strip() or str(default)
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _settings() -> dict:
    return {
        "code_root": _expanded_path("DEEPSAT_HARVEST_CODE_ROOT", DEFAULT_CODE_ROOT),
        "model_root": _expanded_path("DEEPSAT_HARVEST_MODEL_ROOT", DEFAULT_MODEL_ROOT),
        "outputs_root": _expanded_path("DEEPSAT_HARVEST_OUTPUTS_ROOT", DEFAULT_OUTPUTS_ROOT),
        "dataset_root": _expanded_path("DEEPSAT_HARVEST_DATASET_ROOT", DEFAULT_DATASET_ROOT),
        "model_window": os.environ.get("DEEPSAT_HARVEST_MODEL_WINDOW", "1year").strip(),
        "feature_set": os.environ.get("DEEPSAT_HARVEST_FEATURE_SET", "all_indices").strip(),
        "device": os.environ.get("DEEPSAT_HARVEST_DEVICE", "cpu").strip(),
    }


def _load_external_modules(code_root: Path):
    code_root_str = str(code_root)
    if code_root_str not in sys.path:
        sys.path.insert(0, code_root_str)
    try:
        from doy_prediction.tile_cnn_data import (
            build_record_from_rows,
            get_feature_names,
            read_all_crops_workbook,
        )
        from doy_prediction.tile_cnn_model import normalized_to_doy
        from doy_prediction.tile_hybrid_model import TileCNNRNNHybridRegressor
    except Exception as exc:
        raise RuntimeError(f"Could not import harvest model code from {code_root}: {exc}") from exc
    return (
        build_record_from_rows,
        get_feature_names,
        read_all_crops_workbook,
        normalized_to_doy,
        TileCNNRNNHybridRegressor,
    )


def _available_years(outputs_root: Path) -> list[int]:
    years: list[int] = []
    if not outputs_root.is_dir():
        return years
    for path in outputs_root.iterdir():
        if not path.is_dir() or not path.name.endswith("_AR"):
            continue
        try:
            years.append(int(path.name.split("_", 1)[0]))
        except ValueError:
            continue
    return sorted(years)


def _model_dir(settings: dict) -> Path:
    return settings["model_root"] / settings["model_window"] / settings["feature_set"]


def _tile_raster(tile_dir: Path) -> Path | None:
    cdl_path = tile_dir / "cdl.tif"
    if cdl_path.is_file():
        return cdl_path
    return next(iter(sorted(tile_dir.glob("*/B4_*.tif"))), None)


@lru_cache(maxsize=16)
def _tile_bounds_for_year(dataset_root: str, year: int) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    year_dir = Path(dataset_root) / f"{int(year)}_AR"
    if not year_dir.is_dir():
        raise FileNotFoundError(f"Harvest source dataset not found: {year_dir}")

    bounds_by_tile: list[tuple[str, tuple[float, float, float, float]]] = []
    for tile_dir in sorted((path for path in year_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        raster_path = _tile_raster(tile_dir)
        if raster_path is None:
            continue
        with rasterio.open(raster_path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        bounds_by_tile.append((tile_dir.name, tuple(float(value) for value in bounds)))
    return tuple(bounds_by_tile)


def _tiles_intersecting_bbox(
    bbox: BBox,
    tile_bounds: tuple[tuple[str, tuple[float, float, float, float]], ...],
) -> list[str]:
    tiles: list[str] = []
    for tile, (lon_min, lat_min, lon_max, lat_max) in tile_bounds:
        intersects = (
            lon_min <= bbox.lon_max
            and lon_max >= bbox.lon_min
            and lat_min <= bbox.lat_max
            and lat_max >= bbox.lat_min
        )
        if intersects:
            tiles.append(tile)
    return tiles


def _validate_bbox(bbox: BBox) -> None:
    if bbox.lon_min >= bbox.lon_max or bbox.lat_min >= bbox.lat_max:
        raise HTTPException(status_code=400, detail="bbox min values must be smaller than max values.")
    if (
        bbox.lon_max < HARVEST_LON_MIN
        or bbox.lon_min > HARVEST_LON_MAX
        or bbox.lat_max < HARVEST_LAT_MIN
        or bbox.lat_min > HARVEST_LAT_MAX
    ):
        raise HTTPException(status_code=400, detail="bbox does not intersect the DeepSatModels harvest coverage.")


def _checkpoint_for_crop(model_dir: Path, crop: str) -> Path:
    return model_dir / crop.replace("/", "_") / "best_model.pt"


@lru_cache(maxsize=64)
def _load_model(checkpoint_path: str, device_name: str, code_root: str):
    *_, model_class = _load_external_modules(Path(code_root))
    checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    if checkpoint.get("model_name") != "cnn_rnn_hybrid":
        raise RuntimeError(f"Unexpected model type in {checkpoint_path}: {checkpoint.get('model_name')}")
    model = model_class(
        in_channels=int(checkpoint["in_channels"]),
        **dict(checkpoint.get("model_config") or {}),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch.device(device_name))
    model.eval()
    return model


@lru_cache(maxsize=128)
def _test_mae(metrics_path: str) -> float | None:
    try:
        data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        value = (data.get("test_metrics") or {}).get("mae_mean")
        return round(float(value), 1) if value is not None else None
    except Exception:
        return None


def _doy_to_date(year: int, doy: int) -> str:
    max_doy = 366 if calendar.isleap(year) else 365
    safe_doy = max(1, min(max_doy, int(doy)))
    return (date(year, 1, 1) + timedelta(days=safe_doy - 1)).isoformat()


def _percentile(values: list[int], percentile: float) -> int:
    return int(np.rint(np.percentile(np.asarray(values, dtype=np.float32), percentile)))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    settings = _settings()
    model_dir = _model_dir(settings)
    available_years = _available_years(settings["outputs_root"])
    checkpoint_count = len(list(model_dir.glob("*/best_model.pt"))) if model_dir.is_dir() else 0
    return {
        "ready": bool(
            settings["code_root"].is_dir()
            and settings["dataset_root"].is_dir()
            and available_years
            and checkpoint_count
        ),
        "supported_region": {
            "lon_min": HARVEST_LON_MIN,
            "lat_min": HARVEST_LAT_MIN,
            "lon_max": HARVEST_LON_MAX,
            "lat_max": HARVEST_LAT_MAX,
        },
        "coverage_source": "DeepSatModels right_bottom_ar.json; tile selection uses source GeoTIFF bounds",
        "available_years": available_years,
        "model": "hybrid",
        "model_window": settings["model_window"],
        "feature_set": settings["feature_set"],
        "checkpoint_count": checkpoint_count,
        "trained_years": [2022],
        "evaluated_years": [2023],
        "request_schema": {"year": "int", "bbox": "lon_min/lat_min/lon_max/lat_max"},
    }


@app.post("/predict")
def predict(request: HarvestRequest) -> dict:
    _validate_bbox(request.bbox)
    settings = _settings()
    year_dir = settings["outputs_root"] / f"{request.year}_AR"
    model_dir = _model_dir(settings)
    if not year_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Harvest inputs are unavailable for {request.year}. Available years: {_available_years(settings['outputs_root'])}",
        )
    if not model_dir.is_dir():
        raise HTTPException(status_code=503, detail=f"Hybrid model directory not found: {model_dir}")

    try:
        tile_bounds = _tile_bounds_for_year(str(settings["dataset_root"]), request.year)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    requested_tiles = _tiles_intersecting_bbox(request.bbox, tile_bounds)
    if not requested_tiles:
        raise HTTPException(
            status_code=404,
            detail="No source GeoTIFF tile intersects the requested bbox for this year.",
        )

    try:
        (
            build_record_from_rows,
            get_feature_names,
            read_all_crops_workbook,
            normalized_to_doy,
            _,
        ) = _load_external_modules(settings["code_root"])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    records_by_crop: Dict[str, list] = defaultdict(list)
    missing_tiles: list[str] = []
    feature_names = get_feature_names(settings["feature_set"])

    for tile in requested_tiles:
        workbook = year_dir / tile / "harvest_summary_all_crops.xlsx"
        if not workbook.is_file():
            missing_tiles.append(tile)
            continue
        try:
            rows = read_all_crops_workbook(workbook)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read {workbook}: {exc}") from exc

        grouped: Dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            crop = str(row.get("Crop", "")).strip()
            if crop:
                grouped[crop].append(row)

        for crop, crop_rows in grouped.items():
            checkpoint_path = _checkpoint_for_crop(model_dir, crop)
            if not checkpoint_path.is_file():
                continue
            record = build_record_from_rows(
                crop_rows,
                year=request.year,
                tile=tile,
                crop=crop,
                source_workbook=workbook,
                feature_set=settings["feature_set"],
                feature_names=feature_names,
                min_points=2,
                require_labels=False,
                crop_window=None,
            )
            if record is not None:
                records_by_crop[crop].append(record)

    predictions: list[dict] = []
    device = torch.device(settings["device"])
    for crop in sorted(records_by_crop):
        records = records_by_crop[crop]
        checkpoint_path = _checkpoint_for_crop(model_dir, crop)
        try:
            model = _load_model(str(checkpoint_path), str(device), str(settings["code_root"]))
            batch = torch.from_numpy(np.stack([record.x for record in records])).float().to(device)
            with torch.inference_mode():
                pred_doy = normalized_to_doy(model(batch))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed for crop={crop}: {exc}") from exc

        starts = [int(value) for value in pred_doy[:, 0]]
        ends = [int(value) for value in pred_doy[:, 1]]
        start_doy = _percentile(starts, 50)
        end_doy = max(start_doy, _percentile(ends, 50))
        predictions.append(
            {
                "crop": crop,
                "harvest_start_date": _doy_to_date(request.year, start_doy),
                "harvest_end_date": _doy_to_date(request.year, end_doy),
                "start_doy": start_doy,
                "end_doy": end_doy,
                "tiles_with_crop": len(records),
                "start_date_p10": _doy_to_date(request.year, _percentile(starts, 10)),
                "start_date_p90": _doy_to_date(request.year, _percentile(starts, 90)),
                "end_date_p10": _doy_to_date(request.year, _percentile(ends, 10)),
                "end_date_p90": _doy_to_date(request.year, _percentile(ends, 90)),
                "median_observations_per_tile": _percentile(
                    [int(record.num_observations) for record in records], 50
                ),
                "model_test_mae_days": _test_mae(str(checkpoint_path.with_name("metrics.json"))),
            }
        )

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail="No crops with sufficient observations and a hybrid checkpoint were found in the selected region.",
        )

    return {
        "year": request.year,
        "bbox": request.bbox.model_dump(),
        "model": "hybrid",
        "model_window": settings["model_window"],
        "feature_set": settings["feature_set"],
        "aggregation": "median across source GeoTIFF tiles intersecting the requested bbox",
        "requested_tiles": requested_tiles,
        "tiles_with_inputs": len(requested_tiles) - len(missing_tiles),
        "missing_tiles": missing_tiles,
        "crop_count": len(predictions),
        "predictions": predictions,
        "training_note": "The selected model was trained on 2022 and evaluated on 2023.",
    }
