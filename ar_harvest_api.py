from __future__ import annotations

import calendar
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rasterio.warp import transform_bounds

# Crop Identification's Arkansas 20x20 grid.
HARVEST_LON_MIN = -94.7610
HARVEST_LON_MAX = -89.5522
HARVEST_LAT_MIN = 32.8376
HARVEST_LAT_MAX = 36.6652

APP_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS_ROOT = APP_DIR / "runtime_data" / "harvest_predictions"
DEFAULT_DATASET_ROOT = APP_DIR / "runtime_data" / "harvest_source"


class BBox(BaseModel):
    lon_min: float = Field(..., ge=-180.0, le=180.0)
    lat_min: float = Field(..., ge=-90.0, le=90.0)
    lon_max: float = Field(..., ge=-180.0, le=180.0)
    lat_max: float = Field(..., ge=-90.0, le=90.0)


class HarvestRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100)
    bbox: BBox


@dataclass(frozen=True)
class PrecomputedPrediction:
    year: int
    tile: str
    crop: str
    start_doy: int
    end_doy: int
    num_observations: int | None
    model_test_mae_days: float | None


app = FastAPI(title="DeepSat Arkansas Harvest Estimation API", version="1.1")


def _expanded_path(env_name: str, default: str | Path) -> Path:
    value = os.environ.get(env_name, "").strip() or str(default)
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _settings() -> dict:
    return {
        "predictions_root": _expanded_path(
            "DEEPSAT_HARVEST_PRED_ROOT", DEFAULT_PREDICTIONS_ROOT
        ),
        "dataset_root": _expanded_path(
            "DEEPSAT_HARVEST_DATASET_ROOT", DEFAULT_DATASET_ROOT
        ),
        "model_window": os.environ.get("DEEPSAT_HARVEST_MODEL_WINDOW", "1year").strip(),
        "feature_set": os.environ.get(
            "DEEPSAT_HARVEST_FEATURE_SET", "all_indices"
        ).strip(),
    }


def _prediction_path(
    *, predictions_root: Path, year: int, model_window: str, feature_set: str
) -> Path:
    output_path = (
        predictions_root
        / f"output_{int(year)}"
        / model_window
        / feature_set
        / "predictions.csv"
    )
    legacy_path = (
        predictions_root
        / str(int(year))
        / model_window
        / feature_set
        / "predictions.csv"
    )
    if output_path.is_file() or not legacy_path.is_file():
        return output_path
    return legacy_path


def _metadata_path(prediction_path: Path) -> Path:
    return prediction_path.with_name("metadata.json")


def _available_years(
    *, predictions_root: Path, model_window: str, feature_set: str
) -> list[int]:
    if not predictions_root.is_dir():
        return []

    years: set[int] = set()
    for year_dir in predictions_root.iterdir():
        if not year_dir.is_dir():
            continue
        year_name = year_dir.name
        if year_name.startswith("output_"):
            year_name = year_name.removeprefix("output_")
        try:
            year = int(year_name)
        except ValueError:
            continue
        path = _prediction_path(
            predictions_root=predictions_root,
            year=year,
            model_window=model_window,
            feature_set=feature_set,
        )
        if path.is_file():
            years.add(year)
    return sorted(years)


def _tile_raster(tile_dir: Path) -> Path | None:
    cdl_path = tile_dir / "cdl.tif"
    if cdl_path.is_file():
        return cdl_path
    return next(iter(sorted(tile_dir.glob("*/B4_*.tif"))), None)


@lru_cache(maxsize=16)
def _tile_bounds_for_year(
    dataset_root: str, year: int
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    year_dir = Path(dataset_root) / f"{int(year)}_AR"
    if not year_dir.is_dir():
        raise FileNotFoundError(f"Harvest source dataset not found: {year_dir}")

    bounds_by_tile: list[tuple[str, tuple[float, float, float, float]]] = []
    for tile_dir in sorted(
        (path for path in year_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
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


def _tile_bounds_from_metadata(
    metadata: dict,
) -> tuple[tuple[str, tuple[float, float, float, float]], ...] | None:
    raw_bounds = metadata.get("tile_bounds_wgs84")
    if raw_bounds is None:
        return None
    if not isinstance(raw_bounds, dict):
        raise TypeError("tile_bounds_wgs84 in prediction metadata must be an object")

    tile_bounds: list[tuple[str, tuple[float, float, float, float]]] = []
    for tile, raw_values in raw_bounds.items():
        if not isinstance(raw_values, list | tuple) or len(raw_values) != 4:
            raise ValueError(f"Invalid WGS84 bounds for tile {tile!r}")
        values = tuple(float(value) for value in raw_values)
        if values[0] >= values[2] or values[1] >= values[3]:
            raise ValueError(f"Invalid WGS84 bounds order for tile {tile!r}")
        tile_bounds.append((str(tile), values))
    if not tile_bounds:
        raise ValueError("tile_bounds_wgs84 in prediction metadata is empty")
    return tuple(sorted(tile_bounds))


def _supported_region_from_tile_bounds(
    tile_bounds: tuple[tuple[str, tuple[float, float, float, float]], ...],
) -> dict[str, float]:
    values = [bounds for _, bounds in tile_bounds]
    return {
        "lon_min": min(bounds[0] for bounds in values),
        "lat_min": min(bounds[1] for bounds in values),
        "lon_max": max(bounds[2] for bounds in values),
        "lat_max": max(bounds[3] for bounds in values),
    }


def _validate_bbox(bbox: BBox) -> None:
    if bbox.lon_min >= bbox.lon_max or bbox.lat_min >= bbox.lat_max:
        raise HTTPException(
            status_code=400, detail="bbox min values must be smaller than max values."
        )
    if (
        bbox.lon_max < HARVEST_LON_MIN
        or bbox.lon_min > HARVEST_LON_MAX
        or bbox.lat_max < HARVEST_LAT_MIN
        or bbox.lat_min > HARVEST_LAT_MAX
    ):
        raise HTTPException(
            status_code=400,
            detail="bbox does not intersect the DeepSatModels harvest coverage.",
        )


def _parse_optional_int(row: dict[str, str], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return int(value)
    return None


def _parse_optional_float(row: dict[str, str], name: str) -> float | None:
    value = str(row.get(name, "")).strip()
    return float(value) if value else None


@lru_cache(maxsize=16)
def _load_predictions_cached(
    path_str: str, modified_ns: int
) -> tuple[PrecomputedPrediction, ...]:
    del modified_ns
    path = Path(path_str)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"year", "tile", "crop", "pred_start_doy", "pred_end_doy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

        predictions: list[PrecomputedPrediction] = []
        seen_keys: set[tuple[int, str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            year = int(row["year"])
            tile = str(row["tile"]).strip()
            crop = str(row["crop"]).strip()
            start_doy = int(row["pred_start_doy"])
            end_doy = int(row["pred_end_doy"])
            if not tile or not crop:
                raise ValueError(f"Blank tile or crop at {path}:{line_number}")
            if not 1 <= start_doy <= 366 or not 1 <= end_doy <= 366:
                raise ValueError(f"Invalid prediction DOY at {path}:{line_number}")
            if start_doy > end_doy:
                raise ValueError(
                    f"Harvest start is after harvest end at {path}:{line_number}"
                )

            key = (year, tile, crop)
            if key in seen_keys:
                raise ValueError(f"Duplicate year/tile/crop row {key} in {path}")
            seen_keys.add(key)
            predictions.append(
                PrecomputedPrediction(
                    year=year,
                    tile=tile,
                    crop=crop,
                    start_doy=start_doy,
                    end_doy=end_doy,
                    num_observations=_parse_optional_int(
                        row,
                        (
                            "num_observations",
                            "cnn_num_observations",
                            "rnn_num_observations",
                        ),
                    ),
                    model_test_mae_days=_parse_optional_float(
                        row, "model_test_mae_days"
                    ),
                )
            )
    return tuple(predictions)


def _load_predictions(path: Path) -> tuple[PrecomputedPrediction, ...]:
    stat = path.stat()
    return _load_predictions_cached(str(path), stat.st_mtime_ns)


def _read_metadata(prediction_path: Path) -> dict:
    path = _metadata_path(prediction_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prediction metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Prediction metadata must contain an object: {path}")
    return data


def _doy_to_date(year: int, doy: int) -> str:
    max_doy = 366 if calendar.isleap(year) else 365
    safe_doy = max(1, min(max_doy, int(doy)))
    return (date(year, 1, 1) + timedelta(days=safe_doy - 1)).isoformat()


def _percentile(values: list[int], percentile: float) -> int:
    return int(np.rint(np.percentile(np.asarray(values, dtype=np.float32), percentile)))


def _artifact_summaries(settings: dict, available_years: list[int]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for year in available_years:
        path = _prediction_path(
            predictions_root=settings["predictions_root"],
            year=year,
            model_window=settings["model_window"],
            feature_set=settings["feature_set"],
        )
        try:
            metadata = _read_metadata(path)
            tile_bounds = _tile_bounds_from_metadata(metadata)
            if tile_bounds is None:
                tile_bounds = _tile_bounds_for_year(str(settings["dataset_root"]), year)
            supported_region = _supported_region_from_tile_bounds(tile_bounds)
        except (OSError, TypeError, ValueError, FileNotFoundError):
            metadata = {}
            tile_bounds = ()
            supported_region = None
        summaries[str(year)] = {
            "model": str(metadata.get("model", "hybrid")),
            "model_name": str(metadata.get("model_name", "cnn_rnn_hybrid")),
            "trained_years": list(metadata.get("trained_years", [])),
            "evaluated_years": list(metadata.get("evaluated_years", [])),
            "supported_region": supported_region,
            "tile_count": len(tile_bounds),
            "ready": bool(tile_bounds),
        }
    return summaries


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    settings = _settings()
    available_years = _available_years(
        predictions_root=settings["predictions_root"],
        model_window=settings["model_window"],
        feature_set=settings["feature_set"],
    )
    artifacts = _artifact_summaries(settings, available_years)
    return {
        "ready": bool(available_years)
        and all(artifact["ready"] for artifact in artifacts.values()),
        "serving_mode": "precomputed",
        "predictions_root": str(settings["predictions_root"]),
        "supported_region": {
            "lon_min": HARVEST_LON_MIN,
            "lat_min": HARVEST_LAT_MIN,
            "lon_max": HARVEST_LON_MAX,
            "lat_max": HARVEST_LAT_MAX,
        },
        "coverage_source": "Crop Identification Arkansas 20x20 grid; each artifact stores its source GeoTIFF bounds",
        "available_years": available_years,
        "artifacts": artifacts,
        "model": "hybrid",
        "model_window": settings["model_window"],
        "feature_set": settings["feature_set"],
        "prediction_file_count": len(available_years),
        "request_schema": {"year": "int", "bbox": "lon_min/lat_min/lon_max/lat_max"},
    }


@app.post("/predict")
def predict(request: HarvestRequest) -> dict:
    _validate_bbox(request.bbox)
    settings = _settings()
    prediction_path = _prediction_path(
        predictions_root=settings["predictions_root"],
        year=request.year,
        model_window=settings["model_window"],
        feature_set=settings["feature_set"],
    )
    if not prediction_path.is_file():
        available_years = _available_years(
            predictions_root=settings["predictions_root"],
            model_window=settings["model_window"],
            feature_set=settings["feature_set"],
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Precomputed harvest predictions are unavailable for {request.year} at {prediction_path}. "
                f"Available years: {available_years}"
            ),
        )

    try:
        all_predictions = _load_predictions(prediction_path)
        metadata = _read_metadata(prediction_path)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Invalid precomputed predictions: {exc}"
        ) from exc

    try:
        tile_bounds = _tile_bounds_from_metadata(metadata)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Invalid precomputed prediction bounds: {exc}"
        ) from exc
    if tile_bounds is None:
        try:
            tile_bounds = _tile_bounds_for_year(
                str(settings["dataset_root"]), request.year
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    requested_tiles = _tiles_intersecting_bbox(request.bbox, tile_bounds)
    if not requested_tiles:
        raise HTTPException(
            status_code=404,
            detail="No source GeoTIFF tile intersects the requested bbox for this year.",
        )

    requested_tile_set = set(requested_tiles)
    rows_by_crop: dict[str, list[PrecomputedPrediction]] = defaultdict(list)
    tiles_with_predictions: set[str] = set()
    for row in all_predictions:
        if row.year == request.year and row.tile in requested_tile_set:
            rows_by_crop[row.crop].append(row)
            tiles_with_predictions.add(row.tile)

    predictions: list[dict] = []
    for crop in sorted(rows_by_crop):
        rows = rows_by_crop[crop]
        starts = [row.start_doy for row in rows]
        ends = [row.end_doy for row in rows]
        observation_counts = [
            row.num_observations for row in rows if row.num_observations is not None
        ]
        test_mae_values = [
            row.model_test_mae_days
            for row in rows
            if row.model_test_mae_days is not None
        ]
        start_doy = _percentile(starts, 50)
        end_doy = max(start_doy, _percentile(ends, 50))
        predictions.append(
            {
                "crop": crop,
                "harvest_start_date": _doy_to_date(request.year, start_doy),
                "harvest_end_date": _doy_to_date(request.year, end_doy),
                "start_doy": start_doy,
                "end_doy": end_doy,
                "tiles_with_crop": len(rows),
                "start_date_p10": _doy_to_date(request.year, _percentile(starts, 10)),
                "start_date_p90": _doy_to_date(request.year, _percentile(starts, 90)),
                "end_date_p10": _doy_to_date(request.year, _percentile(ends, 10)),
                "end_date_p90": _doy_to_date(request.year, _percentile(ends, 90)),
                "median_observations_per_tile": (
                    _percentile([int(value) for value in observation_counts], 50)
                    if observation_counts
                    else None
                ),
                "model_test_mae_days": round(float(test_mae_values[0]), 1)
                if test_mae_values
                else None,
            }
        )

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail="No precomputed crop predictions were found in the selected region.",
        )

    missing_tiles = sorted(requested_tile_set - tiles_with_predictions)
    return {
        "year": request.year,
        "bbox": request.bbox.model_dump(),
        "model": str(metadata.get("model", "hybrid")),
        "model_window": settings["model_window"],
        "feature_set": settings["feature_set"],
        "serving_mode": "precomputed",
        "aggregation": "median across source GeoTIFF tiles intersecting the requested bbox",
        "requested_tiles": requested_tiles,
        "tiles_with_inputs": len(tiles_with_predictions),
        "missing_tiles": missing_tiles,
        "crop_count": len(predictions),
        "predictions": predictions,
        "training_note": str(
            metadata.get(
                "training_note",
                "The selected model was trained on 2022 and evaluated on 2023.",
            )
        ),
    }
