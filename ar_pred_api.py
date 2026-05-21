from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image


# Supported Arkansas ROI (same as app_AR.py / Download.py)
AR_ROIG = [
    [-94.7610, 36.6652],
    [-94.7610, 32.8376],
    [-89.5522, 36.6652],
    [-89.5522, 32.8376],
]
AR_LON_MIN = min(AR_ROIG[0][0], AR_ROIG[1][0])
AR_LON_MAX = max(AR_ROIG[2][0], AR_ROIG[3][0])
AR_LAT_MIN = min(AR_ROIG[1][1], AR_ROIG[3][1])
AR_LAT_MAX = max(AR_ROIG[0][1], AR_ROIG[2][1])

GRID_N = 20  # 20x20 meta-patches


def _clamp_model_month(month: int) -> int:
    # Supported models are 6..11 months. If user asks for 12, serve 11mo.
    return max(6, min(11, int(month)))


def _prediction_dir(*, pred_root: str, year: int, model_month: int) -> Path:
    return Path(pred_root) / f"AR_{year}_preds" / f"{model_month:02d}mo"


def _find_latest_model_month(pred_root: str, year: int) -> int | None:
    base = Path(pred_root) / f"AR_{year}_preds"
    if not base.exists():
        return None
    candidates: list[int] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(\d{2})mo", d.name)
        if not m:
            continue
        candidates.append(int(m.group(1)))
    return max(candidates) if candidates else None


def _meta_patches_intersecting_bbox(*, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list[str]:
    lon_min_ar = AR_LON_MIN
    lon_max_ar = AR_LON_MAX
    lat_min_ar = AR_LAT_MIN
    lat_max_ar = AR_LAT_MAX

    lon_range = np.linspace(lon_min_ar, lon_max_ar, GRID_N + 1)
    lat_range = np.linspace(lat_min_ar, lat_max_ar, GRID_N + 1)

    out: list[str] = []
    for i in range(GRID_N):
        cell_lon1, cell_lon2 = lon_range[i], lon_range[i + 1]
        for j in range(GRID_N):
            cell_lat1, cell_lat2 = lat_range[j], lat_range[j + 1]
            intersects = (cell_lon1 <= lon_max) and (cell_lon2 >= lon_min) and (cell_lat1 <= lat_max) and (cell_lat2 >= lat_min)
            if intersects:
                out.append(f"{i}_{j}")
    return out


def _make_palette_major_crops() -> tuple[np.ndarray, dict[int, str]]:
    # label_remap convention used in this repo for crop-only major classes:
    # 0=Background, 1=Corn, 2=Cotton, 3=Rice, 4=Soybeans, 5=Winter Wheat
    names = {
        0: "Background",
        1: "Corn",
        2: "Cotton",
        3: "Rice",
        4: "Soybeans",
        5: "Winter Wheat",
    }
    palette = np.array(
        [
            [0, 0, 0],         # 0 background
            [255, 215, 0],     # 1 corn (gold)
            [220, 20, 60],     # 2 cotton (crimson)
            [30, 144, 255],    # 3 rice (dodgerblue)
            [34, 139, 34],     # 4 soybeans (forestgreen)
            [255, 140, 0],     # 5 winter wheat (darkorange)
        ],
        dtype=np.uint8,
    )
    return palette, names


def _labels_to_rgb(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    labels = np.where((labels >= 0) & (labels < len(palette)), labels, 0)
    return palette[labels]


def _png_base64(rgb: np.ndarray) -> str:
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _npz_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class BBox(BaseModel):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


class PredictRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100)
    month: Optional[int] = Field(default=None, ge=1, le=12, description="Calendar month for model selection; clamped to 6..11.")
    model_month: Optional[int] = Field(default=None, ge=6, le=11, description="Override: explicitly choose 6..11mo model.")
    bbox: BBox
    output: Literal["png", "png+npz", "npz"] = "png"


class PredictResponse(BaseModel):
    year: int
    model_month: int
    meta_patches: List[str]

    height: int
    width: int
    crs: str
    transform_gdal: Tuple[float, float, float, float, float, float]

    pred_png_base64: Optional[str] = None
    pred_labels_npz_base64: Optional[str] = None
    pred_class_hist: Dict[str, int]
    class_names: Dict[str, str]


app = FastAPI(title="DeepSatModels Arkansas Predictions API", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/info")
def info():
    pred_root = os.environ.get("DEEPSAT_AR_PRED_ROOT", "/mnt/vhvkhoa_ssd/datasets")
    return {
        "pred_root": pred_root,
        "supported_region": {
            "lon_min": AR_LON_MIN,
            "lat_min": AR_LAT_MIN,
            "lon_max": AR_LON_MAX,
            "lat_max": AR_LAT_MAX,
            "grid_n": GRID_N,
        },
        "models_supported_months": [6, 7, 8, 9, 10, 11],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred_root = os.environ.get("DEEPSAT_AR_PRED_ROOT", "/mnt/vhvkhoa_ssd/datasets")

    if req.model_month is not None:
        model_month = int(req.model_month)
    elif req.month is not None:
        model_month = _clamp_model_month(req.month)
    else:
        latest = _find_latest_model_month(pred_root, req.year)
        if latest is None:
            raise HTTPException(status_code=404, detail=f"No predictions found for year={req.year} under {pred_root}.")
        model_month = int(latest)

    bbox = req.bbox
    lon_min = min(bbox.lon_min, bbox.lon_max)
    lon_max = max(bbox.lon_min, bbox.lon_max)
    lat_min = min(bbox.lat_min, bbox.lat_max)
    lat_max = max(bbox.lat_min, bbox.lat_max)

    if lon_max < AR_LON_MIN or lon_min > AR_LON_MAX or lat_max < AR_LAT_MIN or lat_min > AR_LAT_MAX:
        raise HTTPException(status_code=400, detail="Requested bbox does not intersect the supported Arkansas region.")

    meta_patches = _meta_patches_intersecting_bbox(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)
    pred_dir = _prediction_dir(pred_root=pred_root, year=req.year, model_month=model_month)
    if not pred_dir.exists():
        raise HTTPException(status_code=404, detail=f"Prediction directory not found: {pred_dir}")

    datasets = []
    for mp in meta_patches:
        fp = pred_dir / f"{mp}.tif"
        if fp.exists():
            datasets.append(rasterio.open(fp))

    if not datasets:
        raise HTTPException(status_code=404, detail="No prediction tiles found for the requested bbox.")

    # All saved preds are expected to be EPSG:3857 GeoTIFFs.
    crs = datasets[0].crs
    if crs is None:
        for ds in datasets:
            ds.close()
        raise HTTPException(status_code=500, detail="Predictions missing CRS; re-generate with georeferencing.")

    bounds_crs = transform_bounds("EPSG:4326", crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21)
    merged, out_transform = merge(datasets, bounds=bounds_crs, nodata=0)
    for ds in datasets:
        ds.close()

    pred = merged[0].astype(np.uint8)

    palette, names = _make_palette_major_crops()
    pu, pc = np.unique(pred, return_counts=True)
    pred_hist = {str(int(k)): int(v) for k, v in zip(pu, pc)}
    class_names = {str(k): v for k, v in names.items()}

    pred_png = None
    pred_npz = None
    if req.output in ("png", "png+npz"):
        pred_png = _png_base64(_labels_to_rgb(pred, palette))
    if req.output in ("npz", "png+npz"):
        pred_npz = _npz_base64(pred)

    return PredictResponse(
        year=req.year,
        model_month=model_month,
        meta_patches=meta_patches,
        height=int(pred.shape[0]),
        width=int(pred.shape[1]),
        crs=str(crs),
        transform_gdal=tuple(float(x) for x in out_transform.to_gdal()),
        pred_png_base64=pred_png,
        pred_labels_npz_base64=pred_npz,
        pred_class_hist=pred_hist,
        class_names=class_names,
    )
