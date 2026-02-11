import base64
import io
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import blosc2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

# Make relative paths work whether launched from repo root or DeepSatModels/
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data.Arkansas.dataloader import get_dataloader as get_arkansas_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint


DEFAULT_TILE_SIZE = 24

try:
    DATASET_INFO = read_yaml("data/datasets.yaml")
except Exception:
    DATASET_INFO = {}
DEFAULT_BFRAME2_SYMLINK = "../raw_arkansas/2023_bframe2"
DEFAULT_BFRAME2_ROOT = (
    DEFAULT_BFRAME2_SYMLINK
    if Path(DEFAULT_BFRAME2_SYMLINK).exists()
    else (DATASET_INFO.get("AR23", {}) or {}).get("basedir", "")
)

DEFAULT_CONFIG = (
    "models/saved_models/AR23_focal_11mo/config_file.yaml"
    if Path("models/saved_models/AR23_focal_11mo/config_file.yaml").exists()
    else "configs/Arkansas/TSViT_AR23_11mo_focal.yaml"
)
DEFAULT_CHECKPOINT = (
    "models/saved_models/AR23_focal_11mo/best.pth"
    if Path("models/saved_models/AR23_focal_11mo/best.pth").exists()
    else ""
)


def _read_b2frame(fp: str | Path, *, dtype) -> np.ndarray:
    schunk = blosc2.open(str(fp), mode="r")
    shape = np.frombuffer(schunk.vlmeta["shape"], dtype=np.int32)
    out = np.empty(shape, dtype=dtype)
    schunk.get_slice(out=out)
    return out


def _iter_tile_ids(meta_dir: Path) -> list[str]:
    label_dir = meta_dir / "label_remap"
    if not label_dir.exists():
        label_dir = meta_dir / "label"
    if not label_dir.exists():
        return []

    label_files = sorted(label_dir.glob("*_label.b2frame"))
    tile_ids = [fp.stem.replace("_label", "") for fp in label_files]

    img_dir = meta_dir / "img"
    if img_dir.exists():
        tile_ids = [tid for tid in tile_ids if (img_dir / f"{tid}_img.b2frame").exists()]
    return tile_ids


def _stitch_tiles(tile_map: dict[tuple[int, int], np.ndarray], *, tile_size: int) -> np.ndarray:
    if not tile_map:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    max_y = max(y for y, _ in tile_map.keys()) + tile_size
    max_x = max(x for _, x in tile_map.keys()) + tile_size
    mosaic = np.zeros((max_y, max_x), dtype=np.uint8)
    for (y, x), tile in tile_map.items():
        mosaic[y : y + tile_size, x : x + tile_size] = tile
    return mosaic


def _make_palette(num_classes: int) -> np.ndarray:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20", max(num_classes, 1))
    palette = (cmap(np.arange(max(num_classes, 1)))[:, :3] * 255).astype(np.uint8)
    if num_classes > 0:
        palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


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


@lru_cache(maxsize=2)
def _load_model_cached(config_path: str, checkpoint_override: str | None):
    cfg = read_yaml(config_path)
    device_ids = (cfg.get("DEVICE", {}) or {}).get("device_id", [0])
    device = get_device(device_ids, allow_cpu=True)
    net = get_model(cfg, device)

    ckpt = checkpoint_override or (cfg.get("CHECKPOINT", {}) or {}).get("load_from_checkpoint")
    if ckpt:
        load_from_checkpoint(net, ckpt, partial_restore=False, device=device)

    net.to(device)
    net.eval()
    return net, cfg, device, ckpt


@torch.no_grad()
def _infer_meta_patch(
    *,
    meta_patch: str,
    dataset_root: str,
    config_path: str,
    checkpoint_override: str | None,
    tile_size: int,
    max_doy: int,
    batch_size: int,
    num_workers: int,
    include_ground_truth: bool,
):
    net, cfg, device, ckpt_used = _load_model_cached(config_path, checkpoint_override)

    meta_dir = Path(dataset_root) / meta_patch
    if not meta_dir.exists():
        raise FileNotFoundError(f"Meta-patch not found: {meta_dir}")

    tile_ids = _iter_tile_ids(meta_dir)
    if not tile_ids:
        raise FileNotFoundError(f"No label tiles found under {meta_dir}/label_remap (or /label).")

    df = pd.DataFrame([{"meta_patch": meta_patch, "tile_id": tid, "split": "test"} for tid in tile_ids])
    dataloader = get_arkansas_dataloader(
        paths_file=df,
        root_dir=dataset_root,
        max_doy=max_doy,
        split="test",
        transform=PASTIS_segmentation_transform(cfg["MODEL"], is_training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        return_paths=True,
    )

    pred_tiles: dict[tuple[int, int], np.ndarray] = {}
    for batch, paths in dataloader:
        logits = net(batch["inputs"].to(device))
        preds = logits.argmax(1).cpu().numpy().astype(np.uint8)
        for i, fp in enumerate(paths):
            tile_id = Path(fp).stem.replace("_img", "")
            y, x = map(int, tile_id.split("_"))
            pred_tiles[(y, x)] = preds[i]

    pred_mosaic = _stitch_tiles(pred_tiles, tile_size=tile_size)

    gt_mosaic = None
    if include_ground_truth:
        label_dir = meta_dir / "label_remap"
        if not label_dir.exists():
            label_dir = meta_dir / "label"
        gt_tiles: dict[tuple[int, int], np.ndarray] = {}
        for tid in tile_ids:
            label_fp = label_dir / f"{tid}_label.b2frame"
            if not label_fp.exists():
                continue
            y, x = map(int, tid.split("_"))
            arr = _read_b2frame(label_fp, dtype=np.uint8)
            gt_tiles[(y, x)] = np.asarray(arr).squeeze()
        gt_mosaic = _stitch_tiles(gt_tiles, tile_size=tile_size)

    return pred_mosaic, gt_mosaic, cfg, ckpt_used


class PredictRequest(BaseModel):
    lon_id: int | None = Field(default=None, description="Meta-patch X index (lon grid index).")
    lat_id: int | None = Field(default=None, description="Meta-patch Y index (lat grid index).")
    meta_patch: str | None = Field(default=None, description="Explicit meta-patch name like '10_3'.")

    dataset_root: str | None = Field(default=None, description="Root dir containing <meta_patch>/{img,label_remap,doy}.")
    config_path: str = Field(default=DEFAULT_CONFIG, description="Model config YAML path (repo-relative or absolute).")
    checkpoint_path: str | None = Field(default=DEFAULT_CHECKPOINT or None, description="Checkpoint (.pth).")

    max_doy: int = Field(default=330, ge=1, le=366)
    batch_size: int = Field(default=64, ge=1, le=4096)
    num_workers: int = Field(default=4, ge=0, le=32)
    include_ground_truth: bool = Field(default=False)

    output: Literal["png", "png+npz"] = Field(default="png")

    def resolved_meta_patch(self) -> str:
        if self.meta_patch:
            return self.meta_patch
        if self.lon_id is None or self.lat_id is None:
            raise ValueError("Provide either meta_patch or both lon_id + lat_id.")
        return f"{self.lon_id}_{self.lat_id}"


class PredictResponse(BaseModel):
    meta_patch: str
    config_path: str
    checkpoint_path: str | None
    dataset_root: str

    height: int
    width: int
    num_classes: int

    pred_png_base64: str
    gt_png_base64: str | None = None

    pred_labels_npz_base64: str | None = None
    gt_labels_npz_base64: str | None = None

    pred_class_hist: dict[str, int]
    gt_class_hist: dict[str, int] | None = None


class ServerInfo(BaseModel):
    dataset_root: str
    config_path: str
    checkpoint_path: str | None


app = FastAPI(title="DeepSatModels Arkansas Inference API", version="0.1.0")
_INFER_LOCK = threading.Lock()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/info", response_model=ServerInfo)
def info():
    return ServerInfo(
        dataset_root=os.environ.get("DEEPSAT_AR_DATASET_ROOT", DEFAULT_BFRAME2_ROOT),
        config_path=os.environ.get("DEEPSAT_AR_CONFIG", DEFAULT_CONFIG),
        checkpoint_path=os.environ.get("DEEPSAT_AR_CHECKPOINT", DEFAULT_CHECKPOINT) or None,
    )


@app.get("/meta_patches")
def meta_patches(dataset_root: str | None = None):
    root = dataset_root or os.environ.get("DEEPSAT_AR_DATASET_ROOT", DEFAULT_BFRAME2_ROOT)
    if not root:
        raise HTTPException(status_code=400, detail="dataset_root is required (or set DEEPSAT_AR_DATASET_ROOT).")
    p = Path(root)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"dataset_root not found: {root}")
    return {"dataset_root": root, "meta_patches": sorted([d.name for d in p.iterdir() if d.is_dir()])}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        meta_patch = req.resolved_meta_patch()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dataset_root = req.dataset_root or os.environ.get("DEEPSAT_AR_DATASET_ROOT", DEFAULT_BFRAME2_ROOT)
    if not dataset_root:
        raise HTTPException(status_code=400, detail="dataset_root is required (or set DEEPSAT_AR_DATASET_ROOT).")

    config_path = os.environ.get("DEEPSAT_AR_CONFIG", req.config_path)
    checkpoint_path = os.environ.get("DEEPSAT_AR_CHECKPOINT", req.checkpoint_path or "") or None

    try:
        with _INFER_LOCK:
            pred, gt, cfg, ckpt_used = _infer_meta_patch(
                meta_patch=meta_patch,
                dataset_root=dataset_root,
                config_path=config_path,
                checkpoint_override=checkpoint_path,
                tile_size=DEFAULT_TILE_SIZE,
                max_doy=int(req.max_doy),
                batch_size=int(req.batch_size),
                num_workers=int(req.num_workers),
                include_ground_truth=bool(req.include_ground_truth),
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    num_classes = int((cfg.get("MODEL", {}) or {}).get("num_classes", int(pred.max()) + 1))
    palette = _make_palette(num_classes)
    pred_png = _png_base64(_labels_to_rgb(pred, palette))
    gt_png = _png_base64(_labels_to_rgb(gt, palette)) if (req.include_ground_truth and gt is not None) else None

    pu, pc = np.unique(pred, return_counts=True)
    pred_hist = {str(int(k)): int(v) for k, v in zip(pu, pc)}
    gt_hist = None
    if req.include_ground_truth and gt is not None:
        gu, gc = np.unique(gt, return_counts=True)
        gt_hist = {str(int(k)): int(v) for k, v in zip(gu, gc)}

    pred_npz = _npz_base64(pred) if req.output == "png+npz" else None
    gt_npz = _npz_base64(gt) if (req.output == "png+npz" and req.include_ground_truth and gt is not None) else None

    return PredictResponse(
        meta_patch=meta_patch,
        config_path=config_path,
        checkpoint_path=ckpt_used or checkpoint_path,
        dataset_root=dataset_root,
        height=int(pred.shape[0]),
        width=int(pred.shape[1]),
        num_classes=num_classes,
        pred_png_base64=pred_png,
        gt_png_base64=gt_png,
        pred_labels_npz_base64=pred_npz,
        gt_labels_npz_base64=gt_npz,
        pred_class_hist=pred_hist,
        gt_class_hist=gt_hist,
    )
