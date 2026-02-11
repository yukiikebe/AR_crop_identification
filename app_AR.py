import os
import sys
import base64
from datetime import datetime, timedelta
from pathlib import Path

import blosc2
import folium
from folium.plugins import Draw
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import torch

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


MAPBOX_TOKEN = os.getenv("DEEPSAT_MAPBOX_TOKEN") or os.getenv("MAPBOX_TOKEN")

DEFAULT_TILE_SIZE = 24
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
AR_BOUNDS = [[AR_LAT_MIN, AR_LON_MIN], [AR_LAT_MAX, AR_LON_MAX]]
AR_CENTER = [(AR_LAT_MIN + AR_LAT_MAX) / 2.0, (AR_LON_MIN + AR_LON_MAX) / 2.0]

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
    # Deterministic palette; good enough for quick visual inspection.
    cmap = plt.get_cmap("tab20", max(num_classes, 1))
    palette = (cmap(np.arange(max(num_classes, 1)))[:, :3] * 255).astype(np.uint8)
    if num_classes > 0:
        palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def _labels_to_rgb(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    labels = np.where((labels >= 0) & (labels < len(palette)), labels, 0)
    return palette[labels]


def _api_predict_meta_patch(
    *,
    backend_url: str,
    meta_patch: str,
    dataset_root: str,
    config_path: str,
    checkpoint_path: str | None,
    max_doy: int,
    batch_size: int,
    num_workers: int,
    include_ground_truth: bool,
):
    url = backend_url.rstrip("/") + "/predict"
    payload = {
        "meta_patch": meta_patch,
        "dataset_root": dataset_root,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "max_doy": int(max_doy),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "include_ground_truth": bool(include_ground_truth),
        "output": "png",
    }
    resp = requests.post(url, json=payload, timeout=600)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    data = resp.json()
    pred_png = base64.b64decode(data["pred_png_base64"])
    gt_png = base64.b64decode(data["gt_png_base64"]) if data.get("gt_png_base64") else None
    return pred_png, gt_png, data


@st.cache_resource(show_spinner=False)
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
    net, cfg, device, _ = _load_model_cached(config_path, checkpoint_override)

    meta_dir = Path(dataset_root) / meta_patch
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

    return pred_mosaic, gt_mosaic, cfg


def get_map():
    st.markdown("#### Select Region of Interest on the map")

    if MAPBOX_TOKEN:
        tiles_url = (
            "https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}"
            f"?access_token={MAPBOX_TOKEN}"
        )
        attr = "Mapbox"
    else:
        tiles_url = "OpenStreetMap"
        attr = "OpenStreetMap"
    m = folium.Map(
        location=AR_CENTER,
        zoom_start=8,
        tiles=tiles_url,
        attr=attr,
        min_lat=AR_LAT_MIN,
        max_lat=AR_LAT_MAX,
        min_lon=AR_LON_MIN,
        max_lon=AR_LON_MAX,
        max_bounds=True,
    )

    folium.Rectangle(bounds=AR_BOUNDS, color="#ff4b4b", weight=2, fill=False, tooltip="Supported region").add_to(m)
    m.fit_bounds(AR_BOUNDS)

    Draw(export=True).add_to(m)

    st.sidebar.title("Location Search")
    query = st.sidebar.text_input("Enter a location (city, state, country):")
    geolocator = Nominatim(user_agent="geoapp")

    if query:
        location = geolocator.geocode(query)
        if location:
            in_bounds = (AR_LAT_MIN <= location.latitude <= AR_LAT_MAX) and (
                AR_LON_MIN <= location.longitude <= AR_LON_MAX
            )
            if in_bounds:
                m.location = [location.latitude, location.longitude]
                m.zoom_start = 12
            else:
                st.sidebar.error("Location is outside the supported Arkansas region.")
                m.fit_bounds(AR_BOUNDS)
        else:
            st.sidebar.error("Location not found. Showing default location.")
            m.fit_bounds(AR_BOUNDS)

    return st_folium(m, width=700, height=600)


def get_start_end_date():
    st.sidebar.title("Date Range (raw visualization only)")
    start_date = st.sidebar.date_input("Start date:", datetime(2023, 1, 1))
    end_date = st.sidebar.date_input("End date:", start_date + timedelta(days=364))
    vis_day = st.sidebar.date_input("Visualization date:", start_date + timedelta(days=30))

    if start_date > end_date or vis_day > end_date or vis_day < start_date:
        st.sidebar.error("Invalid date range.")

    return str(start_date), str(end_date), str(vis_day)


def get_roi(map_data):
    if (
        map_data
        and isinstance(map_data, dict)
        and "all_drawings" in map_data
        and map_data["all_drawings"] is not None
        and len(map_data["all_drawings"]) > 0
    ):
        drawing = map_data["all_drawings"][0]
        if not isinstance(drawing, dict):
            return None

        geom = drawing.get("geometry", drawing)
        if not isinstance(geom, dict):
            return None

        coords = geom.get("coordinates")
        if not coords:
            return None

        # Polygon -> coordinates: [ [ [lon,lat], ... ] ]
        return coords[0]
    return None


def get_subregion_indices_in_roi(ar_roig, user_roi):
    lon_min_ar = min(ar_roig[0][0], ar_roig[1][0])
    lon_max_ar = max(ar_roig[2][0], ar_roig[3][0])
    lat_min_ar = min(ar_roig[1][1], ar_roig[3][1])
    lat_max_ar = max(ar_roig[0][1], ar_roig[2][1])

    lon_range_ar = np.linspace(lon_min_ar, lon_max_ar, 21)
    lat_range_ar = np.linspace(lat_min_ar, lat_max_ar, 21)

    lon_centers = (lon_range_ar[:-1] + lon_range_ar[1:]) / 2
    lat_centers = (lat_range_ar[:-1] + lat_range_ar[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    lon_min_user = min(user_roi[0][0], user_roi[1][0])
    lon_max_user = max(user_roi[2][0], user_roi[3][0])
    lat_min_user = min(user_roi[1][1], user_roi[3][1])
    lat_max_user = max(user_roi[0][1], user_roi[2][1])

    mask = (lon_grid >= lon_min_user) & (lon_grid <= lon_max_user) & (lat_grid >= lat_min_user) & (lat_grid <= lat_max_user)
    return [tuple(idx) for idx in np.argwhere(mask)]


def find_closest_day(query_day, dates):
    query_day = datetime.strptime(query_day, "%Y-%m-%d")
    closest_date, smallest_diff = None, float("inf")
    for date in dates:
        current_date = datetime.strptime(date, "%Y-%m-%d")
        diff = abs((current_date - query_day).days)
        if diff < smallest_diff:
            smallest_diff = diff
            closest_date = date
    return closest_date


def retrieve_band(subregion_indices, vis_day, raw_root, band_type="TCI"):
    all_imgs = []
    ext_type = "jpg" if band_type == "TCI" else "png"

    for lat_id, lon_id in subregion_indices:
        subregion_path = os.path.join(raw_root, f"{lon_id}_{lat_id}")
        if not os.path.isdir(subregion_path):
            continue

        dates = [d for d in os.listdir(subregion_path) if os.path.isdir(os.path.join(subregion_path, d))]
        closest_date = find_closest_day(vis_day, dates)
        if not closest_date:
            continue

        band_path = os.path.join(subregion_path, closest_date, f"{band_type}_{closest_date}.{ext_type}")
        if not os.path.isfile(band_path):
            continue
        all_imgs.append([lat_id, lon_id, plt.imread(band_path)])

    all_imgs.sort(key=lambda x: (-x[0], x[1]))
    reorganized_imgs, current_lat, current_row = [], None, []
    for img in all_imgs:
        lat_id = img[0]
        if lat_id != current_lat:
            if current_row:
                reorganized_imgs.append(current_row)
            current_row = [img]
            current_lat = lat_id
        else:
            current_row.append(img)
    if current_row:
        reorganized_imgs.append(current_row)

    return reorganized_imgs


def app():
    st.title("Demo for Satellite Project for Arkansas (bframe2)")

    map_data = get_map()

    with st.sidebar:
        st.subheader("Paths")
        config_file = st.text_input("Model config (.yaml)", value=DEFAULT_CONFIG)
        checkpoint_override = st.text_input("Checkpoint override (.pth, optional)", value=DEFAULT_CHECKPOINT).strip() or None
        dataset_root = st.text_input("BFrame2 dataset root", value=DEFAULT_BFRAME2_ROOT)
        raw_root = st.text_input("Raw tiles root (optional, for Visualize)", value="")

        st.subheader("Backend (optional)")
        use_backend = st.checkbox("Use FastAPI backend for inference", value=False)
        backend_url = st.text_input("Backend URL", value="http://localhost:8000", disabled=not use_backend).strip()

        st.subheader("Runtime")
        # Preview config defaults (optional)
        try:
            cfg_preview = read_yaml(config_file)
        except Exception:
            cfg_preview = {}
        default_max_doy = int((cfg_preview.get("DATASETS", {}) or {}).get("max_doy", 366))
        default_batch = int((((cfg_preview.get("DATASETS", {}) or {}).get("eval", {}) or {}).get("batch_size", 256)))
        default_workers = int((((cfg_preview.get("DATASETS", {}) or {}).get("eval", {}) or {}).get("num_workers", 4)))

        max_doy = st.number_input("Max DOY (<=365 to truncate)", min_value=1, max_value=366, value=default_max_doy, step=1)
        batch_size = st.number_input("Batch size", min_value=1, max_value=4096, value=default_batch, step=1)
        num_workers = st.number_input("DataLoader workers", min_value=0, max_value=32, value=default_workers, step=1)
        include_ground_truth = st.checkbox("Show ground truth (label_remap)", value=True)
        show_debug = st.checkbox("Show debug stats", value=True)

        if not checkpoint_override and not ((cfg_preview.get("CHECKPOINT", {}) or {}).get("load_from_checkpoint")):
            st.warning("No checkpoint selected. Predictions will be random.")

        start_date, end_date, vis_day = get_start_end_date()
        visualize_trigger = st.button("Visualize")
        predict_trigger = st.button("Analyze")

    user_roig = get_roi(map_data)

    if visualize_trigger:
        if not raw_root:
            st.warning("Set 'Raw tiles root' to enable visualization (TCI_*.jpg).")
        elif not user_roig:
            st.warning("Draw a rectangle on the map first.")
        else:
            subregion_indices = get_subregion_indices_in_roi(AR_ROIG, user_roig)
            all_imgs = retrieve_band(subregion_indices, vis_day, raw_root)
            for img_row in all_imgs:
                cols = st.columns(len(img_row))
                for i, (lat_id, lon_id, img) in enumerate(img_row):
                    with cols[i]:
                        st.image(img, caption=f"{lon_id}_{lat_id}", use_container_width=True)

    if predict_trigger:
        if not user_roig:
            st.warning("Draw a rectangle on the map first.")
            return
        if not dataset_root:
            st.error("Set 'BFrame2 dataset root' (e.g. AR_2023_bframe2).")
            return

        subregion_indices = get_subregion_indices_in_roi(AR_ROIG, user_roig)
        if not subregion_indices:
            st.warning("ROI does not intersect the Arkansas 20x20 meta-patch grid.")
            return

        # Use config value for palette size.
        try:
            cfg_for_palette = read_yaml(config_file)
        except Exception:
            cfg_for_palette = {}
        num_classes = int((cfg_for_palette.get("MODEL", {}) or {}).get("num_classes", 2))
        palette = _make_palette(num_classes)

        with st.spinner(f"Running inference for {len(subregion_indices)} meta-patches…"):
            for lat_id, lon_id in subregion_indices:
                meta_patch = f"{lon_id}_{lat_id}"
                st.markdown(f"### Meta-patch `{meta_patch}`")
                if use_backend:
                    try:
                        pred_png, gt_png, api_dbg = _api_predict_meta_patch(
                            backend_url=backend_url,
                            meta_patch=meta_patch,
                            dataset_root=dataset_root,
                            config_path=config_file,
                            checkpoint_path=checkpoint_override,
                            max_doy=int(max_doy),
                            batch_size=int(batch_size),
                            num_workers=int(num_workers),
                            include_ground_truth=include_ground_truth,
                        )
                    except Exception as exc:
                        st.error(f"Backend failed for {meta_patch}: {exc}")
                        continue

                    if include_ground_truth and gt_png is not None:
                        st.image([pred_png, gt_png], caption=["Prediction", "Ground truth"], use_container_width=True)
                    else:
                        st.image(pred_png, caption="Prediction", use_container_width=True)
                else:
                    try:
                        pred, gt, _ = _infer_meta_patch(
                            meta_patch=meta_patch,
                            dataset_root=dataset_root,
                            config_path=config_file,
                            checkpoint_override=checkpoint_override,
                            tile_size=DEFAULT_TILE_SIZE,
                            max_doy=int(max_doy),
                            batch_size=int(batch_size),
                            num_workers=int(num_workers),
                            include_ground_truth=include_ground_truth,
                        )
                    except Exception as exc:
                        st.error(f"Failed for {meta_patch}: {exc}")
                        continue

                    pred_rgb = _labels_to_rgb(pred, palette)
                    if include_ground_truth and gt is not None:
                        gt_rgb = _labels_to_rgb(gt, palette)
                        st.image([pred_rgb, gt_rgb], caption=["Prediction", "Ground truth"], use_container_width=True)
                    else:
                        st.image(pred_rgb, caption="Prediction", use_container_width=True)

                if show_debug:
                    # Model/config summary
                    if use_backend:
                        st.code(
                            f"backend_url={backend_url}\n"
                            f"config={api_dbg.get('config_path')}\n"
                            f"checkpoint={api_dbg.get('checkpoint_path')}\n"
                            f"dataset_root={api_dbg.get('dataset_root')}\n"
                            f"num_classes={api_dbg.get('num_classes')}\n",
                            language="text",
                        )
                    else:
                        try:
                            _, cfg_dbg, _, ckpt_used = _load_model_cached(config_file, checkpoint_override)
                            model_dbg = cfg_dbg.get("MODEL", {}) or {}
                            st.code(
                                f"config={config_file}\n"
                                f"checkpoint={ckpt_used}\n"
                                f"dataset_root={dataset_root}\n"
                                f"num_classes={model_dbg.get('num_classes')}\n"
                                f"max_seq_len={model_dbg.get('max_seq_len')}\n"
                                f"num_channels={model_dbg.get('num_channels')}\n",
                                language="text",
                            )
                        except Exception:
                            pass

                    # Class distributions
                    if use_backend:
                        pred_hist = (api_dbg or {}).get("pred_class_hist") or {}
                        st.write("Prediction class distribution:", pred_hist)
                        if include_ground_truth:
                            gt_hist = (api_dbg or {}).get("gt_class_hist") or {}
                            if gt_hist:
                                st.write("Ground truth class distribution:", gt_hist)
                    else:
                        pu, pc = np.unique(pred, return_counts=True)
                        pred_df = pd.DataFrame({"class_id": pu.astype(int), "pixels": pc.astype(int)})
                        pred_df["pct"] = (pred_df["pixels"] * 100.0 / pred_df["pixels"].sum()).round(2)
                        st.write("Prediction class distribution:", pred_df.sort_values("pixels", ascending=False).head(15))

                        if include_ground_truth and gt is not None:
                            gu, gc = np.unique(gt, return_counts=True)
                            gt_df = pd.DataFrame({"class_id": gu.astype(int), "pixels": gc.astype(int)})
                            gt_df["pct"] = (gt_df["pixels"] * 100.0 / gt_df["pixels"].sum()).round(2)
                            st.write("Ground truth class distribution:", gt_df.sort_values("pixels", ascending=False).head(15))


if __name__ == "__main__":
    app()
