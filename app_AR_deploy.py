import base64
import io
import os
from datetime import datetime

import folium
from folium.plugins import Draw
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
from PIL import Image


MAPBOX_TOKEN = os.getenv("DEEPSAT_MAPBOX_TOKEN") or os.getenv("MAPBOX_TOKEN")

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


def _major_crops_palette_rgb() -> dict[int, tuple[int, int, int]]:
    # Must match DeepSatModels/ar_pred_api.py:_make_palette_major_crops()
    return {
        0: (0, 0, 0),         # Background
        1: (255, 215, 0),     # Corn
        2: (220, 20, 60),     # Cotton
        3: (30, 144, 255),    # Rice
        4: (34, 139, 34),     # Soybeans
        5: (255, 140, 0),     # Winter Wheat
    }


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _st_image_compat(*, image, caption: str | None = None) -> None:
    """
    Streamlit is deprecating use_container_width in favor of width='stretch' (2026+).
    Keep compatibility with older versions.
    """
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


def _st_dataframe_compat(df) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _get_roi(map_data):
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


def _roi_to_bbox(user_roig: list[list[float]]):
    lons = [p[0] for p in user_roig]
    lats = [p[1] for p in user_roig]
    return min(lons), min(lats), max(lons), max(lats)


def _get_map():
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
    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
    ).add_to(m)
    return st_folium(m, width=750, height=600)


def _api_predict_bbox(*, backend_url: str, year: int, month: int, bbox, timeout_s: int = 600):
    url = backend_url.rstrip("/") + "/predict"
    payload = {
        "year": int(year),
        "month": int(month),
        "bbox": {
            "lon_min": float(bbox[0]),
            "lat_min": float(bbox[1]),
            "lon_max": float(bbox[2]),
            "lat_max": float(bbox[3]),
        },
        "output": "png",
    }
    resp = requests.post(url, json=payload, timeout=int(timeout_s))
    if resp.status_code >= 400:
        msg = resp.text
        try:
            detail = (resp.json() or {}).get("detail")
            if isinstance(detail, str):
                msg = detail
        except Exception:
            pass

        # Common misconfiguration: pointing to ar_api.py (live inference) instead of ar_pred_api.py (precomputed).
        if "Provide either meta_patch" in msg:
            raise RuntimeError(
                "This backend looks like `ar_api.py` (expects meta_patch/lon_id/lat_id). "
                "For the deployment UI you must point to `ar_pred_api.py` (expects year/month/bbox)."
            )
        raise RuntimeError(f"{resp.status_code} {msg}")
    data = resp.json()
    pred_png = base64.b64decode(data["pred_png_base64"]) if data.get("pred_png_base64") else None
    return pred_png, data


def _check_backend(*, backend_url: str, timeout_s: int = 5) -> tuple[bool, str]:
    """
    Return (ok, message). We expect ar_pred_api.py to expose GET /info.
    """
    try:
        resp = requests.get(backend_url.rstrip("/") + "/info", timeout=int(timeout_s))
    except Exception as exc:
        return False, f"Cannot reach backend at {backend_url}: {exc}"
    if resp.status_code != 200:
        return False, f"Backend {backend_url} does not expose /info (status={resp.status_code})."
    try:
        data = resp.json() or {}
    except Exception:
        return False, f"Backend {backend_url} /info did not return JSON."
    if "supported_region" not in data or "models_supported_months" not in data:
        return False, f"Backend {backend_url} /info is not the expected schema for ar_pred_api.py."
    return True, "OK"


def app():
    st.title("Arkansas Crop-ID (deployment)")
    st.caption("Draw a rectangle inside the supported Arkansas region, then query precomputed predictions by year/month.")

    map_data = _get_map()
    user_roig = _get_roi(map_data)

    with st.sidebar:
        st.subheader("Query")
        backend_url_default = os.environ.get("DEEPSAT_AR_PRED_API_URL", "http://localhost:8001").strip()
        backend_url = st.text_input("Backend URL (env: DEEPSAT_AR_PRED_API_URL)", value=backend_url_default).strip()

        ok, msg = _check_backend(backend_url=backend_url)
        if not ok:
            st.error(msg)
            st.caption("Expected backend: `uvicorn ar_pred_api:app --port 8001` (not `ar_api:app`).")

        default_year = int(os.environ.get("DEEPSAT_AR_DEFAULT_YEAR", "2024"))
        default_month = int(os.environ.get("DEEPSAT_AR_DEFAULT_MONTH", str(datetime.utcnow().month)))
        year = st.number_input("Year", min_value=2000, max_value=2100, value=default_year, step=1)
        month = st.selectbox("Available month (1-12)", list(range(1, 13)), index=max(0, min(11, default_month - 1)))

        analyze = st.button("Analyze", type="primary")

    if analyze:
        if not user_roig:
            st.warning("Draw a rectangle on the map first.")
            return

        lon_min, lat_min, lon_max, lat_max = _roi_to_bbox(user_roig)
        bbox = (lon_min, lat_min, lon_max, lat_max)

        try:
            pred_png, dbg = _api_predict_bbox(backend_url=backend_url, year=int(year), month=int(month), bbox=bbox)
        except Exception as exc:
            st.error(f"Backend request failed: {exc}")
            return

        if pred_png is None:
            st.error("Backend did not return a PNG.")
            return

        im = Image.open(io.BytesIO(pred_png))
        _st_image_compat(
            image=im,
            caption=f"Prediction (year={int(year)}, month={int(month)}, model_month={dbg.get('model_month')})",
        )

        meta_patches = dbg.get("meta_patches") or []
        if meta_patches:
            st.write(f"Meta-patches used: {len(meta_patches)}")

        hist = (dbg.get("pred_class_hist") or {}) if isinstance(dbg, dict) else {}
        names = (dbg.get("class_names") or {}) if isinstance(dbg, dict) else {}
        if hist:
            palette = _major_crops_palette_rgb()
            df = pd.DataFrame([{"class_id": int(k), "name": names.get(str(k), ""), "pixels": int(v)} for k, v in hist.items()])
            df = df.sort_values("pixels", ascending=False).reset_index(drop=True)
            df["pct"] = (df["pixels"] * 100.0 / max(1, int(df["pixels"].sum()))).round(2)
            df["color"] = df["class_id"].map(lambda cid: _rgb_to_hex(palette.get(int(cid), (128, 128, 128))))

            st.markdown("#### Legend")
            legend_ids = (
                sorted({int(k) for k in names.keys()})
                if isinstance(names, dict) and names
                else df["class_id"].astype(int).tolist()
            )
            for cid in legend_ids:
                color = _rgb_to_hex(palette.get(int(cid), (128, 128, 128)))
                name = names.get(str(cid), "") if isinstance(names, dict) else ""
                label = f"{int(cid)}: {name}" if name else f"{int(cid)}"
                st.markdown(
                    (
                        "<div style='display:flex;align-items:center;gap:10px;margin:2px 0;'>"
                        f"<div style='width:14px;height:14px;border:1px solid #666;background:{color};'></div>"
                        f"<div style='font-size:14px;'>{label}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

            _st_dataframe_compat(df[["class_id", "name", "color", "pixels", "pct"]])


if __name__ == "__main__":
    app()
