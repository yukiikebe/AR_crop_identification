from __future__ import annotations

import base64
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import folium
from branca.element import Template
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
GRID_N = 20


def _ps_scene_root() -> Path:
    root = os.getenv("DEEPSAT_PS_SCENE_ROOT", "").strip() or "/home/yuki/PSScene"
    return Path(os.path.expanduser(os.path.expandvars(root)))


@st.cache_data(show_spinner=False)
def _load_ps_scene_features() -> list[dict]:
    root = _ps_scene_root()
    if not root.exists():
        return []

    features: list[dict] = []
    for fp in sorted(root.glob("*_metadata.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        geom = obj.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if str(geom.get("type")) != "Polygon" or not coords or not isinstance(coords[0], list):
            continue

        ring = []
        for pt in coords[0]:
            if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                ring = []
                break
            ring.append([float(pt[1]), float(pt[0])])
        if len(ring) < 4:
            continue

        props = obj.get("properties") or {}
        acquired = str(props.get("acquired") or "").strip()
        acquired_day = acquired[:10] if len(acquired) >= 10 else ""
        features.append(
            {
                "id": str(obj.get("id") or fp.stem),
                "acquired_day": acquired_day,
                "locations": ring,
            }
        )
    return features


def _ps_scene_bounds(features: list[dict]) -> tuple[float, float, float, float] | None:
    if not features:
        return None
    lons: list[float] = []
    lats: list[float] = []
    for feat in features:
        for lat, lon in feat.get("locations") or []:
            lats.append(float(lat))
            lons.append(float(lon))
    if not lons or not lats:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (
        float(a[2]) < float(b[0])
        or float(a[0]) > float(b[2])
        or float(a[3]) < float(b[1])
        or float(a[1]) > float(b[3])
    )


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
        drawing = map_data["all_drawings"][-1]
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


def _grid_edges(v_min: float, v_max: float, n: int) -> list[float]:
    step = (float(v_max) - float(v_min)) / float(n)
    return [float(v_min) + step * i for i in range(int(n) + 1)]


def _meta_patch_bounds(i: int, j: int) -> tuple[float, float, float, float]:
    lon_edges = _grid_edges(AR_LON_MIN, AR_LON_MAX, GRID_N)
    lat_edges = _grid_edges(AR_LAT_MIN, AR_LAT_MAX, GRID_N)
    return lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1]


def _add_tile_grid_debug_overlay(m) -> None:
    grid_layer = folium.FeatureGroup(name="Tile Grid Debug", overlay=True, control=True)
    for i in range(GRID_N):
        for j in range(GRID_N):
            lon_min, lat_min, lon_max, lat_max = _meta_patch_bounds(i, j)
            folium.Rectangle(
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                color="#111111",
                weight=1,
                opacity=0.55,
                fill=False,
                tooltip=(
                    f"{i}_{j} | "
                    f"lon=[{lon_min:.4f}, {lon_max:.4f}] "
                    f"lat=[{lat_min:.4f}, {lat_max:.4f}]"
                ),
            ).add_to(grid_layer)
    grid_layer.add_to(m)


def _add_ps_scene_overlay(m) -> None:
    features = _load_ps_scene_features()
    if not features:
        return

    layer = folium.FeatureGroup(name="Planet PSScene Coverage", overlay=True, control=True)
    for feat in features:
        scene_id = str(feat.get("id") or "")
        acquired_day = str(feat.get("acquired_day") or "")
        tooltip = scene_id if not acquired_day else f"{scene_id} ({acquired_day})"
        folium.Polygon(
            locations=feat["locations"],
            color="#0057d9",
            weight=4,
            opacity=1.0,
            fill=True,
            fill_color="#6fb6ff",
            fill_opacity=0.18,
            tooltip=tooltip,
        ).add_to(layer)
    bounds = _ps_scene_bounds(features)
    if bounds is not None:
        lon_min, lat_min, lon_max, lat_max = bounds
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="#0047b3",
            weight=3,
            opacity=0.95,
            dash_array="8, 8",
            fill=False,
            tooltip="Planet coverage extent",
        ).add_to(layer)
    layer.add_to(m)


class SingleRectangleDraw(Draw):
    # Same Draw control behavior, but keep only the latest rectangle in the editable layer.
    _template = Template(
        """
        {% macro html(this, kwargs) %}
            {% if this.export %}
            <style>
                #export {
                    position: absolute;
                    top: 5px;
                    right: 10px;
                    z-index: 999;
                    background: white;
                    color: black;
                    padding: 6px;
                    border-radius: 4px;
                    font-family: 'Helvetica Neue';
                    cursor: pointer;
                    font-size: 12px;
                    text-decoration: none;
                    top: 90px;
                }
            </style>
            <a href='#' id='export'>Export</a>
            {% endif %}
        {% endmacro %}

        {% macro script(this, kwargs) %}
            var options = {
              position: {{ this.position|tojson }},
              draw: {{ this.draw_options|tojson }},
              edit: {{ this.edit_options|tojson }},
            }
            {%- if this.feature_group  %}
                var drawnItems_{{ this.get_name() }} =
                    {{ this.feature_group.get_name() }};
            {%- else %}
                var drawnItems_{{ this.get_name() }} =
                    new L.featureGroup().addTo(
                        {{ this._parent.get_name() }}
                    );
            {%- endif %}

            options.edit.featureGroup = drawnItems_{{ this.get_name() }};
            var {{ this.get_name() }} = new L.Control.Draw(
                options
            ).addTo({{ this._parent.get_name() }});

            {{ this._parent.get_name() }}.on(L.Draw.Event.CREATED, function(e) {
                var layer = e.layer;
                var coords = JSON.stringify(layer.toGeoJSON());
                drawnItems_{{ this.get_name() }}.clearLayers();
                {%- if this.show_geometry_on_click %}
                layer.on('click', function() {
                    alert(coords);
                    console.log(coords);
                });
                {%- endif %}

                {%- for event, handler in this.on.items() %}
                layer.on(
                    "{{event}}",
                    {{handler}}
                );
                {%- endfor %}
                drawnItems_{{ this.get_name() }}.addLayer(layer);
            });

            {% if this.export %}
            document.getElementById('export').onclick = function(e) {
                var data = drawnItems_{{ this.get_name() }}.toGeoJSON();
                var convertedData = 'text/json;charset=utf-8,'
                    + encodeURIComponent(JSON.stringify(data));
                document.getElementById('export').setAttribute(
                    'href', 'data:' + convertedData
                );
                document.getElementById('export').setAttribute(
                    'download', {{ this.filename|tojson }}
                );
            }
            {% endif %}
        {% endmacro %}
        """
    )


def _get_map(*, show_tile_grid_debug: bool = False):
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
    _add_ps_scene_overlay(m)
    if bool(show_tile_grid_debug):
        _add_tile_grid_debug_overlay(m)
    if bool(show_tile_grid_debug) or bool(_load_ps_scene_features()):
        folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(AR_BOUNDS)
    SingleRectangleDraw(
        export=True,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": {"repeatMode": False},
        },
    ).add_to(m)
    return st_folium(m, width=750, height=600)


def _api_predict_crop_bbox(*, backend_url: str, year: int, month: int, bbox, timeout_s: int = 600):
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


def _api_predict_fastdiffsr(
    *,
    backend_url: str,
    year: int,
    month: int,
    bbox,
    timeout_s: int = 3600,
):
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
        raise RuntimeError(f"{resp.status_code} {msg}")
    return int(resp.status_code), resp.json()


def _api_fastdiffsr_job_status(
    *,
    backend_url: str,
    status_url: str | None = None,
    job_id: str | None = None,
    timeout_s: int = 30,
):
    if status_url:
        if str(status_url).startswith("http://") or str(status_url).startswith("https://"):
            url = str(status_url)
        else:
            url = backend_url.rstrip("/") + "/" + str(status_url).lstrip("/")
    elif job_id:
        url = backend_url.rstrip("/") + f"/jobs/{job_id}"
    else:
        raise RuntimeError("Need status_url or job_id for SR job status.")

    resp = requests.get(url, timeout=int(timeout_s))
    if resp.status_code >= 400:
        msg = resp.text
        try:
            detail = (resp.json() or {}).get("detail")
            if isinstance(detail, str):
                msg = detail
        except Exception:
            pass
        raise RuntimeError(f"{resp.status_code} {msg}")
    return resp.json()


def _render_fastdiffsr_result(*, resp: dict, year: int, month: int) -> None:
    def _decode_preview(b64_value: object):
        if not (isinstance(b64_value, str) and b64_value):
            return None, None
        try:
            png = base64.b64decode(b64_value)
            return Image.open(io.BytesIO(png)), None
        except Exception as exc:
            return None, str(exc)

    raw_im, raw_err = _decode_preview(resp.get("input_preview_png_base64"))
    sr_im, sr_err = _decode_preview(resp.get("preview_png_base64"))
    planet_im, planet_err = _decode_preview(resp.get("planet_preview_png_base64"))

    if raw_im is not None or sr_im is not None or planet_im is not None:
        cols = st.columns(3)
        with cols[0]:
            st.caption("Input (raw Sentinel-2 RGB)")
            if raw_im is not None:
                raw_w = int(resp.get("input_preview_width") or 0)
                raw_h = int(resp.get("input_preview_height") or 0)
                raw_native_w = int(resp.get("input_preview_native_width") or raw_w)
                raw_native_h = int(resp.get("input_preview_native_height") or raw_h)
                raw_downsampled = bool(resp.get("input_preview_was_downsampled", False))
                _st_image_compat(
                    image=raw_im,
                    caption=(
                        f"Raw input preview (year={int(year)}, month={int(month)}, "
                        f"size={raw_w}x{raw_h})"
                    ),
                )
                if raw_downsampled:
                    st.caption(f"Raw preview downsampled for web display from {raw_native_w}x{raw_native_h}.")
            elif raw_err:
                st.warning(f"Failed to decode raw input preview image: {raw_err}")
            else:
                st.info("Raw input preview is not available for this response.")
        with cols[1]:
            st.caption("Output (FastDiffSR)")
            if sr_im is not None:
                sr_w = int(resp.get("preview_width") or 0)
                sr_h = int(resp.get("preview_height") or 0)
                sr_native_w = int(resp.get("preview_native_width") or sr_w)
                sr_native_h = int(resp.get("preview_native_height") or sr_h)
                sr_downsampled = bool(resp.get("preview_was_downsampled", False))
                _st_image_compat(
                    image=sr_im,
                    caption=(
                        f"FastDiffSR preview (year={int(year)}, month={int(month)}, "
                        f"size={sr_w}x{sr_h})"
                    ),
                )
                if sr_downsampled:
                    st.info(f"SR preview was downsampled for web display from {sr_native_w}x{sr_native_h}.")
            elif sr_err:
                st.warning(f"Failed to decode SR preview image: {sr_err}")
            else:
                st.info("SR preview image is not available for this response (for example, date_policy=all).")
        with cols[2]:
            st.caption("Reference (Planet PSScene, 2024-08-13)")
            if planet_im is not None:
                planet_w = int(resp.get("planet_preview_width") or 0)
                planet_h = int(resp.get("planet_preview_height") or 0)
                planet_native_w = int(resp.get("planet_preview_native_width") or planet_w)
                planet_native_h = int(resp.get("planet_preview_native_height") or planet_h)
                planet_downsampled = bool(resp.get("planet_preview_was_downsampled", False))
                _st_image_compat(
                    image=planet_im,
                    caption=f"Planet reference preview (size={planet_w}x{planet_h})",
                )
                if planet_downsampled:
                    st.caption(
                        f"Planet preview downsampled for web display from {planet_native_w}x{planet_native_h}."
                    )
                planet_debug = resp.get("planet_preview_debug")
                if isinstance(planet_debug, dict) and bool(planet_debug.get("partial_coverage")):
                    st.caption("Only the PSScene-intersecting part of the selected bbox has Planet coverage.")
            elif planet_err:
                st.warning(f"Failed to decode Planet preview image: {planet_err}")
            else:
                st.info("Planet reference is not available for this bbox.")
    else:
        if sr_err:
            st.warning(f"Failed to decode SR preview image: {sr_err}")
        else:
            st.info("SR preview image is not available for this response (for example, date_policy=all).")

    st.write(f"Output dir: `{resp.get('output_dir', '')}`")
    st.write(f"Meta-patches requested from bbox: {int(resp.get('total_meta_patches', 0))}")
    st.write(f"Merged preview tiles used: {int(resp.get('preview_tile_count', 0))}")
    if resp.get("input_preview_tile_count") is not None:
        st.write(f"Merged raw-input preview tiles used: {int(resp.get('input_preview_tile_count', 0))}")
    if resp.get("planet_preview_tile_count") is not None:
        st.write(f"Merged Planet preview tiles used: {int(resp.get('planet_preview_tile_count', 0))}")
    preview_debug = resp.get("preview_debug")
    if isinstance(preview_debug, dict) and preview_debug:
        st.write("Preview debug:")
        try:
            st.json(preview_debug)
        except Exception:
            st.write(preview_debug)
    input_preview_debug = resp.get("input_preview_debug")
    if isinstance(input_preview_debug, dict) and input_preview_debug:
        st.write("Raw input preview debug:")
        try:
            st.json(input_preview_debug)
        except Exception:
            st.write(input_preview_debug)
    planet_preview_debug = resp.get("planet_preview_debug")
    if isinstance(planet_preview_debug, dict) and planet_preview_debug:
        st.write("Planet preview debug:")
        try:
            st.json(planet_preview_debug)
        except Exception:
            st.write(planet_preview_debug)

    outputs = resp.get("outputs") or []
    if outputs:
        df = pd.DataFrame(outputs)
        if "path" in df.columns:
            df["path"] = df["path"].fillna("")
        _st_dataframe_compat(df)


def _check_crop_backend(*, backend_url: str, timeout_s: int = 5) -> tuple[bool, str]:
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


def _check_fastdiffsr_backend(*, backend_url: str, timeout_s: int = 5) -> tuple[bool, str]:
    """
    Return (ok, message). We expect ar_fastdiffsr_api.py to expose GET /info.
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
    if "request_schema" not in data or "ready" not in data:
        return False, f"Backend {backend_url} /info is not the expected schema for ar_fastdiffsr_api.py."
    if not bool(data.get("ready", False)):
        missing = data.get("missing_required_env") or []
        return False, f"SR backend is not ready. Missing required env(s): {missing}"
    return True, "OK"


def _init_app_state() -> None:
    st.session_state.setdefault("ar_sr_active_request", None)
    st.session_state.setdefault("ar_sr_job", None)
    st.session_state.setdefault("ar_sr_result", None)
    st.session_state.setdefault("ar_sr_error", None)


def _clear_sr_state() -> None:
    st.session_state["ar_sr_active_request"] = None
    st.session_state["ar_sr_job"] = None
    st.session_state["ar_sr_result"] = None
    st.session_state["ar_sr_error"] = None


def app():
    _init_app_state()
    st.title("Arkansas Crop-ID (deployment)")
    st.caption("Draw a rectangle inside Arkansas, then choose Crop Identification or Super Resolution.")

    with st.sidebar:
        st.subheader("Query")
        task = st.selectbox("Task", ["Crop Identification", "Super Resolution"], index=0)
        show_tile_grid_debug = st.checkbox("Show Tile Grid Debug", value=False)

        if task == "Crop Identification":
            backend_url_default = os.environ.get("DEEPSAT_AR_PRED_API_URL", "http://localhost:8001").strip()
            backend_url = st.text_input("Crop Backend URL (env: DEEPSAT_AR_PRED_API_URL)", value=backend_url_default).strip()

            ok, msg = _check_crop_backend(backend_url=backend_url)
            if not ok:
                st.error(msg)
                st.caption("Expected: `uvicorn ar_pred_api:app --port 8001`.")
        else:
            backend_url_default = os.environ.get("DEEPSAT_FASTDIFFSR_API_URL", "http://localhost:8002").strip()
            backend_url = st.text_input("SR Backend URL (env: DEEPSAT_FASTDIFFSR_API_URL)", value=backend_url_default).strip()
            ok, msg = _check_fastdiffsr_backend(backend_url=backend_url)
            if not ok:
                st.error(msg)
                st.caption("Expected: `uvicorn ar_fastdiffsr_api:app --port 8002`.")

        default_year = int(os.environ.get("DEEPSAT_AR_DEFAULT_YEAR", "2024"))
        default_month = int(os.environ.get("DEEPSAT_AR_DEFAULT_MONTH", str(datetime.utcnow().month)))
        year = st.number_input("Year", min_value=2000, max_value=2100, value=default_year, step=1)
        month = st.selectbox("Available month (1-12)", list(range(1, 13)), index=max(0, min(11, default_month - 1)))

        analyze = st.button("Analyze", type="primary")

    map_data = _get_map(show_tile_grid_debug=bool(show_tile_grid_debug))
    user_roig = _get_roi(map_data)
    ps_scene_features = _load_ps_scene_features()
    ps_scene_bounds = _ps_scene_bounds(ps_scene_features)
    if bool(show_tile_grid_debug):
        st.caption("Tile grid debug enabled. Hover a black tile boundary to see its meta-patch id and lon/lat range.")
    if ps_scene_bounds is not None:
        st.caption(
            "Planet coverage loaded: "
            f"{len(ps_scene_features)} scenes | "
            f"lon=[{ps_scene_bounds[0]:.4f}, {ps_scene_bounds[2]:.4f}] "
            f"lat=[{ps_scene_bounds[1]:.4f}, {ps_scene_bounds[3]:.4f}]"
        )
    else:
        st.caption("Planet coverage unavailable: no PSScene metadata loaded.")

    if analyze:
        if not user_roig:
            st.warning("Draw a rectangle on the map first.")
            return

        lon_min, lat_min, lon_max, lat_max = _roi_to_bbox(user_roig)
        bbox = (lon_min, lat_min, lon_max, lat_max)

        if task == "Crop Identification":
            try:
                pred_png, dbg = _api_predict_crop_bbox(backend_url=backend_url, year=int(year), month=int(month), bbox=bbox)
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
        else:
            if ps_scene_bounds is not None and not _bbox_intersects(bbox, ps_scene_bounds):
                st.warning("Selected bbox is outside Planet PSScene coverage. SR will still run, but Planet reference will be unavailable.")
            st.session_state["ar_sr_active_request"] = {
                "backend_url": str(backend_url),
                "year": int(year),
                "month": int(month),
                "bbox": [float(x) for x in bbox],
            }
            st.session_state["ar_sr_job"] = None
            st.session_state["ar_sr_result"] = None
            st.session_state["ar_sr_error"] = None

    if task == "Super Resolution":
        active_req = st.session_state.get("ar_sr_active_request")
        if not isinstance(active_req, dict):
            return

        req_backend_url = str(active_req.get("backend_url", "")).strip()
        req_year = int(active_req.get("year"))
        req_month = int(active_req.get("month"))
        req_bbox_raw = active_req.get("bbox") or []
        if not (isinstance(req_bbox_raw, list) and len(req_bbox_raw) == 4):
            st.warning("Stored SR request is invalid. Clearing state.")
            _clear_sr_state()
            return
        req_bbox = tuple(float(x) for x in req_bbox_raw)

        st.caption(
            f"Active SR request: year={req_year}, month={req_month}, "
            f"bbox=({req_bbox[0]:.5f}, {req_bbox[1]:.5f}, {req_bbox[2]:.5f}, {req_bbox[3]:.5f})"
        )
        st.caption("Map zoom/pan reruns no longer clear the active SR request. The app resumes from session state.")

        cols = st.columns(2)
        with cols[0]:
            if st.button("Refresh SR Status"):
                st.rerun()
        with cols[1]:
            if st.button("Clear SR State"):
                _clear_sr_state()
                st.rerun()

        sr_error = st.session_state.get("ar_sr_error")
        if isinstance(sr_error, str) and sr_error:
            st.warning(sr_error)

        sr_result = st.session_state.get("ar_sr_result")
        if isinstance(sr_result, dict):
            st.success(
                f"SR complete: wrote={int(sr_result.get('wrote', 0))}, "
                f"skipped_existing={int(sr_result.get('skipped_existing', 0))}"
            )
            _render_fastdiffsr_result(resp=sr_result, year=req_year, month=req_month)
            return

        sr_job = st.session_state.get("ar_sr_job")
        if isinstance(sr_job, dict):
            job_id = str(sr_job.get("job_id", "") or "")
            status_url = str(sr_job.get("status_url", "") or "")
            try:
                job = _api_fastdiffsr_job_status(
                    backend_url=req_backend_url,
                    status_url=status_url or None,
                    job_id=job_id or None,
                )
            except Exception as exc:
                st.session_state["ar_sr_error"] = f"Polling failed: {exc}"
                st.warning(st.session_state["ar_sr_error"])
                return

            progress = job.get("progress") or {}
            done = int(progress.get("done", 0))
            total = int(progress.get("total", 0))
            written = int(progress.get("written", 0))
            skipped = int(progress.get("skipped", 0))
            failed = int(progress.get("failed", 0))
            state = str(job.get("state", "unknown"))
            phase = str(job.get("phase", state))
            phase_msg = str(job.get("phase_message", "") or "")

            st.info(
                f"SR job `{job_id}` state={state} phase={phase} | done={done}/{total}, "
                f"written={written}, skipped={skipped}, failed={failed}"
                + (f" | {phase_msg}" if phase_msg else "")
            )

            if state in {"queued", "running"}:
                return

            if state == "completed":
                try:
                    status_code2, resp2 = _api_predict_fastdiffsr(
                        backend_url=req_backend_url,
                        year=req_year,
                        month=req_month,
                        bbox=req_bbox,
                    )
                except Exception as exc:
                    st.session_state["ar_sr_error"] = f"SR completed but result refresh failed: {exc}"
                    st.warning(st.session_state["ar_sr_error"])
                    return

                if int(status_code2) == 200:
                    st.session_state["ar_sr_result"] = resp2
                    st.session_state["ar_sr_job"] = None
                    st.session_state["ar_sr_error"] = None
                    st.success(
                        f"SR completed: written={int(progress.get('written', 0))}, "
                        f"skipped={int(progress.get('skipped', 0))}, failed={int(progress.get('failed', 0))}"
                    )
                    _render_fastdiffsr_result(resp=resp2, year=req_year, month=req_month)
                    return

                st.warning("SR job completed, but backend still returned a queued/running response. Click Refresh SR Status.")
                return

            st.error(f"SR job ended with state: {state}")
            errors = job.get("errors") or []
            if errors:
                st.write("Job errors:")
                for e in errors:
                    st.write(f"- {e}")
            outputs = job.get("outputs") or []
            if outputs:
                df = pd.DataFrame(outputs)
                if "path" in df.columns:
                    df["path"] = df["path"].fillna("")
                _st_dataframe_compat(df)
            return

        try:
            status_code, resp = _api_predict_fastdiffsr(
                backend_url=req_backend_url,
                year=req_year,
                month=req_month,
                bbox=req_bbox,
            )
        except Exception as exc:
            st.session_state["ar_sr_error"] = f"SR backend request failed: {exc}"
            st.error(st.session_state["ar_sr_error"])
            return

        st.session_state["ar_sr_error"] = None
        if int(status_code) == 202:
            st.session_state["ar_sr_job"] = {
                "job_id": str(resp.get("job_id", "") or ""),
                "status_url": str(resp.get("status_url", "") or ""),
            }
            st.info(resp.get("message", "SR backend started a generation job."))
            st.write(f"Job ID: `{resp.get('job_id', '')}`")
            if resp.get("status_url"):
                st.write(f"Status URL: `{resp.get('status_url', '')}`")
            st.write(f"Output dir: `{resp.get('output_dir', '')}`")
            st.write(f"Requested bbox meta-patches: {int(resp.get('requested_meta_patches', 0))}")
            st.write(f"Month job total meta-patches: {int(resp.get('total_meta_patches', 0))}")
            st.write(f"Month job remaining meta-patches: {int(resp.get('missing_meta_patches', 0))}")
            return

        st.session_state["ar_sr_result"] = resp
        st.session_state["ar_sr_job"] = None
        st.success(
            f"SR complete: wrote={int(resp.get('wrote', 0))}, "
            f"skipped_existing={int(resp.get('skipped_existing', 0))}"
        )
        _render_fastdiffsr_result(resp=resp, year=req_year, month=req_month)
        return


if __name__ == "__main__":
    app()
