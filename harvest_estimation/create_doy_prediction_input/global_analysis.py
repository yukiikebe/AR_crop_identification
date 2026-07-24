# global_analysis.py (flow-matching version)
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from harvest_estimation.create_doy_prediction_input.config import Config
from harvest_estimation.create_doy_prediction_input.harvest_detector import (
    HarvestDetector,
)
from harvest_estimation.create_doy_prediction_input.harvest_voter import HarvestVoter
from harvest_estimation.create_doy_prediction_input.log import PipelineLogger
from harvest_estimation.create_doy_prediction_input.seeding_estimator import (
    SeedingEstimator,
)
from harvest_estimation.create_doy_prediction_input.summary import (
    summarize_global_crop_outputs,
)
from harvest_estimation.create_doy_prediction_input.utils.metrics import (
    IoU_calculation,
    calculate_evi,
    calculate_ndvi,
    calculate_ndwi,
)
from harvest_estimation.create_doy_prediction_input.utils.plot_and_csv import (
    _plot_harvest_detection,
    _write_crop_summary_csv,
)
from harvest_estimation.create_doy_prediction_input.utils.raster_io import (
    ensure_dir,
    read_and_resample_band,
    save_resampled_plot_mask_once,
)


class GlobalAnalyzer:
    """
    Behavior-matching version of run_global_analysis() from esteban_harvest_estimation.py:
      - aligned plot mask saved once
      - loop timestamp folders
      - compute NDVI/NDWI/EVI maps
      - per-crop masking + per-timestamp saves (npy/png)
      - build per-crop time series
      - detect harvest start/end (per index)
      - vote start (with rule) + vote end (fallback)
      - IoU + graph + csv + excel summary
    """

    def __init__(self, cfg: Config, logger: PipelineLogger):
        self.cfg = cfg
        self.logger = logger
        self.detector = HarvestDetector(cfg, logger)
        self.voter = HarvestVoter(cfg, logger)
        self.seeding_estimator = SeedingEstimator.from_config(cfg, logger)

    def run(
        self,
        *,
        sample_path: Path,
        plot_mask_path: Path,
        output_root: Path,
        year: int,
        run_global: bool = True,
    ) -> None:
        if not run_global:
            return

        sample_path = Path(sample_path)
        plot_mask_path = Path(plot_mask_path)
        output_root = Path(output_root)
        ensure_dir(output_root)

        self.logger.info("🌍 Starting global crop-level analysis...")
        overwrite_outputs = bool(getattr(self.cfg, "overwrite_outputs", False))

        # ---- 1) ensure aligned mask exists (same as original) ----
        aligned_mask_path = sample_path / "plot_mask_resized.npy"
        save_resampled_plot_mask_once(
            plot_mask_path,
            sample_path,
            aligned_mask_path,
        )

        # ---- 2) accumulation containers (same as original) ----
        all_crops: set[int] = set()
        all_timestamps: list[datetime] = []
        data_by_crop_and_time: dict[
            int, dict[datetime, tuple[float, float, float]]
        ] = {}
        summary_rows: list[dict] = []

        # ---- 3) loop timestamp folders (same as original) ----
        for ts_folder in sorted(os.listdir(sample_path)):
            ts_path = sample_path / ts_folder
            if not ts_path.is_dir():
                continue

            self.logger.info(f"🗂️ Processing timestamp folder: {ts_folder}")

            try:
                files = os.listdir(ts_path)

                # NOTE: original looks for 'B4','B8','B2','B11' substrings
                red_file = next(f for f in files if "B4" in f)
                nir_file = next(f for f in files if "B8" in f)
                blue_file = next(f for f in files if "B2" in f)
                swir_file = next(f for f in files if ("B11" in f) or ("B12" in f))

                scl_file = next((f for f in files if "SCL" in f), None)

                with rasterio.open(ts_path / red_file) as red_src:
                    red = red_src.read(1).astype("float32") / 10000.0
                    nir = read_and_resample_band(ts_path / nir_file, red_src)
                    blue = read_and_resample_band(ts_path / blue_file, red_src)
                    swir = read_and_resample_band(ts_path / swir_file, red_src)

                    scl = None
                    if scl_file:
                        scl = read_and_resample_band(
                            ts_path / scl_file,
                            red_src,
                            scl=True,
                            resampling=Resampling.nearest,
                        ).astype(np.int16)

                    plot_mask_resized = np.load(aligned_mask_path)
                    plot_mask_resized[red == 0] = 0  # identical behavior

                    # crop labels present in this tile
                    crop_labels = [
                        v
                        for v in np.unique(plot_mask_resized)
                        if (v in self.cfg.valid_crop_labels) and (v != 0)
                    ]

                    # compute index maps
                    ndvi = calculate_ndvi(nir, red)
                    ndwi = calculate_ndwi(nir, swir)
                    evi = calculate_evi(nir, red, blue)

                    # SCL valid mask (same classes as original)
                    if scl is not None:
                        bad = np.array([3, 8, 9, 10, 11], dtype=np.int16)
                        scl_valid = ~np.isin(scl, bad)
                    else:
                        scl_valid = np.ones_like(red, dtype=bool)

                    timestamp = datetime.strptime(ts_folder, "%Y-%m-%d")
                    all_timestamps.append(timestamp)

                    # ---- 4) loop crops for this timestamp (same as original) ----
                    for crop_label_val in crop_labels:
                        crop_name = self.cfg.crop_dict.get(
                            int(crop_label_val),
                            f"Crop_{crop_label_val}",
                        )

                        if crop_name not in self.cfg.allowed_crops:
                            continue

                        crop_dir_name = self.cfg.crop_dir_name(crop_name)
                        crop_dir = output_root / crop_dir_name
                        ndvi_dir = crop_dir / "NDVI"
                        ndwi_dir = crop_dir / "NDWI"
                        evi_dir = crop_dir / "EVI"
                        for d in (crop_dir, ndvi_dir, ndwi_dir, evi_dir):
                            ensure_dir(d)
                        metadata_path = crop_dir / ".crop_name"
                        if not metadata_path.exists():
                            metadata_path.write_text(crop_name, encoding="utf-8")

                        mask_crop = plot_mask_resized == crop_label_val
                        mask = mask_crop & scl_valid

                        ndvi_val = float(np.nanmean(np.where(mask, ndvi, np.nan)))
                        ndwi_val = float(np.nanmean(np.where(mask, ndwi, np.nan)))
                        evi_val = float(np.nanmean(np.where(mask, evi, np.nan)))

                        all_crops.add(int(crop_label_val))
                        data_by_crop_and_time.setdefault(int(crop_label_val), {})[
                            timestamp
                        ] = (ndvi_val, ndwi_val, evi_val)

                        # Per-pixel images are not needed to build the tile-level
                        # workbooks consumed by the inference models.
                        if bool(getattr(self.cfg, "save_index_images", True)):
                            for index_data, index_val, index_dir, cmap in [
                                (ndvi, ndvi_val, ndvi_dir, "RdYlGn"),
                                (ndwi, ndwi_val, ndwi_dir, "Blues"),
                                (evi, evi_val, evi_dir, "Greens"),
                            ]:
                                masked = np.where(mask, index_data, np.nan)
                                npy_path = index_dir / f"{timestamp:%Y%m%d}.npy"
                                png_path = index_dir / f"{timestamp:%Y%m%d}.png"
                                if (not npy_path.exists()) or (not png_path.exists()):
                                    if not np.isnan(index_val):
                                        np.save(npy_path, masked)
                                        plt.imsave(
                                            png_path,
                                            np.nan_to_num(masked, nan=-1),
                                            cmap=cmap,
                                            vmin=-1,
                                            vmax=1,
                                        )
            except Exception as e:
                self.logger.warning(f"❌ Error in {ts_folder}: {e}")
                continue

        # ---- 5) per-crop series -> detect -> vote -> IoU -> graph/csv/excel (same as original) ----
        timestamps_array = sorted(set(all_timestamps))

        for label in sorted(all_crops):
            crop_name = self.cfg.crop_dict.get(label, f"Crop_{label}")
            if crop_name not in self.cfg.allowed_crops:
                continue

            crop_dir_name = self.cfg.crop_dir_name(crop_name)
            crop_dir = output_root / crop_dir_name
            csv_path = crop_dir / f"{crop_dir_name}_summary.csv"
            graph_path = crop_dir / f"{crop_dir_name}_graph.png"

            ndvi_series, ndwi_series, evi_series = [], [], []
            for ts in timestamps_array:
                values = data_by_crop_and_time.get(label, {}).get(
                    ts, (np.nan, np.nan, np.nan)
                )
                ndvi_series.append(values[0])
                ndwi_series.append(values[1])
                evi_series.append(values[2])

            ndvi_array = np.array(ndvi_series, dtype=float)
            ndwi_array = np.array(ndwi_series, dtype=float)
            evi_array = np.array(evi_series, dtype=float)

            if np.count_nonzero(~np.isnan(ndvi_array)) < self.cfg.min_points:
                continue

            # detection (same call pattern as original)
            detection_ndvi = self.detector.detect(
                timestamps_array,
                ndvi_array,
                label=f"crop:{crop_name}",
            )
            detection_ndwi = self.detector.detect(
                timestamps_array,
                ndwi_array,
                label=f"crop:{crop_name}",
            )
            detection_evi = self.detector.detect(
                timestamps_array,
                evi_array,
                label=f"crop:{crop_name}",
            )
            seeding_estimate = self.seeding_estimator.estimate(
                timestamps_array,
                ndvi_array,
                evi_array,
                crop_name=crop_name,
                ndvi_smoothed=detection_ndvi.smoothed,
                evi_smoothed=detection_evi.smoothed,
            )

            # start vote (use your voter if it matches, otherwise keep the old rule)
            pred_start, chosen_src, chosen_rule, div_start = self.voter.vote_with_rule(
                {
                    "NDVI": detection_ndvi.start,
                    "NDWI": detection_ndwi.start,
                    "EVI": detection_evi.start,
                },
                {
                    "NDVI": detection_ndvi.start_rule,
                    "NDWI": detection_ndwi.start_rule,
                    "EVI": detection_evi.start_rule,
                },
                kind="start",
                crop_name=crop_name,
                year=year,
                fallback_dates=timestamps_array,
            )
            pred_end, div_end = self.voter.vote_simple(
                [detection_ndvi.end, detection_ndwi.end, detection_evi.end],
                timestamps_array,
                kind="end",
                crop_name=crop_name,
                year=year,
            )

            if pred_start is None:
                continue
            if pred_end is None or pred_end < pred_start:
                end_candidates = [
                    d
                    for d in (detection_ndvi.end, detection_ndwi.end, detection_evi.end)
                    if d is not None and d >= pred_start
                ]
                if end_candidates:
                    pred_end = max(end_candidates)
                else:
                    later_ts = [t for t in timestamps_array if t >= pred_start]
                    pred_end = later_ts[-1] if later_ts else pred_start
            print("crop name:", crop_name)
            gt_window = self.cfg.gt_windows.get(crop_name)
            iou = IoU_calculation(pred_start, pred_end, gt_window) if gt_window else 0.0
            self.logger.info(
                f"🏆 {crop_name}: {pred_start.date()} to {pred_end.date()} "
                f"(by {chosen_src}, rule={chosen_rule}), IoU={iou:.3f}"
            )

            # graph
            if overwrite_outputs or not graph_path.exists():
                _plot_harvest_detection(
                    dates=timestamps_array,
                    ndvi=ndvi_array,
                    ndvi_smooth=detection_ndvi.smoothed,
                    ndwi=ndwi_array,
                    ndwi_smooth=detection_ndwi.smoothed,
                    evi=evi_array,
                    evi_smooth=detection_evi.smoothed,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    crop_name=crop_name,
                    graph_path=graph_path,
                    gt_windows=self.cfg.gt_windows,
                    green_mask=None,
                    logger=self.logger,
                )
            else:
                print(f"↪️  Skip plotting {graph_path} (exists)")

            # csv + excel rows (same content style)
            if overwrite_outputs or not csv_path.exists():
                _write_crop_summary_csv(
                    csv_path=csv_path,
                    crop_name=crop_name,
                    timestamps=timestamps_array,
                    ndvi=ndvi_array,
                    ndwi=ndwi_array,
                    evi=evi_array,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    chosen_rule=chosen_rule,
                    div_start=div_start,
                    div_end=div_end,
                    iou=iou,
                    summary_rows=summary_rows,
                    seeding_estimate=seeding_estimate,
                )

        summarize_global_crop_outputs(output_root)
