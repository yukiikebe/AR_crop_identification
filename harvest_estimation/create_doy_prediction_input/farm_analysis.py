# farm_level_analyzer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

from harvest_estimation.create_doy_prediction_input.config import Config
from harvest_estimation.create_doy_prediction_input.harvest_detector import (
    HarvestDetector,
)
from harvest_estimation.create_doy_prediction_input.harvest_voter import HarvestVoter
from harvest_estimation.create_doy_prediction_input.log import PipelineLogger
from harvest_estimation.create_doy_prediction_input.seeding_estimator import (
    SeedingEstimator,
)
from harvest_estimation.create_doy_prediction_input.utils.metrics import (
    IoU_calculation,
)  # keep your existing one (or replace with temporal_iou)
from harvest_estimation.create_doy_prediction_input.utils.plot_and_csv import (
    _plot_harvest_detection,
    _write_crop_summary_csv,
)
from harvest_estimation.create_doy_prediction_input.utils.raster_io import (
    get_farms_in_crop_mask,
    load_saved_index_images,
    reconstruct_mask,
    save_resampled_plot_mask_once,
)


@dataclass
class FarmResult:
    crop_name: str
    farm_id: str
    pred_start: datetime
    pred_end: datetime
    chosen_source: str
    chosen_rule: str
    div_start: Optional[int]
    div_end: Optional[int]
    iou: float


class FarmLevelAnalyzer:
    """
    Flow-matching refactor of run_farm_level_analysis().

    Assumes global stage already created per-crop folders:
      output_root/<crop_name>/NDVI/YYYYMMDD.npy
      output_root/<crop_name>/NDWI/YYYYMMDD.npy
      output_root/<crop_name>/EVI/YYYYMMDD.npy
    """

    def __init__(self, cfg: Config, logger: PipelineLogger):
        self.cfg = cfg
        self.logger = logger
        self.detector = HarvestDetector(cfg, logger=logger)
        self.voter = HarvestVoter(cfg, logger=logger)
        self.seeding_estimator = SeedingEstimator.from_config(cfg, logger)

    def run(
        self,
        *,
        sample_path: Path,
        plot_mask_path: Path,
        output_root: Path,
        year: int,
        min_pixels: int = 15,
    ) -> List[FarmResult]:
        self.logger.info("🚜 Starting farm-level analysis...")
        overwrite_outputs = bool(getattr(self.cfg, "overwrite_outputs", False))
        sample_path = Path(sample_path)
        plot_mask_path = Path(plot_mask_path)
        output_root = Path(output_root)

        # Ensure aligned plot mask exists
        aligned_mask_path = sample_path / "plot_mask_resized.npy"
        if not aligned_mask_path.exists():
            self.logger.info("ℹ️ Generating aligned mask...")
            save_resampled_plot_mask_once(
                str(plot_mask_path), str(sample_path), str(aligned_mask_path)
            )
        mask_data = np.load(aligned_mask_path)

        # Get spatial transform from first timestamp B4
        ts_folder = next(
            (f for f in sorted(os.listdir(sample_path)) if (sample_path / f).is_dir()),
            None,
        )
        if ts_folder is None:
            raise RuntimeError(f"No timestamp folder found under {sample_path}")

        ts_path = sample_path / ts_folder
        red_file = next((f for f in os.listdir(ts_path) if "B4" in f), None)
        if red_file is None:
            raise RuntimeError(f"No B4 file found under {ts_path}")

        with rasterio.open(ts_path / red_file) as src:
            mask_transform = src.transform

        crop_dirs = [
            d for d in os.listdir(output_root) if (output_root / d).is_dir()
        ]  # pathlib.Path
        results: List[FarmResult] = []

        for crop_dir_name in crop_dirs:
            crop_name = self.cfg.crop_name_from_dir(crop_dir_name)
            if crop_name is None:
                self.logger.warning(
                    f"⚠️ No crop name for crop_dir={crop_dir_name}; skipping"
                )
                continue

            self.logger.info(f"📦 Processing crop: {crop_name}")
            crop_dir = output_root / crop_dir_name
            ndvi_dir = crop_dir / "NDVI"
            ndwi_dir = crop_dir / "NDWI"
            evi_dir = crop_dir / "EVI"

            if not ndvi_dir.exists():
                continue

            # base timestamps from NDVI pngs
            base_timestamp: List[datetime] = []
            png_files = sorted(
                [
                    p
                    for p in ndvi_dir.glob("*.png")
                    if p.stem.isdigit() and len(p.stem) == 8
                ]
            )
            base_timestamp = [datetime.strptime(p.stem, "%Y%m%d") for p in png_files]

            if not base_timestamp:
                continue

            # crop label from cfg.crop_dict (same as original)
            crop_label = next(
                (k for k, v in self.cfg.crop_dict.items() if v == crop_name), None
            )
            if crop_label is None:
                self.logger.warning(
                    f"⚠️ No crop_label for crop_name={crop_name}; skipping"
                )
                continue

            farms = get_farms_in_crop_mask(
                mask_data, int(crop_label), mask_transform, min_pixels=min_pixels
            )
            self.logger.info(f"🌾 Found {len(farms)} farm plots for {crop_name}")

            # Load stacks once per crop (same as original: load_saved_index_images)
            ndvi_stack = load_saved_index_images(ndvi_dir, list(base_timestamp))
            ndwi_stack = load_saved_index_images(ndwi_dir, list(base_timestamp))
            evi_stack = load_saved_index_images(evi_dir, list(base_timestamp))

            for farm_id, coords in farms.items():
                farms_dir = crop_dir / "Farms" / farm_id
                farms_dir.mkdir(parents=True, exist_ok=True)

                csv_path = farms_dir / f"{crop_dir_name}_{farm_id}_summary.csv"
                graph_path = farms_dir / f"{crop_dir_name}_{farm_id}_combined.png"

                if (
                    (not overwrite_outputs)
                    and csv_path.exists()
                    and graph_path.exists()
                ):
                    self.logger.info(
                        f"⏭️ Skipping farm {farm_id} (CSV & plot already exist)"
                    )
                    continue

                farm_mask = reconstruct_mask(coords, mask_data.shape)
                green_mask = np.zeros((*farm_mask.shape, 3), dtype=np.uint8)
                green_mask[..., 1] = farm_mask.astype(np.uint8) * 255

                ndvi_series, ndwi_series, evi_series = [], [], []
                for ts, ndvi, ndwi, evi in zip(
                    base_timestamp, ndvi_stack, ndwi_stack, evi_stack
                ):
                    if ndvi is None or ndwi is None or evi is None:
                        ndvi_series.append(np.nan)
                        ndwi_series.append(np.nan)
                        evi_series.append(np.nan)
                        continue

                    masked_ndvi = np.where(farm_mask, ndvi, np.nan)
                    masked_ndwi = np.where(farm_mask, ndwi, np.nan)
                    masked_evi = np.where(farm_mask, evi, np.nan)

                    v_ndvi = np.isfinite(masked_ndvi)
                    v_ndwi = np.isfinite(masked_ndwi)
                    v_evi = np.isfinite(masked_evi)

                    if not (np.any(v_ndvi) or np.any(v_ndwi) or np.any(v_evi)):
                        ndvi_series.append(np.nan)
                        ndwi_series.append(np.nan)
                        evi_series.append(np.nan)
                        continue

                    ndvi_series.append(
                        masked_ndvi[v_ndvi].mean() if np.any(v_ndvi) else np.nan
                    )
                    ndwi_series.append(
                        masked_ndwi[v_ndwi].mean() if np.any(v_ndwi) else np.nan
                    )
                    evi_series.append(
                        masked_evi[v_evi].mean() if np.any(v_evi) else np.nan
                    )

                ndvi_array = np.array(ndvi_series, dtype=float)
                ndwi_array = np.array(ndwi_series, dtype=float)
                evi_array = np.array(evi_series, dtype=float)

                if np.count_nonzero(~np.isnan(ndvi_array)) < int(self.cfg.min_points):
                    self.logger.info(
                        f"⏭️ Skipping farm {farm_id} due to insufficient data"
                    )
                    continue

                # ---- Detection using your HarvestDetector ----
                det_ndvi = self.detector.detect(
                    base_timestamp, ndvi_array, label=f"farm:{farm_id}/NDVI"
                )
                det_ndwi = self.detector.detect(
                    base_timestamp, ndwi_array, label=f"farm:{farm_id}/NDWI"
                )
                det_evi = self.detector.detect(
                    base_timestamp, evi_array, label=f"farm:{farm_id}/EVI"
                )
                seeding_estimate = self.seeding_estimator.estimate(
                    base_timestamp,
                    ndvi_array,
                    evi_array,
                    crop_name=crop_name,
                    ndvi_smoothed=det_ndvi.smoothed,
                    evi_smoothed=det_evi.smoothed,
                )

                # ---- Voting using your HarvestVoter ----
                pred_start, chosen_source, chosen_rule, div_start = (
                    self.voter.vote_with_rule(
                        {
                            "NDVI": det_ndvi.start,
                            "NDWI": det_ndwi.start,
                            "EVI": det_evi.start,
                        },
                        {
                            "NDVI": det_ndvi.start_rule,
                            "NDWI": det_ndwi.start_rule,
                            "EVI": det_evi.start_rule,
                        },
                        kind="start",
                        crop_name=crop_name,
                        year=year,
                        fallback_dates=base_timestamp,
                    )
                )
                pred_end, div_end = self.voter.vote_simple(
                    [det_ndvi.end, det_ndwi.end, det_evi.end],
                    base_timestamp,
                    kind="end",
                    crop_name=crop_name,
                    year=year,
                )

                if pred_end is None or pred_end < pred_start:
                    end_candidates = [
                        d
                        for d in (det_ndvi.end, det_ndwi.end, det_evi.end)
                        if d is not None and d >= pred_start
                    ]
                    if end_candidates:
                        pred_end = max(end_candidates)
                    else:
                        later_ts = [t for t in base_timestamp if t >= pred_start]
                        pred_end = later_ts[-1] if later_ts else pred_start

                assert pred_start <= pred_end, (
                    f"Invalid harvest period for farm {farm_id}"
                )

                # IoU
                gt_window = self.cfg.gt_windows.get(crop_name)
                iou = (
                    IoU_calculation(pred_start, pred_end, gt_window)
                    if gt_window
                    else 0.0
                )

                # Plot
                _plot_harvest_detection(
                    dates=base_timestamp,
                    ndvi=ndvi_array,
                    ndvi_smooth=det_ndvi.smoothed,
                    ndwi=ndwi_array,
                    ndwi_smooth=det_ndwi.smoothed,
                    evi=evi_array,
                    evi_smooth=det_evi.smoothed,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    crop_name=crop_name,
                    graph_path=graph_path,
                    gt_windows=self.cfg.gt_windows,  # same dict as global
                    green_mask=green_mask,
                    farm_id=farm_id,
                    farm_mask=farm_mask,
                    logger=self.logger,
                )

                # CSV
                _write_crop_summary_csv(
                    csv_path=csv_path,
                    crop_name=crop_name,
                    timestamps=base_timestamp,
                    ndvi=ndvi_array,
                    ndwi=ndwi_array,
                    evi=evi_array,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    chosen_rule=chosen_rule,
                    div_start=div_start,
                    div_end=div_end,
                    iou=iou,
                    summary_rows=None,
                    farm_id=farm_id,
                    seeding_estimate=seeding_estimate,
                )

                results.append(
                    FarmResult(
                        crop_name=crop_name,
                        farm_id=farm_id,
                        pred_start=pred_start,
                        pred_end=pred_end,
                        chosen_source=chosen_source,
                        chosen_rule=chosen_rule,
                        div_start=div_start,
                        div_end=div_end,
                        iou=iou,
                    )
                )

                self.logger.info(
                    f"✅ Saved output for farm {farm_id}. Start: {pred_start}, End: {pred_end}"
                )

        self.logger.info("🏁 All farms processed.")
        return results
