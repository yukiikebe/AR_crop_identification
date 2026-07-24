from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.signal import savgol_filter

from harvest_estimation.create_doy_prediction_input.config import (
    load_seeding_config_yaml,
)
from harvest_estimation.create_doy_prediction_input.log import PipelineLogger

RULE_NAME = "sustained_rise_next3_count2_delta0.01_total0.03"


@dataclass(frozen=True)
class SeedingEstimate:
    greenup_start: Optional[datetime]
    seeding_date: Optional[datetime]
    source_index: str
    rule: str
    offset_days: Optional[int]
    window_start: str
    window_end: str


class SeedingEstimator:
    def __init__(
        self,
        *,
        min_points: int,
        savgol_window: int,
        savgol_polyorder: int,
        seeding_config: dict[str, dict[str, object]],
        logger: Optional[PipelineLogger] = None,
    ) -> None:
        self.min_points = int(min_points)
        self.savgol_window = int(savgol_window)
        self.savgol_polyorder = int(savgol_polyorder)
        self.seeding_config = {
            crop_name: dict(values) for crop_name, values in seeding_config.items()
        }
        self.logger = logger or PipelineLogger(name="SeedingEstimator")

    @classmethod
    def from_config(
        cls, cfg, logger: Optional[PipelineLogger] = None
    ) -> "SeedingEstimator":
        return cls(
            min_points=cfg.min_points,
            savgol_window=cfg.savgol_window,
            savgol_polyorder=cfg.savgol_polyorder,
            seeding_config=cfg.seeding_config,
            logger=logger,
        )

    @classmethod
    def from_yaml(
        cls,
        seeding_config_yaml: str | Path | None,
        *,
        min_points: int = 11,
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        logger: Optional[PipelineLogger] = None,
    ) -> "SeedingEstimator":
        return cls(
            min_points=min_points,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            seeding_config=load_seeding_config_yaml(seeding_config_yaml),
            logger=logger,
        )

    def estimate(
        self,
        dates: list[datetime],
        ndvi_series: Iterable[float],
        evi_series: Iterable[float],
        *,
        crop_name: str,
        ndvi_smoothed: Optional[np.ndarray] = None,
        evi_smoothed: Optional[np.ndarray] = None,
    ) -> SeedingEstimate:
        settings = self.seeding_config.get(crop_name, {})
        window_start = str(settings.get("window_start", ""))
        window_end = str(settings.get("window_end", ""))
        offset_days = settings.get("offset_days")
        offset_days = None if offset_days in (None, "", "null") else int(offset_days)

        blank = SeedingEstimate(
            greenup_start=None,
            seeding_date=None,
            source_index="",
            rule="",
            offset_days=offset_days,
            window_start=window_start,
            window_end=window_end,
        )

        if not dates or not window_start or not window_end:
            return blank

        year = dates[0].year
        window_start_dt = datetime.strptime(f"{year}-{window_start}", "%Y-%m-%d")
        window_end_dt = datetime.strptime(f"{year}-{window_end}", "%Y-%m-%d")

        ndvi_smoothed = (
            ndvi_smoothed if ndvi_smoothed is not None else self._smooth(ndvi_series)
        )
        greenup_start = self._find_greenup_start(
            dates, ndvi_smoothed, window_start_dt, window_end_dt
        )
        source_index = ""

        if greenup_start is not None:
            source_index = "NDVI"
        else:
            evi_smoothed = (
                evi_smoothed if evi_smoothed is not None else self._smooth(evi_series)
            )
            greenup_start = self._find_greenup_start(
                dates, evi_smoothed, window_start_dt, window_end_dt
            )
            if greenup_start is not None:
                source_index = "EVI"

        if greenup_start is None:
            return blank

        seeding_date = None
        if offset_days is not None:
            candidate = greenup_start - timedelta(days=offset_days)
            if candidate.year == year:
                seeding_date = candidate

        return SeedingEstimate(
            greenup_start=greenup_start,
            seeding_date=seeding_date,
            source_index=source_index,
            rule=RULE_NAME,
            offset_days=offset_days,
            window_start=window_start,
            window_end=window_end,
        )

    def _smooth(self, series: Iterable[float]) -> Optional[np.ndarray]:
        values = np.asarray(list(series), dtype=float)
        n = len(values)
        if n == 0:
            return None

        valid_mask = np.isfinite(values)
        valid_count = int(np.count_nonzero(valid_mask))
        if n < 3 or valid_count < self.min_points:
            return None

        if valid_count < n:
            values = np.interp(
                np.arange(n), np.flatnonzero(valid_mask), values[valid_mask]
            )

        window_length = min(self.savgol_window, n)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length <= self.savgol_polyorder:
            window_length = self.savgol_polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        if window_length > n:
            window_length = n if n % 2 == 1 else n - 1

        if window_length < 3:
            return values.copy()

        return savgol_filter(
            values,
            window_length=window_length,
            polyorder=min(self.savgol_polyorder, window_length - 1),
        )

    def _find_greenup_start(
        self,
        dates: list[datetime],
        smoothed: Optional[np.ndarray],
        window_start: datetime,
        window_end: datetime,
    ) -> Optional[datetime]:
        if smoothed is None or len(smoothed) != len(dates):
            return None

        for idx, current_date in enumerate(dates):
            if current_date < window_start or current_date > window_end:
                continue

            next_idx = idx + 3
            if next_idx >= len(smoothed):
                continue

            deltas = np.diff(smoothed[idx : next_idx + 1])
            positive_count = int(np.count_nonzero(deltas > 0.01))
            total_rise = float(smoothed[next_idx] - smoothed[idx])
            if positive_count >= 2 and total_rise >= 0.03:
                return current_date

        return None
