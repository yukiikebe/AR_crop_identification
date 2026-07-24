# harvest_detector.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
from scipy.signal import savgol_filter

from harvest_estimation.create_doy_prediction_input.config import Config
from harvest_estimation.create_doy_prediction_input.log import PipelineLogger


@dataclass
class HarvestDetectionResult:
    start: Optional[datetime]
    end: Optional[datetime]
    smoothed: Optional[np.ndarray]
    start_rule: str


class HarvestDetector:
    """
    Detect harvest period from a single index time-series.

    Input:
      - dates: list[datetime] (sorted)
      - series: iterable[float] (same length as dates; may contain NaN)

    Output:
      - HarvestDetectionResult(start, end, smoothed, start_rule)
    """

    def __init__(self, cfg: Config, logger: Optional[PipelineLogger] = None):
        self.cfg = cfg
        self.logger = logger or PipelineLogger(name="HarvestDetector")

    def _log(self, level: str, msg: str) -> None:
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(msg)

    def detect(
        self,
        dates: List[datetime],
        series: Iterable[float],
        *,
        label: Optional[str] = None,
    ) -> HarvestDetectionResult:
        min_points = int(self.cfg.min_points)
        prefer_window = int(self.cfg.savgol_window)
        polyorder = int(self.cfg.savgol_polyorder)

        start_rule = "steepest_decline_after_peak"

        y = np.asarray(list(series), dtype=float)
        n = len(y)

        if n != len(dates):
            raise ValueError(f"dates length ({len(dates)}) != series length ({n})")

        valid_mask = np.isfinite(y)
        valid_count = int(np.count_nonzero(valid_mask))

        if n < 3 or valid_count < min_points:
            name = f" ({label})" if label else ""
            self._log(
                "info",
                f"Skipping{name} – only {valid_count} valid points (<{min_points}).",
            )
            return HarvestDetectionResult(None, None, None, "insufficient_data")

        # fill NaNs(can't calculate NDVI/NDWI/EVI) by linear interpolation
        if valid_count < n:
            y = np.interp(np.arange(n), np.flatnonzero(valid_mask), y[valid_mask])

        # choose safe Savitzky–Golay window
        w = min(prefer_window, n)
        if w % 2 == 0:
            w -= 1
        if w <= polyorder:
            w = polyorder + 2
            if w % 2 == 0:
                w += 1
        if w > n:
            w = n if n % 2 == 1 else n - 1

        if w < 3:
            smoothed = y.copy()
        else:
            smoothed = savgol_filter(
                y, window_length=w, polyorder=min(polyorder, w - 1)
            )

        # Derivative: use numerical gradient
        deriv = np.gradient(smoothed)

        # Peak of growth then find decline
        peak_idx = int(np.argmax(smoothed))
        if peak_idx <= 0 or peak_idx >= n - 1:
            return HarvestDetectionResult(None, None, smoothed, "boundary_fallback")

        post_peak_deriv = deriv[peak_idx + 1 :]
        if len(post_peak_deriv) == 0:
            start_idx = min(peak_idx, n - 1)
            start_rule = "no_derivative_after_peak"
        else:
            start_idx = peak_idx + int(np.argmin(post_peak_deriv))

        end_idx = peak_idx + int(np.argmin(smoothed[peak_idx:]))

        # clamp
        start_idx = max(0, min(start_idx, n - 1))
        end_idx = max(0, min(end_idx, n - 1))

        # if start >= end, try a threshold+persistence fallback
        if start_idx >= end_idx and start_idx != peak_idx:
            alpha = 0.85
            k_persist = 3

            def first_persistent_idx(arr, cond, k):
                run = 0
                for i, v in enumerate(arr):
                    run = run + 1 if cond(v) else 0
                    if run >= k:
                        return i - k + 1
                return None

            # after peak and before end
            right_bound = max(peak_idx + 1, min(end_idx, n - 1))
            segment = smoothed[peak_idx + 1 : right_bound]
            threshold = alpha * smoothed[peak_idx]

            under = first_persistent_idx(segment, lambda v: v <= threshold, k_persist)
            if under is not None:
                start_idx = (peak_idx + 1) + under
                start_rule = f"threshold_{alpha}_p{k_persist}"
            else:
                post_deriv = np.gradient(smoothed)[peak_idx + 1 : right_bound]
                start_idx = (
                    (peak_idx + 1) + int(np.argmin(post_deriv))
                    if len(post_deriv)
                    else peak_idx
                )
                start_rule = (
                    "steepest_decline_after_peak_retry"
                    if len(post_deriv)
                    else "no_derivative_after_peak_retry"
                )

            # ensure start < end
            start_idx = max(peak_idx + 1, min(start_idx, end_idx - 1))

        return HarvestDetectionResult(
            dates[start_idx], dates[end_idx], smoothed, start_rule
        )
