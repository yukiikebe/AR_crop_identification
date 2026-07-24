from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from harvest_estimation.create_doy_prediction_input.config import Config
from harvest_estimation.create_doy_prediction_input.log import PipelineLogger


class HarvestVoter:
    """
    Choose final harvest start/end date from multiple sources (NDVI/NDWI/EVI),
    optionally comparing against GT window (cfg.gt_windows).
    """

    def __init__(self, cfg: Config, logger: Optional[PipelineLogger] = None):
        self.cfg = cfg
        self.logger = logger or PipelineLogger(name="HarvestVoter")

    def _log(self, level: str, msg: str) -> None:
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(msg)

    def _gt_target(self, crop_name: str, *, kind: str, year: int) -> Optional[datetime]:
        """
        Returns GT datetime (start or end) if configured, else None.
        """
        gt_window = self.cfg.gt_windows.get(crop_name)
        if not gt_window:
            return None
        ws, we = gt_window
        month_date = ws if kind == "start" else we
        if not month_date:
            return None
        return datetime.strptime(f"{year}-{month_date}", "%Y-%m-%d")

    def vote_simple(
        self,
        candidates: List[Optional[datetime]],
        dates: List[datetime],
        *,
        kind: str,  # "start" or "end"
        crop_name: str,
        year: int,
    ) -> Tuple[datetime, Optional[int]]:
        """
        Like your vote_or_fallback:
          - Majority vote among non-None dates
          - Tie-break: earliest for start, latest for end
          - Fallback: first/last observation date
        Returns: (chosen_dt, div_days_vs_gt_or_None)
        """
        assert kind in ("start", "end")
        valid = [d for d in candidates if d is not None]

        gt = self._gt_target(crop_name, kind=kind, year=year)

        if valid:
            counts = Counter(valid).most_common()
            top_count = counts[0][1]
            top_dates = [d for d, n in counts if n == top_count]

            chosen = min(top_dates) if kind == "start" else max(top_dates)
        else:
            chosen = dates[0] if kind == "start" else dates[-1]

        div = abs((chosen - gt).days) if gt is not None else None
        return chosen, div

    def vote_with_rule(
        self,
        dates_by_src: Dict[str, Optional[datetime]],
        rules_by_src: Dict[str, str],
        *,
        kind: str,  # "start" or "end"
        crop_name: str,
        year: int,
        fallback_dates: Optional[List[datetime]] = None,
    ) -> Tuple[datetime, str, str, Optional[int]]:
        """
        Like your vote_or_fallback_with_rule:
          - Choose the candidate date closest to GT target date (if exists),
            otherwise choose earliest(start) / latest(end).
          - If multiple sources have same chosen_dt, pick one by rule priority.
        Returns: (chosen_dt, chosen_source, chosen_rule, div_days_vs_gt_or_None)
        """
        assert kind in ("start", "end")
        items = [(src, dt) for src, dt in dates_by_src.items() if dt is not None]

        gt = self._gt_target(crop_name, kind=kind, year=year)

        if not items:
            # If no candidates: fall back to GT if present, else error
            if gt is not None:
                return gt, "FALLBACK", "no_valid_dates_all_sources", 0
            if fallback_dates:
                fallback = fallback_dates[0] if kind == "start" else fallback_dates[-1]
                return fallback, "FALLBACK", "no_valid_dates_all_sources", None
            raise ValueError("No valid candidate dates and no GT window available.")

        if gt is not None:
            target = gt
            chosen_dt = min(
                (d for _, d in items), key=lambda d: (abs((d - target).days), d)
            )
            div = abs((chosen_dt - target).days)
        else:
            chosen_dt = (
                min((d for _, d in items))
                if kind == "start"
                else max((d for _, d in items))
            )
            div = None

        winners = [src for src, dt in items if dt == chosen_dt]

        def rule_rank(rule: str) -> int:
            r = (rule or "").lower()
            if "steepest_decline_" in r:
                return 0
            if r.startswith("threshold"):
                return 1
            return 2

        chosen_src = min(winners, key=lambda s: (rule_rank(rules_by_src.get(s, "")), s))
        chosen_rule = rules_by_src.get(chosen_src, "")
        return chosen_dt, chosen_src, chosen_rule, div
