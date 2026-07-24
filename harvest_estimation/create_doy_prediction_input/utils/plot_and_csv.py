import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from harvest_estimation.create_doy_prediction_input.seeding_estimator import (
    SeedingEstimate,
)


def _plot_harvest_detection(
    dates,
    ndvi,
    ndvi_smooth,
    ndwi,
    ndwi_smooth,
    evi,
    evi_smooth,
    pred_start,
    pred_end,
    crop_name,
    graph_path,
    gt_windows=None,
    green_mask=None,
    farm_id=None,
    farm_mask=None,
    logger=None,
):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import Circle

    if green_mask is None:
        plt.figure(figsize=(10, 5))

        plt.plot(dates, ndvi, linestyle="--", alpha=0.4, label="NDVI (Raw)")
        plt.plot(dates, ndvi_smooth, linewidth=2, label="NDVI (Smoothed)")

        plt.plot(dates, ndwi, linestyle="--", alpha=0.4, label="NDWI (Raw)")
        plt.plot(dates, ndwi_smooth, linewidth=2, label="NDWI (Smoothed)")

        plt.plot(dates, evi, linestyle="--", alpha=0.4, label="EVI (Raw)")
        plt.plot(dates, evi_smooth, linewidth=2, label="EVI (Smoothed)")

        if pred_start:
            plt.axvline(
                pred_start, color="orange", linestyle=":", label="Start of Harvest"
            )
        if pred_end:
            plt.axvline(pred_end, color="red", linestyle="-.", label="End of Harvest")

        if (
            gt_windows is not None
            and crop_name in gt_windows
            and pred_start is not None
        ):
            gt_start_str, gt_end_str = gt_windows[crop_name]
            year = pred_start.year

            try:
                gt_start_dt = datetime.strptime(f"{year}-{gt_start_str}", "%Y-%m-%d")
                gt_end_dt = datetime.strptime(f"{year}-{gt_end_str}", "%Y-%m-%d")

                plt.axvspan(
                    gt_start_dt,
                    gt_end_dt,
                    color="gray",
                    alpha=0.2,
                    label="Typical Harvest Window",
                )
            except Exception as e:
                logger.warning(f"[Plot] Failed to parse GT window for {crop_name}: {e}")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        plt.title(f"{crop_name} - Multi-Index Harvest Detection")
        plt.xlabel("Date")
        plt.ylabel("Index Value")
        plt.ylim(-1.5, 1.5)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
    elif green_mask is not None:
        plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

        ax0 = plt.subplot(gs[0])
        ax0.imshow(green_mask)
        ax0.set_title(f"Farm Mask - {farm_id}")
        ax0.axis("off")

        y_coords, x_coords = np.where(farm_mask)
        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            circle = Circle(
                (center_x, center_y),
                radius=200,
                edgecolor="red",
                facecolor="none",
                linewidth=0.5,
            )
            ax0.add_patch(circle)

        ax1 = plt.subplot(gs[1])
        ax1.plot(dates, ndvi, linestyle="--", alpha=0.4, label="NDVI (Raw)")
        if ndvi_smooth is not None:
            ax1.plot(dates, ndvi_smooth, linewidth=2, label="NDVI (Smoothed)")

        ax1.plot(dates, ndwi, linestyle="--", alpha=0.4, label="NDWI (Raw)")
        if ndwi_smooth is not None:
            ax1.plot(dates, ndwi_smooth, linewidth=2, label="NDWI (Smoothed)")

        ax1.plot(dates, evi, linestyle="--", alpha=0.4, label="EVI (Raw)")
        if evi_smooth is not None:
            ax1.plot(dates, evi_smooth, linewidth=2, label="EVI (Smoothed)")

        ax1.axvline(pred_start, color="orange", linestyle=":", label="Predicted Start")
        ax1.axvline(pred_end, color="red", linestyle="-.", label="Predicted End")

        # GT window shading (same idea as your earlier request)
        if (
            gt_windows is not None
            and crop_name in gt_windows
            and pred_start is not None
        ):
            ws, we = gt_windows[crop_name]
            try:
                gt_start = datetime.strptime(f"{pred_start.year}-{ws}", "%Y-%m-%d")
                gt_end = datetime.strptime(f"{pred_start.year}-{we}", "%Y-%m-%d")
                ax1.axvspan(
                    gt_start,
                    gt_end,
                    color="gray",
                    alpha=0.2,
                    label="Typical Harvest Window",
                )
            except Exception as e:
                logger.warning(f"[Plot] Failed to parse GT window for {crop_name}: {e}")

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)

        ax1.set_title(f"{crop_name} / Farm {farm_id} - Harvest Detection")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Index Value")
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True)
        ax1.legend(loc="best")

        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()


def _write_crop_summary_csv(
    csv_path,
    crop_name: str,
    timestamps: Sequence[datetime],
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    evi: np.ndarray,
    pred_start: Optional[datetime],
    pred_end: Optional[datetime],
    chosen_rule: str = "",
    div_start: str = "",
    div_end: str = "",
    iou: Any = "",
    summary_rows: Optional[list] = None,
    # Optional context (farm-level)
    farm_id: Optional[str] = None,
    seeding_estimate: Optional[SeedingEstimate] = None,
    div_start_header: str = "div_start",
    div_end_header: str = "div_end",
    include_farm_id_column: Optional[bool] = None,
) -> None:
    """
    Write one summary CSV for either:
      - global crop-level (farm_id=None)
      - farm-level (farm_id provided)

    Behavior:
      - Writes regular rows for timestamps where at least one index is non-NaN
      - Also ensures Start/End marker rows exist even if not in timestamps
      - Appends rows to summary_rows (if provided)
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build mapping from timestamp -> (ndvi, ndwi, evi)
    # timestamps can be list[datetime]; ndvi/ndwi/evi are arrays aligned with timestamps
    value_map: dict[datetime, tuple[float, float, float]] = {}
    for i, ts in enumerate(timestamps):
        value_map[ts] = (float(ndvi[i]), float(ndwi[i]), float(evi[i]))

    # Create a unified sorted timeline that includes pred_start/pred_end
    all_ts = list(timestamps)
    if pred_start is not None and pred_start not in value_map:
        all_ts.append(pred_start)
    if pred_end is not None and pred_end not in value_map:
        all_ts.append(pred_end)
    if (
        seeding_estimate is not None
        and seeding_estimate.greenup_start is not None
        and seeding_estimate.greenup_start not in value_map
    ):
        all_ts.append(seeding_estimate.greenup_start)
    if (
        seeding_estimate is not None
        and seeding_estimate.seeding_date is not None
        and seeding_estimate.seeding_date not in value_map
    ):
        all_ts.append(seeding_estimate.seeding_date)

    # Remove duplicates and sort
    all_ts = sorted(set(all_ts))

    # Header
    base_header = [
        "Date",
        "NDVI",
        "NDWI",
        "EVI",
        "Harvest",
        "start_rule",
        div_start_header,
        div_end_header,
        "IoU",
        "SeedingEvent",
        "greenup_rule",
        "greenup_source",
        "seeding_offset_days",
        "seeding_window_start",
        "seeding_window_end",
    ]
    if include_farm_id_column is None:
        include_farm_id_column = farm_id is not None
    header = (["FarmID"] + base_header) if include_farm_id_column else base_header

    def _fmt(x: float) -> str:
        return (
            "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ts in all_ts:
            v_ndvi, v_ndwi, v_evi = value_map.get(ts, (np.nan, np.nan, np.nan))

            # Skip fully-empty rows unless it's Start/End
            is_start = pred_start is not None and ts == pred_start
            is_end = pred_end is not None and ts == pred_end
            is_greenup = (
                seeding_estimate is not None
                and seeding_estimate.greenup_start is not None
                and ts == seeding_estimate.greenup_start
            )
            is_seeding = (
                seeding_estimate is not None
                and seeding_estimate.seeding_date is not None
                and ts == seeding_estimate.seeding_date
            )

            if (np.isnan(v_ndvi) and np.isnan(v_ndwi) and np.isnan(v_evi)) and not (
                is_start or is_end or is_greenup or is_seeding
            ):
                continue

            if is_start:
                harvest_flag = "Start"
                rule = chosen_rule
                ds = div_start
                de = ""
                iou_cell = iou
            elif is_end:
                harvest_flag = "End"
                rule = ""
                ds = ""
                de = div_end
                iou_cell = ""
            else:
                harvest_flag = ""
                rule = ""
                ds = ""
                de = ""
                iou_cell = ""

            seeding_events: list[str] = []
            if is_greenup:
                seeding_events.append("GreenupStart")
            if is_seeding:
                seeding_events.append("EstimatedSeedingDate")

            seeding_event = ";".join(seeding_events)
            if seeding_event:
                greenup_rule = (
                    seeding_estimate.rule if seeding_estimate is not None else ""
                )
                greenup_source = (
                    seeding_estimate.source_index
                    if seeding_estimate is not None
                    else ""
                )
                seeding_offset_days = (
                    ""
                    if seeding_estimate is None or seeding_estimate.offset_days is None
                    else str(seeding_estimate.offset_days)
                )
                seeding_window_start = (
                    seeding_estimate.window_start
                    if seeding_estimate is not None
                    else ""
                )
                seeding_window_end = (
                    seeding_estimate.window_end if seeding_estimate is not None else ""
                )
            else:
                greenup_rule = ""
                greenup_source = ""
                seeding_offset_days = ""
                seeding_window_start = ""
                seeding_window_end = ""

            row = [
                ts.strftime("%Y-%m-%d"),
                _fmt(v_ndvi),
                _fmt(v_ndwi),
                _fmt(v_evi),
                harvest_flag,
                rule,
                ds,
                de,
                iou_cell,
                seeding_event,
                greenup_rule,
                greenup_source,
                seeding_offset_days,
                seeding_window_start,
                seeding_window_end,
            ]

            if include_farm_id_column:
                row = [farm_id] + row

            writer.writerow(row)

            if summary_rows is not None:
                record = {
                    "Crop": crop_name,
                    "FarmID": farm_id if farm_id is not None else "",
                    "Date": ts.strftime("%Y-%m-%d"),
                    "NDVI": v_ndvi,
                    "NDWI": v_ndwi,
                    "EVI": v_evi,
                    "Harvest": harvest_flag,
                    "start_rule": rule,
                    div_start_header: ds,
                    div_end_header: de,
                    "IoU": iou_cell,
                    "SeedingEvent": seeding_event,
                    "greenup_rule": greenup_rule,
                    "greenup_source": greenup_source,
                    "seeding_offset_days": seeding_offset_days,
                    "seeding_window_start": seeding_window_start,
                    "seeding_window_end": seeding_window_end,
                }
                summary_rows.append(record)
