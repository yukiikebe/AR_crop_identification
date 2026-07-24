from datetime import datetime

import numpy as np


# Metric calculations
def calculate_ndvi(nir, red, eps=1e-10):
    return np.clip((nir - red) / (nir + red + eps), -1, 1)


def calculate_ndwi(nir, swir, eps=1e-10):
    return np.clip((nir - swir) / (nir + swir + eps), -1, 1)


def calculate_evi(nir, red, blue, eps=1e-10):
    return np.clip(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps), -1, 1)


def all_metrics(nir, red, blue, swir):
    ndvi = calculate_ndvi(nir, red)
    ndwi = calculate_ndwi(nir, swir)
    evi = calculate_evi(nir, red, blue)
    return ndvi, ndwi, evi


def IoU_calculation(start_date, end_date, harvest_window):
    ws, we = harvest_window
    assert (ws is not None) and (we is not None), "harvest_window must be provided"
    gt_start = datetime.strptime(f"2019-{ws}", "%Y-%m-%d") if ws else None
    gt_end = datetime.strptime(f"2019-{we}", "%Y-%m-%d") if we else None

    delta = max(0, (min(end_date, gt_end) - max(start_date, gt_start)).days)

    union = (max(end_date, gt_end) - min(start_date, gt_start)).days
    iou = delta / union
    return iou
