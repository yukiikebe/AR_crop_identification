# DeepSatModels — Arkansas Crop‑ID

Training + deployment code for Arkansas crop identification from Sentinel‑2 time series.

## Installation

Recommended: Conda (Python 3.10+).

```bash
conda env create -f deepsatmodels_env.yml
conda activate deepsatmodels
pip install "numpy<2"  # chainer/torchfcn deps break on NumPy 2.x
```

For serving + UI:
```bash
pip install fastapi uvicorn streamlit streamlit-folium folium requests pillow
```

## Model checkpoints (6–11 month)

Expected layout (not tracked by git; see `.gitignore`):
- `models/saved_models/AR23_focal_06mo/{config_file.yaml,best.pth}`
- `models/saved_models/AR23_focal_07mo/{config_file.yaml,best.pth}`
- `models/saved_models/AR23_focal_08mo/{config_file.yaml,best.pth}`
- `models/saved_models/AR23_focal_09mo/{config_file.yaml,best.pth}`
- `models/saved_models/AR23_focal_10mo/{config_file.yaml,best.pth}`
- `models/saved_models/AR23_focal_11mo/{config_file.yaml,best.pth}`

Deployment rule: if you have data through calendar month `M`, select `model_month = clamp(M, 6..11)`.

## Train

Example (11‑month model):
```bash
python train_and_eval/segmentation_training_transf.py --config configs/Arkansas/TSViT_AR23_11mo_focal.yaml --device 0
```

## Deployment (monthly automation)

`ar_deploy.py` downloads raw tiles for a month from Google Earth Engine and (if `month >= 6`) runs the corresponding model and writes GeoTIFF predictions.

Manual run:
```bash
python ar_deploy.py run --project <gcp-project> --data-root /mnt/vhvkhoa_ssd/datasets --year 2025 --month 6
```

Cron (03:00 on day 1 each month; downloads previous month automatically):
```bash
0 3 1 * * /path/to/miniconda3/envs/deepsatmodels/bin/python /path/to/DeepSatModels/ar_deploy.py monthly --project <gcp-project> --data-root /mnt/vhvkhoa_ssd/datasets >> /mnt/vhvkhoa_ssd/datasets/ar_monthly.log 2>&1
```

First-time EE auth (interactive): add `--auth` once when running `ar_deploy.py download/run/monthly`, then remove it for cron.

## Prediction server + client queries

Serve **precomputed** predictions:
```bash
export DEEPSAT_AR_PRED_ROOT=/mnt/vhvkhoa_ssd/datasets
uvicorn ar_pred_api:app --host 0.0.0.0 --port 8001
```

Optional Streamlit client (year/month/bbox only):
```bash
export DEEPSAT_AR_PRED_API_URL=http://localhost:8001
streamlit run app_AR_deploy.py
```

Map tiles: set `DEEPSAT_MAPBOX_TOKEN` (or `MAPBOX_TOKEN`) to use Mapbox; otherwise the apps fall back to OpenStreetMap tiles.

## CDL download (labels) for a year

Writes `cdl.tif` into each meta‑patch under your raw root:
```bash
cd data/Arkansas
Rscript get_cdl.r --base-dir /mnt/vhvkhoa_ssd/datasets/AR_2025_raw --year 2025
```

Requires an R install with packages `terra`, `sf`, and `CropScapeR`.

