# DeepSatModels — Arkansas Crop‑ID

Training + deployment code for Arkansas crop identification from Sentinel‑2 time series.

## Installation

Recommended: Conda (Python 3.10+).

```bash
conda env create -f deepsatmodels_env.yml
conda activate deepsatmodels_crop_id
pip install "numpy<2"  # chainer/torchfcn deps break on NumPy 2.x
```

For serving + UI:
```bash
pip install fastapi uvicorn streamlit streamlit-folium folium requests pillow
```

FastDiffSR runs in a separate existing environment:
```bash
conda activate super_res310
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

python maryland_deploy.py download --auth --project ee-yukiikebe20010210 --data-root /home/yikebe/md_sentinel2 --year 2023 --whole-year --roi-bounds -76.78 38.07 -76.25462 38.616701 --download-workers 8 

we can add like --mask-sea
```

Cron (03:00 on day 1 each month; downloads previous month automatically):
```bash
0 3 1 * * /path/to/miniconda3/envs/deepsatmodels/bin/python /path/to/DeepSatModels/ar_deploy.py monthly --project <gcp-project> --data-root /mnt/vhvkhoa_ssd/datasets >> /mnt/vhvkhoa_ssd/datasets/ar_monthly.log 2>&1
```

First-time EE auth (interactive): add `--auth` once when running `ar_deploy.py download/run/monthly`, then remove it for cron.

Arkansas statewide monthly mosaic download:
```bash
python ar_deploy.py statewide_download --project <gcp-project> --data-root /mnt/vhvkhoa_ssd/datasets --year 2025 --month 6
```

This writes monthly statewide mosaics under:
- `/mnt/vhvkhoa_ssd/datasets/AR_2025_statewide/2025-06/`
- files like `B2_2025-06.tif`, `SCL_2025-06.tif`, `TCI_2025-06.jpg`, `_statewide_meta.json`

Retile a statewide monthly mosaic back to the canonical Arkansas raw grid:
```bash
python ar_deploy.py statewide_retile \
  --in-dir /mnt/vhvkhoa_ssd/datasets/AR_2025_statewide/2025-06 \
  --out-root /mnt/vhvkhoa_ssd/datasets/AR_2025_raw \
  --year 2025 \
  --month 6
```

Notes:
- Existing `download`, `run`, and `monthly` commands still use per-grid raw downloads.
- `statewide_download` creates one monthly median composite per calendar month over the Arkansas bounding box.
- The monthly composite is useful for visualization, SR, and lightweight downstream workflows, but it is not a drop-in replacement for the original multi-date crop-ID time series.
- If you need existing preprocessing/inference compatibility, run `statewide_retile` first and then point downstream steps at `AR_<year>_raw`.

## Prediction server + client queries

Serve **precomputed** predictions:
```bash
export DEEPSAT_AR_PRED_ROOT=/mnt/vhvkhoa_ssd/datasets
python3 -m uvicorn ar_pred_api:app --host 0.0.0.0 --port 8001
```

FastDiffSR API (separate service, backend-managed settings):
```bash
conda activate super_res310
# If you hit:
#   python: symbol lookup error: ... libmkl_intel_thread.so.2: undefined symbol: __kmpc_global_thread_num
# use this runtime workaround (also applies to `ar_deploy.py fastdiffsr_retile`):
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE=~/AR_sentinel2_app/AR_{year}_raw
export DEEPSAT_FASTDIFFSR_PREVIEW_SWAP_RB=true



# Optional overrides. If omitted, API defaults are:
#   DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE=../../AR_sentinel2/{year}_AR
#   DEEPSAT_FASTDIFFSR_CHECKPOINT=/home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_train_64_256_Planet_251026_032716/checkpoint/I210496_E1120
#   DEEPSAT_FASTDIFFSR_CONFIG=FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_planet.json
# export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE=/path/to/AR_sentinel2/{year}_AR
# export DEEPSAT_FASTDIFFSR_CHECKPOINT=/path/to/checkpoint/I210496_E1120
# export DEEPSAT_FASTDIFFSR_CONFIG=/path/to/sr_fastdiffsr_infer_x4_planet.json
export DEEPSAT_FASTDIFFSR_OUT_ROOT=/path/to/AR_fastdiffsr
# Optional: if you built monthly statewide mosaics with `fastdiffsr_mosaic`,
# the API will prefer them for bbox preview reads.
# Layout expected:
#   <DEEPSAT_FASTDIFFSR_MOSAIC_ROOT>/AR_{year}_fastdiffsr/{year}_{month:02d}.tif
export DEEPSAT_FASTDIFFSR_MOSAIC_ROOT=/path/to/AR_fastdiffsr_mosaic

# Optional overrides (defaults shown):
# export DEEPSAT_FASTDIFFSR_DEVICE=0
# export DEEPSAT_FASTDIFFSR_SCALE=4
# export DEEPSAT_FASTDIFFSR_LR_TILE_SIZE=64
# export DEEPSAT_FASTDIFFSR_DATE_POLICY=latest
# export DEEPSAT_FASTDIFFSR_CLOUD_THRESH=0.10
# export DEEPSAT_FASTDIFFSR_FORCE=false
export DEEPSAT_FASTDIFFSR_DATE_POLICY=statewide_anchor
python3 -m uvicorn ar_fastdiffsr_api:app --host 0.0.0.0 --port 8002
```

FastDiffSR output layout:
- `<DEEPSAT_FASTDIFFSR_OUT_ROOT>/AR_{year}_fastdiffsr/<MM>mo/<meta_patch>.tif`
- Job status files: `<DEEPSAT_FASTDIFFSR_OUT_ROOT>/_jobs/<job_id>.json`

Bootstrap precompute (one-time, all AR patches for 2019-01):
```bash
conda activate super_res310
python3 ar_deploy.py fastdiffsr \
  --year 2019 \
  --month 1 \
  --raw-root /path/to/AR_sentinel2/AR_2019 \
  --sr-root /path/to/AR_fastdiffsr/AR_2019_fastdiffsr \
  --fastdiffsr-checkpoint "$DEEPSAT_FASTDIFFSR_CHECKPOINT" \
  --fastdiffsr-config "$DEEPSAT_FASTDIFFSR_CONFIG" \
  --device "${DEEPSAT_FASTDIFFSR_DEVICE:-0}" \
  --scale "${DEEPSAT_FASTDIFFSR_SCALE:-4}" \
  --lr-tile-size "${DEEPSAT_FASTDIFFSR_LR_TILE_SIZE:-64}" \
  --date-policy "${DEEPSAT_FASTDIFFSR_DATE_POLICY:-latest}" \
  --cloud-thresh-fastdiffsr "${DEEPSAT_FASTDIFFSR_CLOUD_THRESH:-0.10}"
```

Retile existing FastDiffSR month outputs to the canonical Arkansas 20x20 grid (no re-inference):
```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
python -B ar_deploy.py fastdiffsr_retile \
  --in-dir /path/to/AR_fastdiffsr/AR_2019_fastdiffsr/01mo \
  --out-dir /path/to/AR_fastdiffsr_retile/AR_2019_fastdiffsr/01mo
```

Build one statewide FastDiffSR GeoTIFF mosaic for faster bbox extraction:
```bash
conda activate super_res310
python -B ar_deploy.py fastdiffsr_mosaic \
  --in-dir /path/to/AR_fastdiffsr/AR_2019_fastdiffsr/01mo \
  --out-file /path/to/AR_fastdiffsr_mosaic/AR_2019_fastdiffsr/2019_01.tif
```

Notes:
- This reads existing monthly FastDiffSR tiles like `0_0.tif`, `0_1.tif`, and writes one compressed GeoTIFF.
- Internal overviews are built by default, which makes server-side window reads faster for interactive bbox requests.
- `date_policy=all` month directories are not supported by this mosaic command yet; it expects one output tile per meta-patch.

Monthly append precompute (separate cron in `super_res310`):
```bash
# Example: run at 05:00 on day 1, for previous month
0 5 1 * * bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate super_res310
  Y=$(date -d "last month" +\%Y)
  M=$(date -d "last month" +\%m)
  python3 /home/yikebe/research/AR_Crop_Identification/ar_deploy.py fastdiffsr \
    --year "$Y" --month "$M" \
    --raw-root "/path/to/AR_sentinel2/AR_${Y}" \
    --sr-root "/path/to/AR_fastdiffsr/AR_${Y}_fastdiffsr" \
    --fastdiffsr-checkpoint "$DEEPSAT_FASTDIFFSR_CHECKPOINT" \
    --fastdiffsr-config "$DEEPSAT_FASTDIFFSR_CONFIG" \
    --device "${DEEPSAT_FASTDIFFSR_DEVICE:-0}" \
    --scale "${DEEPSAT_FASTDIFFSR_SCALE:-4}" \
    --lr-tile-size "${DEEPSAT_FASTDIFFSR_LR_TILE_SIZE:-64}" \
    --date-policy "${DEEPSAT_FASTDIFFSR_DATE_POLICY:-latest}" \
    --cloud-thresh-fastdiffsr "${DEEPSAT_FASTDIFFSR_CLOUD_THRESH:-0.10}" \
    >> /path/to/AR_fastdiffsr/fastdiffsr_monthly.log 2>&1
'
```

Optional Streamlit client (year/month/bbox only):
```bash
export DEEPSAT_AR_PRED_API_URL=http://localhost:8001
export DEEPSAT_FASTDIFFSR_API_URL=http://localhost:8002
streamlit run app_AR_deploy.py
```

Example request:
```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2019,
    "month": 6,
    "bbox": {
      "lon_min": -94.2,
      "lat_min": 34.7,
      "lon_max": -93.8,
      "lat_max": 35.0
    }
  }'
```

Async behavior for missing SR data:
- If requested SR files already exist: `POST /predict` returns `200` with outputs.
- If requested SR files are missing: `POST /predict` returns `202 Accepted` with `job_id` and `status_url`.
- Poll job status:
```bash
curl http://localhost:8002/jobs/<job_id>
```

Troubleshooting:
- `404 raw_root not found`: `DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE` resolves to a non-existent path for the selected year/month.
- `422 Unprocessable Entity`: request body does not match strict schema (`year`, `month`, `bbox` only).
- `500 Missing required backend env`: default + env are both empty (rare; usually only if code defaults were removed).
- `202 Accepted` repeats for same query: backend is still generating; keep polling `/jobs/<job_id>` until `state=completed|failed|interrupted`.

Map tiles: set `DEEPSAT_MAPBOX_TOKEN` (or `MAPBOX_TOKEN`) to use Mapbox; otherwise the apps fall back to OpenStreetMap tiles.

## CDL download (labels) for a year

Writes `cdl.tif` into each meta‑patch under your raw root:
```bash
cd data/Arkansas
Rscript get_cdl.r --base-dir /mnt/vhvkhoa_ssd/datasets/AR_2025_raw --year 2025
```

Requires an R install with packages `terra`, `sf`, and `CropScapeR`.

### Rerun command(SR)

(super_res310) c2108:yikebe:~/research/AR_Crop_Identification$ python -B ar_deploy.py fastdiffsr_rerun_incomplete   --year 2019   --month 1   --raw-root ../../AR_sentinel2_app/AR_2019_raw   --sr-root ./fastdiffsr_preds/AR_2019_fastdiffsr   --fastdiffsr-checkpoint /home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_train_64_256_Planet_260220_163421/checkpoint/I283712_E757 --fastdiffsr-config FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_planet.json   --device 0   --date-policy statewide_anchor   --min-size-mb 120

## Run super resolution multiple GPUs
 python ar_deploy.py fastdiffsr --year 2019 --month 1 --raw-root /home/yuki/AR_sentinel2_app_vr3/AR_2019_raw --sr-root /home/yuki/AR_sentinel2_fastdiffsr/AR_2019_fastdiffsr --fastdiffsr-checkpoint /home/yuki/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_train_64_256_Planet_260220_163421/checkpoint/I283712_E757 --fastdiffsr-config FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_planet.json --device 0 --date-policy latest --lr-tile-size 64 --lr-overlap 16 --meta-patches {0..4}_{0..19}

## Make a combined big GEOTiFF file
python -B ar_deploy.py fastdiffsr_mosaic --in-dir /home/yuki/AR_sentinel2_fastdiffsr/AR_2019_fastdiffsr/01mo --out-file /home/yuki/AR_sentinel2_fastdiffsr_mosaic/AR_2019_fastdiffsr/2019_01.tif


## latest download command

cd /home/yikebe/research/AR_Crop_Identification && sed -n '/^# pm5__20241205_20241224__9833/,$p' make_dataset/run_sentinel_downloads_pm5.sh | sed '/^python ar_deploy.py download / s#$# --grid-target-width-deg 1e-6 --grid-target-height-deg 1e-6 --grid-max-splits-per-axis 20#' | bash


cd /home/yikebe/research/AR_Crop_Identification
sed -n '/^# pm5__20210307_20210324__0462/,$p' make_dataset/run_sentinel_downloads_pm5.sh | sed '/^python ar_deploy.py download / s#$# --grid-target-width-deg 1e-6 --grid-target-height-deg 1e-6 --grid-max-splits-per-axis 20#' | bash
# make dataset
# progress downloading sentinel2

cd /home/yikebe/research/AR_Crop_Identification && /home/yikebe/.conda/envs/super_res310/bin/python make_dataset/build_fastdiffsr_sentinel_planet_dataset.py --planet-root /home/thanyu/planet_data/analytic_8b_sr_udm2 --sentinel-root make_dataset/raw_sentinel/pm5 --train-out make_dataset/fastdiffsr_s2_planet/train --val-out make_dataset/fastdiffsr_s2_planet/val --sentinel-patch-size 16 --planet-patch-size 64 --scale 4 --max-scl-cloud-ratio 0.10

# last

LD_PRELOAD=/home/yikebe/.conda/envs/super_res310/lib/libgomp.so.1 MKL_THREADING_LAYER=GNU /home/yikebe/.conda/envs/super_res310/bin/python make_dataset/build_fastdiffsr_sentinel_planet_dataset.py --planet-root /home/thanyu/planet_data/analytic_8b_sr_udm2 --planet-years 2020 --sentinel-root make_dataset/raw_sentinel/pm5 --download-plan-tsv make_dataset/sentinel_download_plan_pm5.tsv --train-out make_dataset/fastdiffsr_s2_planet_clean_v5/train_64_256 --val-out make_dataset/fastdiffsr_s2_planet_clean_v5/val_64_256 --sentinel-patch-size 64 --planet-patch-size 256 --scale 4 --max-pair-day-diff 5 --black-threshold 30 --max-black-ratio 0 --max-scl-cloud-ratio 0 --max-planet-udm2-cloud-ratio 0 --max-mean-brightness 230 --max-bright-ratio 0.25 --max-white-ratio 0.15 --min-mean-brightness 60 --max-dark-ratio 0.15 --max-sentinel-mean-brightness 230 --max-sentinel-bright-ratio 0.35 --max-sentinel-white-ratio 0.1 --min-sentinel-mean-brightness 40 --max-sentinel-dark-ratio 0.25 --num-workers 16 --no-resume --min-edge-corr 0.2 --edge-blur-sigma 1.5 --color-match-sentinel --color-match-method percentile --max-color-mad 11 --max-color-p95-diff 23 --color-diff-blur-sigma 8


# filter out too brightness pairs
/home/yikebe/.conda/envs/super_res310/bin/python make_dataset/filter_fastdiffsr_bright_pairs.py --input-root make_dataset/fastdiffsr_s2_planet_strict --output-root make_dataset/fastdiffsr_s2_planet_strict_brightness_filtered --hr-max-mean-brightness 240 --hr-max-bright-ratio 0.35 --lr-max-mean-brightness 245 --lr-max-bright-ratio 0.50


python sr_mfe.py -p train -c /home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/config/sr_fastdiffsr_train_64_256_sentinel_planet.json -enable_wandb -log_wandb_ckpt -log_eval

Generating URL ...                                                                                                                  [98/1859]
Data downloaded to /scrfs/storage/yikebe/home/research/AR_Crop_Identification/make_dataset/raw_sentinel/pm5/pm5__20210507_20210521__0618/AR_2
021_raw/9_8/2021-05-13/B4_2021-05-13.tmp.tif                                                                                                 
Generating URL ...                                                                                                                           
Downloading data from https://earthengine.googleapis.com/v1/projects/ee-yukiikebe20010210/thumbnails/e8cef60ecf4c9e0e8c9e0f419d4258ca-7fbb04f
0640bd17fc9cb48c891ef3263:getPixels                                                                                                          
Please wait ...                                                                                                                              
Downloading data from https://earthengine.googleapis.com/v1/projects/ee-yukiikebe20010210/thumbnails/63a58a8668a596fd9019fdaa64e7c6bc-e93b0bf
ec8251fec35a0d6bbe8c58017:getPixels                                                                                                          
Please wait ...                                                                                                                              
Data downloaded to /scrfs/storage/yikebe/home/research/AR_Crop_Identification/make_dataset/raw_sentinel/pm5/pm5__20210507_20210521__0618/AR_2
021_raw/9_7/2021-05-18/B3_2021-05-18.tmp.tif                                                                                                 
Generating URL ...                                                                                                                           
Data downloaded to /scrfs/storage/yikebe/home/research/AR_Crop_Identification/make_dataset/raw_sentinel/pm5/pm5__20210507_20210521__0618/AR_2
021_raw/9_8/2021-05-13/SCL_2021-05-13.tmp.tif                                                                                                
Generating URL ...                                                                                                                           
Downloading data from https://earthengine.googleapis.com/v1/projects/ee-yukiikebe20010210/thumbnails/2ccd4b681420225c28ab6e2607c963db-9b60165
e1eefc55c30992ba3f838742f:getPixels 