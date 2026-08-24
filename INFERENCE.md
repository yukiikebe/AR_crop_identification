# Phase 3: Inference

This phase covers batch crop prediction, Harvest workbook preparation, the three FastAPI services, and the Streamlit web interface.

Run commands from the repository root unless a step says otherwise. Replace values enclosed in angle brackets with paths or identifiers for your environment. Each API and the Streamlit application run as separate processes.

## Install the deployment environment

Retrieve the FastDiffSR checkpoint managed by Git LFS:

```bash
git lfs pull
ls -lh fastdiffsr/checkpoints/I283712_E757_gen.pth
```

The checkpoint should be an approximately 91 MB binary file. If it is only a few KB, it is still an LFS pointer.

Create and activate the deployment environment:

```bash
conda env create -f super_res310.yml
conda activate super_res310
```

Update an existing environment with:

```bash
conda env update -n super_res310 -f super_res310.yml --prune
conda activate super_res310
```

Check the main dependencies and CUDA availability:

```bash
python -c "import torch, rasterio, fastapi, streamlit; print('imports: OK'); print('CUDA:', torch.cuda.is_available())"
python -c "import streamlit_folium; print('streamlit-folium: OK')"
```

Crop prediction and FastDiffSR require an NVIDIA GPU with working CUDA support. If CUDA is unavailable, confirm GPU visibility with `nvidia-smi` before continuing.

## Generate crop prediction GeoTIFFs

Skip this section if precomputed crop predictions are already available. Batch crop prediction requires the crop-model code described in [Phase 2](TRAINING.md), a selected configuration, and a `best.pth` checkpoint. Crop checkpoints are not tracked; the default lookup expects them under `models/saved_models/AR23_focal_<MM>mo/`.

The batch command reads a shared `<year>_AR` Sentinel-2 directory from [Phase 1](DATA_DOWNLOAD.md) and writes one prediction GeoTIFF per grid tile:

```bash
conda activate deepsatmodels_crop_id

python ar_deploy.py predict \
  --year 2025 \
  --available-month 6 \
  --raw-root /local/data/sentinel2/2025_AR \
  --pred-root /local/data/predictions/AR_2025_preds \
  --device 0
```

The command uses the corresponding monthly configuration and checkpoint by default. Months 1-5 have no supported model, months 6-11 use their matching models, and month 12 uses the 11-month model. Use `--config` and `--checkpoint` to supply explicit model files.

The output layout is:

```text
<PRED_ROOT>/
└── AR_<year>_preds/
    └── <MM>mo/
        ├── 0_0.tif
        ├── 0_1.tif
        └── ...
```

## Prepare Harvest workbooks

The Harvest API currently serves 2024 only and uses the statewide Arkansas 20x20 grid. It requires a prepared workbook and a source raster for each grid tile:

```text
<OUTPUTS_ROOT>/2024_AR/<grid id>/harvest_summary_all_crops.xlsx
<DATASET_ROOT>/2024_AR/<grid id>/cdl.tif
```

If `cdl.tif` is absent, the API uses the first `<grid id>/*/B4_*.tif` it finds to determine the tile footprint.

The input year must contain Sentinel-2 bands and CDL data in this layout:

```text
<DATASET_ROOT>/2024_AR/
└── <grid id>/
    ├── cdl.tif
    └── <YYYY-MM-DD>/
        ├── B2_<YYYY-MM-DD>.tif
        ├── B4_<YYYY-MM-DD>.tif
        ├── B8_<YYYY-MM-DD>.tif
        ├── B11_<YYYY-MM-DD>.tif
        └── SCL_<YYYY-MM-DD>.tif
```

`B12` can replace `B11`. `SCL` is optional but recommended for filtering clouds and other invalid pixels. The `all` band preset in [Phase 1](DATA_DOWNLOAD.md) supplies the required bands.

Test one tile, replacing `17_10` with a grid ID in the dataset:

```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

python -m harvest_estimation.create_doy_prediction_input.main \
  --dataset-root /local/data/sentinel2/2024_AR \
  --cdl-yaml harvest_estimation/configs/cdl.yaml \
  --gt-windows-yaml harvest_estimation/configs/gt_windows.yaml \
  --seeding-config-yaml harvest_estimation/configs/seeding_config.yaml \
  --output-root /local/data/harvest/outputs/2024_AR \
  --all-crops \
  --no-farm \
  --no-index-images \
  --tiles 17_10
```

The inference-only flags avoid farm-level and per-date image artifacts that the API does not use. Remove `--tiles 17_10` after the test succeeds to process every grid. A full-year run is long-running; use the HPC scheduler or a supervised terminal session. The pipeline resumes from per-tile checkpoints by default.

## Configure runtime data

Set only the variables needed by the services you will run.

### Crop Identification

```bash
export DEEPSAT_AR_PRED_ROOT=/local/data/predictions
```

The directory must contain `AR_<year>_preds/<MM>mo/<grid id>.tif` as shown above.

### FastDiffSR

FastDiffSR reads `B2`, `B3`, `B4`, and `SCL` from the shared Sentinel-2 directory:

```bash
export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE='/local/data/sentinel2/{year}_AR'
```

Keep `{year}` literally in the value. The included inference assets are:

```text
Configuration: fastdiffsr/config/sr_fastdiffsr_infer_x4_planet.json
Model:         fastdiffsr/checkpoints/I283712_E757_gen.pth
Default output: ./fastdiffsr_preds
```

Optional overrides include:

```bash
export DEEPSAT_FASTDIFFSR_OUT_ROOT=/local/data/fastdiffsr_outputs
export DEEPSAT_FASTDIFFSR_MOSAIC_ROOT=/local/data/fastdiffsr_mosaics
export DEEPSAT_PS_SCENE_ROOT=/local/data/planet_metadata
export DEEPSAT_FASTDIFFSR_CONFIG=/path/to/compatible_config.json
export DEEPSAT_FASTDIFFSR_CHECKPOINT=/path/to/checkpoint
export DEEPSAT_FASTDIFFSR_DEVICE=1
```

`DEEPSAT_FASTDIFFSR_CHECKPOINT` accepts a checkpoint directory, a `_gen.pth` or `_opt.pth` file, or a checkpoint prefix.

If requested Sentinel-2 data is missing, the API can download the month from Earth Engine. Authenticate in advance. Set the following variable only when overriding the API's default Earth Engine project:

```bash
export DEEPSAT_FASTDIFFSR_EE_PROJECT=<earth-engine-project>
```

### Crop Harvest Estimation

Generate the 2024 statewide Arkansas tile predictions once. This is a GPU inference job; the API does not run the model itself.

```bash
export DEEPSAT_HARVEST_DATASET_ROOT=/local/data/sentinel2
export DEEPSAT_HARVEST_OUTPUTS_ROOT=/local/data/harvest/outputs
export DEEPSAT_HARVEST_PRED_ROOT=/local/data/harvest/predictions

python precompute_harvest_predictions.py \
  --years 2024 \
  --outputs-root "$DEEPSAT_HARVEST_OUTPUTS_ROOT" \
  --dataset-root "$DEEPSAT_HARVEST_DATASET_ROOT" \
  --predictions-root "$DEEPSAT_HARVEST_PRED_ROOT" \
  --device cuda
```

The command writes `<predictions-root>/2024/<window>/<feature-set>/predictions.csv` and `metadata.json`. Metadata includes the exact WGS84 bounds of every source tile in the statewide Arkansas grid. Existing artifacts are not replaced unless `--overwrite` is passed. Override `--code-root`, `--model-root`, `--model-window`, or `--feature-set` only when generating another compatible artifact.

Use `--model-dir` to select an exact compatible checkpoint leaf. To split crops across multiple visible CUDA devices, replace `--device cuda` with, for example, `--devices cuda:0 cuda:1 cuda:2 cuda:3`. Each device runs a separate worker and the final artifact is written only after every worker succeeds.

## Start the APIs

Open a separate terminal for each API. Enter the repository and activate the deployment environment in every terminal:

```bash
cd /path/to/AR_crop_identification
conda activate super_res310
```

On systems where MKL and OpenMP conflict, apply these settings before starting FastDiffSR or Harvest:

```bash
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
```

Start only the services you need.

### Crop Identification API

```bash
export DEEPSAT_AR_PRED_ROOT=/path/to/predictions
python -m uvicorn ar_pred_api:app --host 0.0.0.0 --port 8001
```

### FastDiffSR API

```bash
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE='/path/to/sentinel2/{year}_AR'

python -m uvicorn ar_fastdiffsr_api:app --host 0.0.0.0 --port 8002
```

### Harvest API

```bash
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export DEEPSAT_HARVEST_PRED_ROOT=/path/to/harvest/predictions

python -m uvicorn ar_harvest_api:app --host 0.0.0.0 --port 8003
```

Harvest serves only precomputed predictions. It does not load PyTorch or run model inference while handling requests. Omit `DEEPSAT_HARVEST_PRED_ROOT` when using the default `<repository>/runtime_data/harvest_predictions` location. New artifacts use their stored tile bounds; `DEEPSAT_HARVEST_DATASET_ROOT` remains the coordinate fallback for older artifacts without stored bounds.

## Start the Streamlit interface

Open another terminal:

```bash
cd /path/to/AR_crop_identification
conda activate super_res310

export DEEPSAT_AR_PRED_API_URL=http://localhost:8001
export DEEPSAT_FASTDIFFSR_API_URL=http://localhost:8002
export DEEPSAT_HARVEST_API_URL=http://localhost:8003

python -m streamlit run app_AR_deploy.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType none
```

Open `http://localhost:8501` when the browser is on the same machine. For a remote server, create a tunnel from the local computer and keep it open:

```bash
ssh -L 8501:localhost:8501 <user>@<server>
```

In the interface, select a task and date, draw a rectangle inside its supported region, and click **Analyze**. A feature whose API is not running reports a connection error without preventing the other configured features from working. When FastDiffSR output is missing, the backend queues generation for the full Arkansas 20x20 tile grid for that month rather than only the requested bounding box. This can require substantial GPU time and storage. Use **Refresh SR Status** to poll progress; the page does not refresh the job automatically.

## Call the APIs directly

Check service health:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/info
curl http://localhost:8002/health
curl http://localhost:8002/info
curl http://localhost:8003/health
curl http://localhost:8003/info
```

For Harvest, `/health` confirms only that the process is running. Check `/info` and verify that `ready` is true, `serving_mode` is `precomputed`, and the requested year appears in `available_years` before sending a prediction request.

Crop Identification request:

```bash
curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "year": 2025,
    "month": 6,
    "bbox": {
      "lon_min": -92.27,
      "lat_min": 33.05,
      "lon_max": -92.23,
      "lat_max": 33.10
    },
    "output": "png"
  }'
```

The API clamps `month` to the supported 6-11 range. Use `model_month` instead to select a specific supported model folder. The `output` value can be `png`, `png+npz`, or `npz`.

FastDiffSR request:

```bash
curl -X POST http://localhost:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "year": 2024,
    "month": 5,
    "bbox": {
      "lon_min": -92.27,
      "lat_min": 33.05,
      "lon_max": -92.23,
      "lat_max": 33.10
    }
  }'
```

When source data or a generated result is missing, the API returns HTTP 202 with a job ID and status URL.

Harvest request:

```bash
curl -X POST http://localhost:8003/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "year": 2024,
    "bbox": {
      "lon_min": -94.64,
      "lat_min": 32.92,
      "lon_max": -94.62,
      "lat_max": 32.94
    }
  }'
```

The response reports median harvest start and end dates, P10-P90 spatial tile variation, tile counts, and the stored model test MAE. P10-P90 is not a confidence interval. The API selects precomputed rows for whole source tiles that intersect the bounding box, so crop records outside the precise box but inside those tiles can influence the aggregate. The available year is 2024, and its supported region is the statewide Arkansas grid reported by `/info`.

## Supported regions

See the region table in [Phase 1: Data Download](DATA_DOWNLOAD.md#select-a-region). The active FastDiffSR checkpoint reconstructs Planet imagery from downsampled Planet imagery; it was not trained as a Sentinel-2-to-Sentinel-2 model.
