# Arkansas Crop Web Services

This repository provides a web interface for three Arkansas satellite-imagery workflows:

| Feature | What it does | API | Runtime data required |
| --- | --- | --- | --- |
| Crop Identification | Displays precomputed crop-classification GeoTIFFs | `ar_pred_api.py` (port 8001) | Precomputed crop predictions |
| Super Resolution | Applies 4x FastDiffSR super-resolution to Sentinel-2 imagery | `ar_fastdiffsr_api.py` (port 8002) | Sentinel-2 imagery |
| Crop Harvest Estimation | Estimates harvest start and end dates by crop | `ar_harvest_api.py` (port 8003) | Sentinel-2 imagery and prepared workbooks |

The browser interface is `app_AR_deploy.py`, a Streamlit application running on port 8501. You only need to start the APIs for the features you want to use.

> This README explains how to run the services with the included trained models and how to regenerate the Harvest API workbooks. Model training is outside its scope.

## Before you begin

- Run the commands from the repository root unless a step says otherwise.
- Replace values enclosed in angle brackets, such as `<earth-engine-project>` or `<DATA_ROOT>`, with values for your environment.
- Each API and the Streamlit application run as separate processes. Open one terminal for every process you start.
- Datasets and generated predictions are not stored in Git. You must provide the data required by each feature.
- FastDiffSR requires an NVIDIA GPU with working CUDA support.
- A Google Earth Engine project and authentication are only required when downloading new Sentinel-2 data.

## Setup and launch overview

1. Clone the repository and download the Git LFS model.
2. Create the Conda environment.
3. Configure the runtime data for the features you need.
4. Start the corresponding APIs.
5. Start Streamlit.
6. Use the application in a browser.

## Step 1: Clone the repository

If you already have a clone, enter its directory and start with `git lfs pull`.

```bash
git lfs install
git clone https://github.com/yukiikebe/AR_crop_identification.git
cd AR_crop_identification
git lfs pull
```

The FastDiffSR checkpoint is managed by Git LFS. Confirm that it is an approximately 91 MB binary file:

```bash
ls -lh fastdiffsr/checkpoints/I283712_E757_gen.pth
```

If the file is only a few KB, it is still an LFS pointer. Run `git lfs pull` again before continuing.

## Step 2: Create the Conda environment

Create and activate the deployment environment. The initial environment creation may take several minutes.

```bash
conda env create -f super_res310.yml
conda activate super_res310
```

If the environment already exists, update it from the YAML file:

```bash
conda env update -n super_res310 -f super_res310.yml --prune
conda activate super_res310
```

Check the main dependencies and CUDA availability:

```bash
python -c "import torch, rasterio, fastapi, streamlit; print('imports: OK'); print('CUDA:', torch.cuda.is_available())"
python -c "import streamlit_folium; print('streamlit-folium: OK')"
```

To use FastDiffSR, the first command must report `CUDA: True`. If it does not, run `nvidia-smi` and confirm that your GPU is visible. Do not switch the GPU workflow to CPU execution.

## Step 3: Configure runtime data

You do not need to enable every feature. Prepare only the data for the features you plan to use.

### 3-A: Crop Identification data

Place precomputed prediction GeoTIFFs in the following layout:

```text
<PRED_ROOT>/
└── AR_<year>_preds/
    └── <MM>mo/
        ├── 0_0.tif
        ├── 0_1.tif
        └── ...
```

For example:

```text
/local/data/predictions/AR_2025_preds/06mo/0_0.tif
```

Set the prediction root in the terminal that will run the Crop Identification API:

```bash
export DEEPSAT_AR_PRED_ROOT=/local/data/predictions
```

The available trained model folders are `06mo` through `11mo`. Requests for months 1-5 use the 6-month model, and a request for month 12 uses the 11-month model.

### 3-B: Super Resolution data

FastDiffSR reads `B2`, `B3`, `B4`, and `SCL` from the shared Sentinel-2 directory:

```text
<DATA_ROOT>/
└── <year>_AR/
    └── <grid id>/
        └── <YYYY-MM-DD>/
            ├── B2_<YYYY-MM-DD>.tif
            ├── B3_<YYYY-MM-DD>.tif
            ├── B4_<YYYY-MM-DD>.tif
            └── SCL_<YYYY-MM-DD>.tif
```

The default path on the project HPC is:

```text
/scrfs/storage/yikebe/home/AR_sentinel2/<year>_AR
```

No environment variable is required if you can read that path. For data stored elsewhere, set the template below and keep `{year}` literally in the value:

```bash
export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE='/local/data/sentinel2/{year}_AR'
```

The trained checkpoint and inference configuration are included in the repository:

```text
Configuration: fastdiffsr/config/sr_fastdiffsr_infer_x4_planet.json
Model:         fastdiffsr/checkpoints/I283712_E757_gen.pth
Default output: ./fastdiffsr_preds
```

### 3-C: Crop Harvest Estimation data

The Harvest API needs both source rasters and prepared workbooks:

```text
<DATASET_ROOT>/<year>_AR/<grid id>/cdl.tif
<OUTPUTS_ROOT>/<year>_AR/<grid id>/harvest_summary_all_crops.xlsx
```

If `cdl.tif` is absent, the API uses the first `<grid id>/*/B4_*.tif` it finds to determine the tile footprint.

The default project HPC paths are:

```text
Sentinel-2/CDL:    /scrfs/storage/yikebe/home/AR_sentinel2
Prepared workbooks: /scrfs/storage/yikebe/home/DeepSatModels_updated/outputs
Verified years:     2023, 2025
```

No environment variables are required if you can read those paths. For data stored elsewhere, set both roots in the terminal that will run the Harvest API:

```bash
export DEEPSAT_HARVEST_DATASET_ROOT=/local/data/sentinel2
export DEEPSAT_HARVEST_OUTPUTS_ROOT=/local/data/harvest/outputs
```

The model code and checkpoints for 41 crops are included in the repository. Normally, you do not need to change `DEEPSAT_HARVEST_CODE_ROOT` or `DEEPSAT_HARVEST_MODEL_ROOT`.

## Step 4: Start the APIs

Open a new terminal for each API you want to use. In every terminal, enter the repository and activate the Conda environment:

```bash
cd /path/to/AR_crop_identification
conda activate super_res310
```

On some HPC systems, use these settings before starting FastDiffSR or Harvest to avoid MKL and OpenMP conflicts:

```bash
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
```

### Terminal 1: Crop Identification API

Start this API only if you want to use Crop Identification. Replace the prediction root with the path prepared in Step 3.

```bash
conda activate super_res310
export DEEPSAT_AR_PRED_ROOT=/local/data/predictions
python -m uvicorn ar_pred_api:app --host 0.0.0.0 --port 8001
```

### Terminal 2: FastDiffSR API

```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# Set these only when you are not using the default HPC paths.
# export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE='/local/data/sentinel2/{year}_AR'
# export DEEPSAT_FASTDIFFSR_OUT_ROOT=/local/data/fastdiffsr_outputs

python -m uvicorn ar_fastdiffsr_api:app --host 0.0.0.0 --port 8002
```

GPU 0 is used by default. To use a different GPU, set its ID before starting the API:

```bash
export DEEPSAT_FASTDIFFSR_DEVICE=1
```

If Sentinel-2 data for a requested month is missing, the API creates an asynchronous job and can attempt to download the data from Earth Engine. To enable automatic downloads, authenticate with Earth Engine in advance and set the project before starting the API:

```bash
export DEEPSAT_FASTDIFFSR_EE_PROJECT=<earth-engine-project>
```

The following overrides are optional and are only needed when using custom outputs, prebuilt mosaics, Planet metadata overlays, or another compatible model:

```bash
export DEEPSAT_FASTDIFFSR_OUT_ROOT=/local/data/fastdiffsr_outputs
export DEEPSAT_FASTDIFFSR_MOSAIC_ROOT=/local/data/fastdiffsr_mosaics
export DEEPSAT_PS_SCENE_ROOT=/local/data/planet_metadata
export DEEPSAT_FASTDIFFSR_CONFIG=/path/to/compatible_config.json
export DEEPSAT_FASTDIFFSR_CHECKPOINT=/path/to/checkpoint_prefix
```

### Terminal 3: Harvest API

```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# Set these only when you are not using the default HPC paths.
# export DEEPSAT_HARVEST_DATASET_ROOT=/local/data/sentinel2
# export DEEPSAT_HARVEST_OUTPUTS_ROOT=/local/data/harvest/outputs

python -m uvicorn ar_harvest_api:app --host 0.0.0.0 --port 8003
```

The Harvest API uses CPU by default. To run it on a GPU, set the device before starting the API:

```bash
export DEEPSAT_HARVEST_DEVICE=cuda:0
```

## Step 5: Start the Streamlit application

Open one more terminal. This process is separate from the APIs.

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

If the browser is running on the same machine, open:

```text
http://localhost:8501
```

If Streamlit is running on an HPC or remote server, create an SSH tunnel from your local computer:

```bash
ssh -L 8501:localhost:8501 <user>@<HPC-hostname>
```

Keep the SSH connection open, then visit `http://localhost:8501` in your local browser.

## Step 6: Use the web interface

1. Select a feature from `Task` in the left sidebar.
2. Select a year. Crop Identification and Super Resolution also require a month.
3. Draw a rectangle inside the highlighted supported region on the map.
4. Click `Analyze`.
5. Wait for the result to appear.

FastDiffSR runs as an asynchronous job when a requested result has not already been generated. Keep the page open to see its progress. A tab whose API is not running will show a connection error, but the other configured features remain usable.

## Downloading new Sentinel-2 data

Skip this section if you can use the existing project HPC data. Downloads require time and storage, so test one month and a small region before attempting a statewide or whole-year download.

### 1. Download one month from Earth Engine

Use `--auth` only for the first interactive authentication. Omit it on later runs.

```bash
conda activate super_res310
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python data_download/download_sentinel2.py \
  --project <earth-engine-project> \
  --auth \
  --year 2026 \
  --month 7 \
  --band-preset all \
  --data-dir /local/data/sentinel2/2026_AR \
  --roi-bbox=-92.30,34.70,-92.20,34.80
```

The `all` band preset creates one `<year>_AR` directory that both FastDiffSR and Harvest can use. Omit `--roi-bbox` to download the default Arkansas deployment region. To download a complete year, replace `--month 7` with `--whole-year`.

Running the command again with the same `--data-dir` and another month adds or verifies that month without replacing existing files. Do not mix grids created with different `--roi-bbox` values in the same `<year>_AR` directory.

See [data_download/README.md](data_download/README.md) for all download options.

### 2. Download CDL for Harvest

The CDL script requires `Rscript` and the R packages `terra`, `CropScapeR`, and `sf`. Install them from R:

```r
install.packages(c("terra", "CropScapeR", "sf"))
```

Start with a one-tile test:

```bash
Rscript data_download/download_cdl.r \
  --base-dir /local/data/sentinel2/2023_AR \
  --year 2023 \
  --limit 1
```

If the test succeeds, remove `--limit 1` to process all tiles. The script writes `cdl.tif` into each grid directory.

These scripts create the Sentinel-2 and CDL inputs. Use the next section to generate `harvest_summary_all_crops.xlsx`.

## Generating Harvest API workbooks

The reusable input-generation pipeline is included under `harvest_estimation/create_doy_prediction_input/`. It converts one year of Sentinel-2 and CDL tiles into the per-tile workbooks read by the Harvest API.

The input year must have this layout:

```text
<DATASET_ROOT>/<year>_AR/
└── <grid id>/
    ├── cdl.tif
    └── <YYYY-MM-DD>/
        ├── B2_<YYYY-MM-DD>.tif
        ├── B4_<YYYY-MM-DD>.tif
        ├── B8_<YYYY-MM-DD>.tif
        ├── B11_<YYYY-MM-DD>.tif
        └── SCL_<YYYY-MM-DD>.tif
```

`B12` can be used instead of `B11`, and `SCL` is optional but recommended for filtering clouds and other invalid pixels. The earlier `--band-preset all` download command provides all required bands.

Start with one tile. Replace `17_10` with a grid ID that exists in your dataset:

```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

python -m harvest_estimation.create_doy_prediction_input.main \
  --dataset-root /local/data/sentinel2/2023_AR \
  --cdl-yaml harvest_estimation/configs/cdl.yaml \
  --gt-windows-yaml harvest_estimation/configs/gt_windows.yaml \
  --seeding-config-yaml harvest_estimation/configs/seeding_config.yaml \
  --output-root /local/data/harvest/outputs/2023_AR \
  --all-crops \
  --no-farm \
  --no-index-images \
  --tiles 17_10
```

The inference-only options `--no-farm` and `--no-index-images` avoid generating farm-level and per-date image artifacts that the Harvest API does not use. A successful run creates:

```text
/local/data/harvest/outputs/2023_AR/17_10/harvest_summary_all_crops.xlsx
```

After the one-tile test succeeds, remove `--tiles 17_10` to process every grid in the year. A full-year run is long-running, so run it in `tmux` or through the HPC scheduler. The pipeline resumes from per-tile checkpoints by default.

Finally, point the Harvest API at the parent output directory:

```bash
export DEEPSAT_HARVEST_OUTPUTS_ROOT=/local/data/harvest/outputs
```

## Calling the APIs directly

### FastDiffSR request

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

If the data or generated result is missing, the API returns HTTP 202 with job information. Check the response's `status_url`, or call `/jobs/<job_id>`, to monitor progress.

### Harvest request

```bash
curl -X POST http://localhost:8003/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "year": 2023,
    "bbox": {
      "lon_min": -92.27,
      "lat_min": 32.91,
      "lon_max": -92.17,
      "lat_max": 32.99
    }
  }'
```

The response reports median harvest start and end dates, P10-P90 spatial variation, tile counts, and the stored 2023 test MAE. The model was trained on 2022 data and evaluated on 2023 data. Treat results for other prepared years as extrapolation.

## Supported regions

Coordinates use WGS84 in `min_lon,min_lat,max_lon,max_lat` order.

| Region | Bounds | Purpose and source |
| --- | --- | --- |
| FastDiffSR training source | `-92.3017581016,34.7471457771,-89.8853046962,36.5201148678` | Smallest bounding rectangle containing the 500 PlanetScope source scenes; it does not imply complete coverage inside the rectangle |
| Harvest supported region | `-92.2800605553,32.9038172827,-89.8804944923,34.7531509549` | Fixed API/UI region derived from `right_bottom_ar.json` and its source GeoTIFF |
| Smallest rectangle containing both | `-92.3017581016,32.9038172827,-89.8804944923,36.5201148678` | Union of the two preceding rectangular extents |
| Default Arkansas download region | `-94.7610,32.8376,-89.5522,36.6652` | Deployment region used when `--roi-bbox` is omitted |

The active FastDiffSR checkpoint was trained to reconstruct Planet imagery from downsampled Planet imagery. It was not trained as a Sentinel-2-to-Sentinel-2 model.

## Repository layout

```text
.
├── app_AR_deploy.py              # Streamlit web interface
├── ar_pred_api.py                # Crop prediction API
├── ar_fastdiffsr_api.py          # FastDiffSR API and asynchronous jobs
├── ar_harvest_api.py             # Harvest-date estimation API
├── ar_deploy.py                  # Prediction and FastDiffSR utility CLI
├── data_download/
│   ├── download_sentinel2.py     # Earth Engine Sentinel-2 downloader
│   ├── download_cdl.r            # USDA CDL download and grid alignment
│   └── README.md                 # Detailed data-download instructions
├── fastdiffsr/
│   ├── config/                   # Active 4x inference configuration
│   ├── model/                    # FastDiffSR inference modules
│   └── checkpoints/              # Trained model managed by Git LFS
├── harvest_estimation/
│   ├── configs/                  # CDL classes and harvest/seeding windows
│   ├── create_doy_prediction_input/
│   │   └── ...                   # Sentinel-2/CDL to Harvest workbook pipeline
│   ├── doy_prediction/           # Harvest-date inference modules
│   └── models/                   # Models and metrics for 41 crops
└── super_res310.yml              # Deployment Conda environment
```
