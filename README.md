# Arkansas Crop Web Services

Deployment-only repository for the Arkansas map UI, precomputed crop maps,
FastDiffSR super-resolution, and crop harvest-date estimation. Training code,
dataset-building experiments, generated manifests, predictions, and unused model
variants are intentionally excluded.

## Repository layout

```text
.
├── app_AR_deploy.py              # Streamlit map UI
├── ar_pred_api.py                # Precomputed crop-map API (port 8001)
├── ar_fastdiffsr_api.py          # FastDiffSR API and async jobs (port 8002)
├── ar_harvest_api.py             # Harvest estimation API (port 8003)
├── ar_deploy.py                  # Sentinel-2 download and SR utility CLI
├── data_download/
│   ├── download_sentinel2.py     # Earth Engine Sentinel-2 downloader
│   ├── download_cdl.r            # USDA CDL download and grid alignment
│   └── README.md                 # Data acquisition instructions
├── fastdiffsr/
│   ├── config/                   # Active x4 inference configuration
│   ├── model/                    # FastDiffSR inference modules only
│   └── checkpoints/              # Active generator checkpoint (Git LFS)
├── harvest_estimation/
│   ├── doy_prediction/           # Hybrid inference/data modules only
│   └── models/                   # 41 crop checkpoints and test metrics
└── super_res310.yml              # Deployment environment
```

The active model assets are:

| Service | Model | Repository path |
| --- | --- | --- |
| FastDiffSR | x4 generator, `I283712_E757` | `fastdiffsr/checkpoints/I283712_E757_gen.pth` |
| Harvest | hybrid CNN/RNN, `1year/all_indices`, 41 crops | `harvest_estimation/models/1year/all_indices/` |

The FastDiffSR checkpoint is stored with Git LFS. Install Git LFS before cloning,
or run `git lfs pull` after cloning, and verify that the checkpoint is a binary
file rather than a small LFS pointer.

## Installation

```bash
conda env create -f super_res310.yml
conda activate super_res310
```

If the environment already exists:

```bash
conda env update -n super_res310 -f super_res310.yml --prune
```

On hosts where MKL and OpenMP conflict, set these before starting an API:

```bash
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
```

## Runtime data

Datasets and generated predictions are not committed. Configure their locations
with environment variables.

FastDiffSR expects Sentinel-2 data under:

```text
<raw root>/AR_<year>_raw/<grid id>/<YYYY-MM-DD>/{B2,B3,B4,SCL}_<date>.tif
```

Harvest estimation expects both prepared workbooks and source rasters:

```text
<outputs root>/<year>_AR/<grid id>/harvest_summary_all_crops.xlsx
<dataset root>/<year>_AR/<grid id>/cdl.tif
```

If `cdl.tif` is absent, the harvest API uses the first
`<grid id>/*/B4_*.tif` for the tile footprint.

## Download Arkansas input data

The repository includes standalone Sentinel-2 and CDL acquisition scripts.
Generated data remains ignored by Git.

Use the entrypoint that matches the target workflow:

| Target | Command |
| --- | --- |
| FastDiffSR input for a new month | `python ar_deploy.py download ...` |
| Harvest Sentinel-2 input for a whole year | `python data_download/download_sentinel2.py ... --data-dir <year>_AR` |
| Harvest CDL after Sentinel-2 | `Rscript data_download/download_cdl.r ...` |

### FastDiffSR input

Download one FastDiffSR-ready Sentinel-2 month. The first run needs interactive
Earth Engine authentication through `--auth`; later runs can omit it.

```bash
conda activate super_res310
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python ar_deploy.py download \
  --project <earth-engine-project> \
  --auth \
  --year 2024 \
  --month 5 \
  --band-preset rgb_scl \
  --data-root /local/data/sentinel2
```

This creates
`/local/data/sentinel2/AR_2024_raw/<grid>/<YYYY-MM-DD>/`.

### Harvest source rasters and CDL

Harvest uses the `<year>_AR` directory convention. Download a whole year of
Sentinel-2 directly into that layout:

```bash
python data_download/download_sentinel2.py \
  --project <earth-engine-project> \
  --auth \
  --year 2023 \
  --whole-year \
  --band-preset all \
  --data-dir /local/data/sentinel2/2023_AR
```

Install an `Rscript` executable and the R packages `terra`, `CropScapeR`, and
`sf`. Then download USDA CDL and align it to each Sentinel tile:

```bash
Rscript data_download/download_cdl.r \
  --base-dir /local/data/sentinel2/2023_AR \
  --year 2023
```

This writes `/local/data/sentinel2/2023_AR/<grid>/cdl.tif`. Use `--limit 1`
for a one-tile test before processing every tile.

For a non-HPC deployment, point the APIs at the downloaded data:

```bash
export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE=/local/data/sentinel2/AR_{year}_raw
export DEEPSAT_FASTDIFFSR_DOWNLOAD_DATA_ROOT=/local/data/sentinel2
export DEEPSAT_HARVEST_DATASET_ROOT=/local/data/sentinel2
export DEEPSAT_HARVEST_OUTPUTS_ROOT=/local/data/harvest/outputs
```

The scripts above generate Sentinel-2 and CDL inputs. They do not generate the
Harvest workbooks at
`<outputs root>/<year>_AR/<grid>/harvest_summary_all_crops.xlsx`; provide those
separately.

See [data_download/README.md](data_download/README.md) for the full workflow,
including direct `--data-dir` usage and required R packages.

## Start the services

Open one terminal per process. Relative model paths work from any launch
directory because the APIs resolve them against the repository root.

### 1. Precomputed crop maps (optional)

```bash
conda activate super_res310
export DEEPSAT_AR_PRED_ROOT=/path/to/prediction/datasets
python -m uvicorn ar_pred_api:app --host 0.0.0.0 --port 8001
```

Expected prediction layout:

```text
<DEEPSAT_AR_PRED_ROOT>/AR_<year>_preds/<MM>mo/<grid id>.tif
```

### 2. FastDiffSR

```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# Optional; defaults to ./fastdiffsr_preds in the clone.
# export DEEPSAT_FASTDIFFSR_OUT_ROOT=/path/to/fastdiffsr_outputs
# Optional prebuilt statewide mosaics:
# export DEEPSAT_FASTDIFFSR_MOSAIC_ROOT=/path/to/fastdiffsr_mosaics
# Optional Planet metadata used only for the reference overlay:
# export DEEPSAT_PS_SCENE_ROOT=/path/to/planet_metadata

python -m uvicorn ar_fastdiffsr_api:app --host 0.0.0.0 --port 8002
```

The repository defaults already select:

```text
Sentinel-2: /scrfs/storage/yikebe/home/AR_sentinel_align_with_Planet/AR_<year>_raw
data years:  2020-2024
config:     fastdiffsr/config/sr_fastdiffsr_infer_x4_planet.json
checkpoint: fastdiffsr/checkpoints/I283712_E757
```

Override them only when intentionally testing another compatible model:

```bash
export DEEPSAT_FASTDIFFSR_RAW_ROOT_TEMPLATE=/path/to/AR_{year}_raw
export DEEPSAT_FASTDIFFSR_CONFIG=/path/to/config.json
export DEEPSAT_FASTDIFFSR_CHECKPOINT=/path/to/checkpoint_prefix
export DEEPSAT_FASTDIFFSR_DEVICE=0
```

When requested SR data is missing, the API creates an async job and can download
the required Sentinel-2 month through Earth Engine. Configure that path with
`DEEPSAT_FASTDIFFSR_DOWNLOAD_DATA_ROOT` and the project with
`DEEPSAT_FASTDIFFSR_EE_PROJECT`.

### 3. Harvest estimation

```bash
conda activate super_res310
unset LD_LIBRARY_PATH
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# Optional GPU; CPU is the default and is sufficient for small bbox requests.
# export DEEPSAT_HARVEST_DEVICE=cuda:0

python -m uvicorn ar_harvest_api:app --host 0.0.0.0 --port 8003
```

On this HPC, the default inputs are:

```text
prepared outputs: /scrfs/storage/yikebe/home/DeepSatModels_updated/outputs
source rasters:   /scrfs/storage/yikebe/home/AR_sentinel2
verified years:   2023, 2025
```

These absolute paths let another user run a clone without copying the datasets,
provided that user has read/traverse access to the `yikebe` storage area. The
bundled model code and checkpoints are also used by default. Non-HPC deployments
can override the paths through `DEEPSAT_HARVEST_OUTPUTS_ROOT`,
`DEEPSAT_HARVEST_DATASET_ROOT`, `DEEPSAT_HARVEST_CODE_ROOT`, and
`DEEPSAT_HARVEST_MODEL_ROOT`.

### 4. Streamlit UI

```bash
conda activate super_res310
export DEEPSAT_AR_PRED_API_URL=http://localhost:8001
export DEEPSAT_FASTDIFFSR_API_URL=http://localhost:8002
export DEEPSAT_HARVEST_API_URL=http://localhost:8003
streamlit run app_AR_deploy.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType none
```

Crop Identification remains available when port 8001 is running. Super Resolution
and Crop Harvest Estimation use ports 8002 and 8003 respectively. Each task shows
its own supported map region. Open `http://<HPC-hostname>:8501`, or forward the
port from your local machine when direct compute-node access is unavailable:

```bash
ssh -L 8501:localhost:8501 <user>@<HPC-hostname>
```

## API checks

```bash
curl http://localhost:8002/health
curl http://localhost:8002/info
curl http://localhost:8003/health
curl http://localhost:8003/info
```

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

Harvest request:

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

The harvest response reports median start/end dates, P10-P90 spatial variation,
tile counts, and archived 2023 test MAE. The model was trained on 2022 and evaluated
on 2023; other prepared years should be treated as extrapolation.

## Generated files and credentials

Do not commit Sentinel-2/Planet imagery, prediction GeoTIFFs, job state, experiment
logs, tokens, or Earth Engine credentials. Set `DEEPSAT_MAPBOX_TOKEN` locally when
Mapbox tiles are needed; otherwise the UI uses OpenStreetMap.

FastDiffSR source is derived from [Meng-333/FastDiffSR](https://github.com/Meng-333/FastDiffSR).
