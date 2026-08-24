# Arkansas Crop Web Services

This repository supports three Arkansas satellite-imagery workflows:

| Feature | What it does | API | Runtime data required |
| --- | --- | --- | --- |
| Crop Identification | Displays precomputed crop-classification GeoTIFFs | `ar_pred_api.py` (port 8001) | Precomputed crop predictions |
| Super Resolution | Applies 4x FastDiffSR super-resolution to Sentinel-2 imagery | `ar_fastdiffsr_api.py` (port 8002) | Sentinel-2 imagery |
| Crop Harvest Estimation | Serves precomputed harvest start and end dates by crop | `ar_harvest_api.py` (port 8003) | Precomputed tile-level harvest predictions and source tile bounds |

The browser interface is `app_AR_deploy.py`, a Streamlit application that connects to any of these APIs.

Harvest inference is a separate batch step. The current Harvest deployment provides 2024 predictions for the statewide Arkansas grid only. The batch artifact stores predictions together with the source grid's tile bounds, and API requests only aggregate those saved rows; they do not rerun the model.

## Workflow

| Phase | Guide | Contents |
| --- | --- | --- |
| 1. Data download | [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md) | Download Sentinel-2 imagery and USDA Cropland Data Layer rasters. |
| 2. Training | [TRAINING.md](TRAINING.md) | Prepare the Arkansas crop-classification dataset and train the monthly TSViT models. |
| 3. Inference | [INFERENCE.md](INFERENCE.md) | Generate predictions, run the APIs, and launch the Streamlit interface. |

Training is optional when using the included FastDiffSR and Harvest models or existing crop prediction GeoTIFFs. Crop-model checkpoints are not included in the deployment-focused checkout. Data download is optional when the required runtime data is already available.

## Quick start for inference

Clone the repository and retrieve the Git LFS checkpoint:

```bash
git lfs install
git clone https://github.com/yukiikebe/AR_crop_identification.git
cd AR_crop_identification
git lfs pull
```

Create the deployment environment:

```bash
conda env create -f super_res310.yml
conda activate super_res310
```

Then follow [Inference](INFERENCE.md) to configure data paths and start only the services you need.

## Repository layout

```text
.
├── README.md                     # Project overview and workflow index
├── DATA_DOWNLOAD.md              # Phase 1: source-data acquisition
├── TRAINING.md                   # Phase 2: crop-model training
├── INFERENCE.md                  # Phase 3: batch and web inference
├── app_AR_deploy.py              # Streamlit web interface
├── ar_pred_api.py                # Crop prediction API
├── ar_fastdiffsr_api.py          # FastDiffSR API and asynchronous jobs
├── ar_harvest_api.py             # Harvest-date estimation API
├── precompute_harvest_predictions.py # Batch harvest prediction generator
├── ar_deploy.py                  # Prediction and FastDiffSR utility CLI
├── data_download/                # Sentinel-2 and CDL download utilities
├── fastdiffsr/                   # FastDiffSR inference code and checkpoint
├── harvest_estimation/           # Harvest preprocessing, inference, and models
└── super_res310.yml              # Deployment Conda environment
```

Datasets, generated predictions, logs, credentials, and experiment outputs must remain outside Git. Set tokens such as `DEEPSAT_MAPBOX_TOKEN` locally when needed.

FastDiffSR source is derived from [Meng-333/FastDiffSR](https://github.com/Meng-333/FastDiffSR).
