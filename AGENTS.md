# Repository Guidelines

## Project Structure & Module Organization
- `train_and_eval/`: training/validation/inference entrypoints.
- `models/`: model definitions (TSViT, UNet3D, legacy CropTypeMapping).
- `configs/`: experiment YAMLs (Arkansas configs in `configs/Arkansas/`).
- `data/`: dataset utilities + download scripts (Arkansas in `data/Arkansas/`).
- `ar_deploy.py`: monthly orchestrator (download → predict to GeoTIFF).
- `ar_pred_api.py`: FastAPI that serves **precomputed** predictions and supports bbox queries.
- `app_AR_deploy.py`: Streamlit client UI (year/month/bbox only).

## Build, Test, and Development Commands
This project is Python/Conda-first.
- Create env: `conda env create -f deepsatmodels_env.yml && conda activate deepsatmodels`
- Train: `python train_and_eval/segmentation_training_transf.py --config configs/Arkansas/TSViT_AR23_11mo_focal.yaml`
- Download one month: `python data_download/download_sentinel2.py --project <gcp-project> --year 2025 --month 6 --band-preset all --data-dir <DATA_ROOT>/2025_AR`
- Run API: `uvicorn ar_pred_api:app --host 0.0.0.0 --port 8001`
- Run UI: `streamlit run app_AR_deploy.py`

Compatibility note: models that pull `torchfcn -> fcn -> chainer` require `numpy<2` (NumPy 2.x removed `np.sctypes`).

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` functions/variables, `CapWords` classes.
- Prefer repo-relative paths and env vars over machine-specific absolute paths (e.g., `DEEPSAT_AR_PRED_ROOT`, `DEEPSAT_AR_PRED_API_URL`).
- Keep YAML structure consistent (`MODEL`, `SOLVER`, `DATASETS`, `CHECKPOINT`) to reduce config drift.

## Testing Guidelines
There is no automated test suite. Before merging, run the affected entrypoint end-to-end on a small sample and confirm expected artifacts (logs, GeoTIFFs, metrics).

## Commit & Pull Request Guidelines
- Prefer `type(scope): summary` (e.g., `deploy(ar): add bbox query`).
- PRs should include: intent, config path(s), and exact reproduction commands.
- Do not commit datasets, predictions, `wandb/`, or model checkpoints (`models/saved_models/` is ignored). Avoid committing tokens/credentials; set `DEEPSAT_MAPBOX_TOKEN` locally if needed.
