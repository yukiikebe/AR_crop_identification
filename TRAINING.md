# Phase 2: Training

This phase trains the Arkansas TSViT crop-identification models used to produce monthly crop-classification GeoTIFFs. Training is optional when using existing checkpoints or precomputed predictions.

The deployment-focused checkout contains the inference services and trained FastDiffSR and Harvest assets. The crop-model commands below require the legacy training tree, including `configs/`, `data/`, `models/`, `train_and_eval/`, `utils/`, and `deepsatmodels_env.yml`.

## Create the training environment

```bash
conda env create -f deepsatmodels_env.yml
conda activate deepsatmodels_crop_id
```

The legacy `torchfcn -> fcn -> chainer` dependency chain requires `numpy<2` because NumPy 2 removed `np.sctypes`.

The current environment manifest does not declare all modules imported by the training path, including PyTorch, Weights & Biases, and Blosc2. Verify those imports in the project environment and install versions compatible with the host CUDA runtime before starting:

```bash
python -c "import torch, torchvision, wandb, blosc2; print('training imports: OK')"
```

Training and validation are GPU-only workflows. Confirm that the intended CUDA device is visible before starting:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Prepare the Arkansas dataset

The downloader produces georeferenced Sentinel-2 and CDL rasters. The TSViT dataloader instead expects 24x24 Blosc2 frames and a CSV that assigns each tile to `train` or `val`:

```text
<TRAIN_DATA_ROOT>/
└── <grid id>/
    ├── img/
    │   └── <tile id>_img.b2frame
    ├── label_remap/
    │   └── <tile id>_label.b2frame
    └── doy/
        └── <tile id>_doy.b2frame
```

Configure the `AR23` entry in `data/datasets.yaml` with the processed dataset root and split CSV:

```yaml
AR23:
  basedir: "/path/to/AR_2023_bframe2"
  paths_train: "data/Arkansas/tiles_train_val_AR.csv"
  paths_eval: "data/Arkansas/tiles_train_val_AR.csv"
  paths_test: "data/Arkansas/tiles_train_val_AR.csv"
```

The CSV must contain `meta_patch`, `tile_id`, and `split` columns. The dataloader resolves each row against the `img`, `label_remap`, and `doy` directories shown above.

## Configure a training run

Arkansas experiment files live in `configs/Arkansas/`. Review these sections before starting:

- `MODEL`: TSViT architecture and input dimensions.
- `DATASETS`: dataset key, batch size, workers, and maximum day of year.
- `SOLVER`: epochs, learning rate, focal-loss settings, and scheduler.
- `CHECKPOINT`: optional source checkpoint, save directory, and evaluation/save intervals.
- `DEVICE`: configured device IDs; the command-line `--device` value selects the GPUs used by the entrypoint.

The seasonal configurations cover June through November:

| Available data | Configuration | Checkpoint directory |
| --- | --- | --- |
| January-June | `configs/Arkansas/TSViT_AR23_06mo_focal.yaml` | `models/saved_models/AR23_focal_06mo/` |
| January-July | `configs/Arkansas/TSViT_AR23_07mo_focal.yaml` | `models/saved_models/AR23_focal_07mo/` |
| January-August | `configs/Arkansas/TSViT_AR23_08mo_focal.yaml` | `models/saved_models/AR23_focal_08mo/` |
| January-September | `configs/Arkansas/TSViT_AR23_09mo_focal.yaml` | `models/saved_models/AR23_focal_09mo/` |
| January-October | `configs/Arkansas/TSViT_AR23_10mo_focal.yaml` | `models/saved_models/AR23_focal_10mo/` |
| January-November | `configs/Arkansas/TSViT_AR23_11mo_focal.yaml` | `models/saved_models/AR23_focal_11mo/` |

Class mappings and band/sample definitions are in `configs/Arkansas/arkansas_data.yaml`. Keep the configured mappings consistent with the remapped labels in the processed dataset.

The current trainer overrides `MODEL.num_classes` with `26`, so the class count declared in the seasonal YAML files is not authoritative. Resolve that mismatch in the code and label mapping before using another number of classes. The trainer also uses AdamW with `SOLVER.lr_base` and `SOLVER.weight_decay`; the scheduler-related YAML fields are not currently consumed by this entrypoint.

## Train

The training entrypoint logs to Weights & Biases and prompts for authentication if no local credentials are available.

Train one seasonal model on GPU 0:

```bash
python train_and_eval/segmentation_training_transf.py \
  --config configs/Arkansas/TSViT_AR23_11mo_focal.yaml \
  --device 0
```

Select multiple GPUs with a comma-separated list such as `--device 0,1`. The YAML `DEVICE.device_id` value is ignored by this entrypoint. Run the other monthly configurations separately to produce the full 6-11 month checkpoint set.

The entrypoint saves periodic checkpoints and the highest macro-IoU checkpoint as `best.pth` under the configuration's `CHECKPOINT.save_path`. It also copies the effective configuration into that directory. Checkpoints contain the model state only; loading one does not restore the optimizer, epoch, step, or previous best-IoU state.

## Validate

Set `CHECKPOINT.load_from_checkpoint` in `configs/Arkansas/TSViT_AR23_infer.yaml` to the trained `best.pth`, then run:

```bash
python train_and_eval/validation_AR24.py \
  --config configs/Arkansas/TSViT_AR23_infer.yaml
```

Padding introduced during preprocessing can affect metrics near tile edges.

Continue to [Inference](INFERENCE.md) to turn checkpoints into GeoTIFF predictions or serve existing results through the APIs.
