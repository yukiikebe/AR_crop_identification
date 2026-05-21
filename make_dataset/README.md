## make patches

python make_patches.py extract-tiff-patches --input-dir /home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene/ --output-dir /home/yikebe/research/FastDiffSR/FastDiffSR/tmp --include-name "SR_8b_clip" --exclude-name "udm2_clip" --skip-sea --hr-only --workers 16 --band-mode rgb


## use prepare_data_mfe.py

python prepare_data_mfe_dm.py --path /home/yikebe/research/FastDiffSR/planet_20240813_165220_13_24/HR --out ../dataset/test_planet

## use prepare_data_infer.py (input LR dir and make dataset for inference)

python data/prepare_data_mfe_infer.py -p /home/yikebe/research/FastDiffSR/planet_20240813_165220_13_24_64by64 -o ../dataset/planet_20240813_165220_13_24_64by64 --size 64,256 --make_hr_placeholder


## merge patches
python merge_patches.py merge-tiff --input-dir ... --output-path ... --reference-image ... --patch-size 256 --overlap 0.5


python merge_patches.py merge-png --input-dir /home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_infer_from_planet_to_maryland_x4_0813_260402_141258/results --output-path /home/yikebe/research/AR_Crop_Identification/super_resolution_result/0813.png --reference-image /home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene/20240813_165220_13_24f3_3B_AnalyticMS_SR_8b_clip.tif --patch-size 256 --overlap 0.5 --scale-multiplier 4

python merge_patches.py merge-png --input-dir /home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_infer_from_planet_to_maryland_x4_latest_0813_260410_133611/results --output-path /home/yikebe/research/AR_Crop_Identification/super_resolution_result/0813.png --reference-image /home/yikebe/planet_20240813_165220_13_24f3/20240813_165220_13_24f3_3B_AnalyticMS_SR_8b_clip.tif --patch-size 256 --overlap 0.5 --scale-multiplier 4

Normal flow:
- New inference results keep the original patch basename, so `merge-png` / `merge-tiff` will automatically select files that match the `--reference-image` stem when mixed folders contain multiple scenes.
- `--skip-first` and `--max-patches` are only recovery options for older result folders that were saved without source-aware patch names.
- If needed, `--file-substring` can explicitly choose one patch set by filename token.
- For x4 super-resolution outputs produced from `64x64 -> 256x256` inference datasets, pass `--scale-multiplier 4` so the merge canvas matches the SR output size rather than the LR reference size.


## Model 1: Sentinel-2 -> Planet dataset flow

This flow builds the Model 1 dataset for `Sentinel-2 RGB 16x16 -> Planet RGB 64x64`.
The planner now groups nearby Planet scenes that share the same Sentinel date window, so we avoid downloading the same Sentinel coverage thousands of times.
The dataset builder can read either the new grouped-request layout or the older one-scene-per-request layout.

### 1. Plan the initial pm5 downloads

```bash
python make_dataset/plan_planet_sentinel_downloads.py \
  --planet-root /home/thanyu/planet_data/analytic_8b_sr_udm2 \
  --windows-tsv make_dataset/sentinel_nearby_windows_pm5d.tsv \
  --project satelite-430703 \
  --raw-out-root make_dataset/raw_sentinel \
  --stage pm5
```

Outputs:
- `make_dataset/planet_scene_manifest.tsv`
- `make_dataset/sentinel_download_plan_pm5.tsv`
- `make_dataset/run_sentinel_downloads_pm5.sh`

Notes:
- default `--request-mode grouped` merges nearby Planet scenes that share the same `start_day/end_day`
- grouped requests are split again if the merged bbox grows beyond the planner limits
- if you ever need the old behavior, add `--request-mode scene`

### 2. Run the Sentinel downloads

```bash
bash make_dataset/run_sentinel_downloads_pm5.sh
```

Each request uses:
- merged bbox from one or more nearby Planet images that share the same time window
- `python ar_deploy.py download ...`
- `--band-preset rgb_scl` so Model 1 downloads only `B2/B3/B4/SCL`
- `--cloud-thresh 100`
- `raw_sentinel/<stage>/<request_id>/AR_<year>_raw/...` as the on-disk layout

### 3. Build the FastDiffSR dataset and audit the pair quality

```bash
python make_dataset/build_fastdiffsr_sentinel_planet_dataset.py \
  --planet-root /home/thanyu/planet_data/analytic_8b_sr_udm2 \
  --sentinel-root make_dataset/raw_sentinel \
  --train-out FastDiffSR/FastDiffSR/dataset/train_sentinel_planet_16_64 \
  --val-out FastDiffSR/FastDiffSR/dataset/val_sentinel_planet_16_64 \
  --sentinel-patch-size 16 \
  --planet-patch-size 64 \
  --scale 4 \
  --max-scl-cloud-ratio 0.10
```

Outputs:
- `FastDiffSR/FastDiffSR/dataset/train_sentinel_planet_16_64`
- `FastDiffSR/FastDiffSR/dataset/val_sentinel_planet_16_64`
- `FastDiffSR/FastDiffSR/dataset/pairs_manifest.jsonl`
- `FastDiffSR/FastDiffSR/dataset/missing_pairs.tsv`

The builder will auto-discover `make_dataset/sentinel_download_plan_*.tsv` so it can map grouped Sentinel requests back to the Planet scenes they cover.
If needed, you can also pass one or more explicit planner TSVs with repeated `--download-plan-tsv ...`.
For Model 1 this is enough because the LR image uses Sentinel RGB (`B4/B3/B2`) and cloud screening uses `SCL`.

Directory layout follows the existing FastDiffSR `img` loader:
- `lr_16`
- `sr_16_64`
- `hr_64`

### 4. Retry only the missing scenes

If `missing_pairs.tsv` still has rows, widen the search only for those scenes.

Retry with `pm10`:

```bash
python make_dataset/plan_planet_sentinel_downloads.py \
  --planet-root /home/thanyu/planet_data/analytic_8b_sr_udm2 \
  --windows-tsv make_dataset/sentinel_nearby_windows_pm5d.tsv \
  --project satelite-430703 \
  --raw-out-root make_dataset/raw_sentinel \
  --retry-from FastDiffSR/FastDiffSR/dataset/missing_pairs.tsv \
  --stage pm10 \
  --plan-tsv make_dataset/sentinel_download_plan_pm10.tsv \
  --commands-sh make_dataset/run_sentinel_downloads_pm10.sh

bash make_dataset/run_sentinel_downloads_pm10.sh
```

Retry with full month:

```bash
python make_dataset/plan_planet_sentinel_downloads.py \
  --planet-root /home/thanyu/planet_data/analytic_8b_sr_udm2 \
  --windows-tsv make_dataset/sentinel_nearby_windows_pm5d.tsv \
  --project satelite-430703 \
  --raw-out-root make_dataset/raw_sentinel \
  --retry-from FastDiffSR/FastDiffSR/dataset/missing_pairs.tsv \
  --stage month \
  --plan-tsv make_dataset/sentinel_download_plan_month.tsv \
  --commands-sh make_dataset/run_sentinel_downloads_month.sh

bash make_dataset/run_sentinel_downloads_month.sh
```

Then rerun the dataset builder with a fresh output directory.

### 5. Train FastDiffSR on the new dataset

```bash
python FastDiffSR/FastDiffSR/sr_mfe.py \
  -c FastDiffSR/FastDiffSR/config/sr_fastdiffsr_train_16_64_sentinel_planet.json
```

## current command to run
cd /home/yikebe/research/AR_Crop_Identification && sed -n '/^# pm5__20200705_20200723__0013/,$p' make_dataset/run_sentinel_downloads_pm5.sh | bash

## Maryland Sentinel -> NOAA orthophoto FastDiffSR flow

This flow builds a Maryland training set where:
- LR is made from Maryland Sentinel-2 64x64 patches.
- A trained Sentinel -> Planet FastDiffSR checkpoint first converts those LR patches to Planet-like 256x256 reconstructions.
- The 256x256 reconstructions are cropped back into 64x64 LR patches and paired with NOAA/Lexington Park orthophoto 256x256 HR patches.
- The final output follows the FastDiffSR `img` dataset layout and can be used with `FastDiffSR/FastDiffSR/sr_mfe.py`.

Inputs used:
- Sentinel roots:
  - `/home/yikebe/md_sentinel2/MD_2018_raw`
  - `/home/yikebe/md_sentinel2_2018-2019/MD_2018_raw`
- Sentinel roots available but too broad for the first Maryland/NOAA run:
  - `/home/yikebe/md_sentinel2/MD_2023_raw`
  - `/home/yikebe/md_sentinel2/MD_2024_raw`
- Sentinel -> Planet checkpoint prefix:
  - `/home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_train_64_256_Sentinel_Planet_clean_v5_260428_223636/checkpoint/I457600_E931`
  - FastDiffSR loads this prefix as `I457600_E931_gen.pth` and `I457600_E931_opt.pth`.
- NOAA/Lexington HR roots:
  - `/home/yikebe/NOAA_NGS/LexingtonPark_RGBN_2018_10140`
  - `/home/yikebe/NOAA_NGS/NOAA_NGS_2019`

Note:
- A dry run with all four Sentinel roots succeeded for 20 patches.
- The full all-root run was stopped because `/home/yikebe/md_sentinel2/MD_2023_raw` and `/home/yikebe/md_sentinel2/MD_2024_raw` contain about 16k date tiles and would take days with the current single-process patch extractor.
- For the first NOAA 2018/2019 training dataset, use only the 2018/2019 Sentinel roots above unless a spatial/date filter is added.
- A partial exploratory run was stopped in `make_dataset/md_sentinel_geopatches_64_256_2018_2019`; use a fresh output directory such as `make_dataset/md_sentinel_geopatches_64_256_2018_2019_full` for the full run.

### 1. Create Maryland Sentinel 64 -> 256 georeferenced patches

Dry run:

```bash
cd /home/yikebe/research/AR_Crop_Identification

/home/yikebe/.conda/envs/super_res310/bin/python make_dataset/make_patches.py sentinel-geopatches \
  --sentinel-root /home/yikebe/md_sentinel2/MD_2018_raw \
  --sentinel-root /home/yikebe/md_sentinel2_2018-2019/MD_2018_raw \
  --out make_dataset/md_sentinel_geopatches_64_256 \
  --sentinel-patch-size 64 \
  --sr-patch-size 256 \
  --scale 4 \
  --patch-overlap 0.0 \
  --max-scl-cloud-ratio 0.10 \
  --max-black-ratio 0.5 \
  --skip-sea \
  --min-sea-ratio 0.70 \
  --limit-dates 2 \
  --limit-patches 20 \
  --debug-png-count 5
```
<!-- --sentinel-root /home/yikebe/md_sentinel2/MD_2023_raw \ -->
<!-- --sentinel-root /home/yikebe/md_sentinel2/MD_2024_raw \ -->

Main 2018/2019 patch run:

```bash
cd /home/yikebe/research/AR_Crop_Identification

/home/yikebe/.conda/envs/super_res310/bin/python make_dataset/make_patches.py sentinel-geopatches \
  --sentinel-root /home/yikebe/md_sentinel2/MD_2018_raw \
  --sentinel-root /home/yikebe/md_sentinel2_2018-2019/MD_2018_raw \
  --out make_dataset/md_sentinel_geopatches_64_256_2018_2019_full \
  --sentinel-patch-size 64 \
  --sr-patch-size 256 \
  --scale 4 \
  --patch-overlap 0.0 \
  --max-scl-cloud-ratio 0.10 \
  --max-black-ratio 0.5 \
  --skip-sea \
  --min-sea-ratio 0.70 \
  --debug-png-count 20
```

Output layout:
- `make_dataset/md_sentinel_geopatches_64_256_2018_2019_full/lr_64`
- `make_dataset/md_sentinel_geopatches_64_256_2018_2019_full/sr_64_256`
- `make_dataset/md_sentinel_geopatches_64_256_2018_2019_full/hr_256`
- `make_dataset/md_sentinel_geopatches_64_256_2018_2019_full/patch_manifest.jsonl`
- `make_dataset/md_sentinel_geopatches_64_256_2018_2019_full/debug_png`

### 2. Run Sentinel -> Planet FastDiffSR inference

Use the inference config:

```bash
FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_sentinel_planet_clean_v5.json
```

It points to the checkpoint prefix:

```bash
/home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR/experiments/sr_fastdiffsr_train_64_256_Sentinel_Planet_clean_v5_260428_223636/checkpoint/I457600_E931
```

Run inference and convert FastDiffSR PNG outputs back to georeferenced 256x256 GeoTIFF reconstructions:

```bash
cd /home/yikebe/research/AR_Crop_Identification

/home/yikebe/.conda/envs/super_res310/bin/python make_dataset/run_fastdiffsr_infer_geopatches.py \
  --dataset-root make_dataset/md_sentinel_geopatches_64_256_2018_2019_full \
  --manifest make_dataset/md_sentinel_geopatches_64_256_2018_2019_full/patch_manifest.jsonl \
  --config FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_sentinel_planet_clean_v5.json \
  --infer-script FastDiffSR/FastDiffSR/infer.py \
  --results-dir FastDiffSR/FastDiffSR/experiments/md_sentinel_planet_clean_v5_infer/results \
  --reconstruction-out make_dataset/md_sentinel_planet_clean_v5_reconstruction_256 \
  --gpu-ids 0

 MKL_THREADING_LAYER=GNU MPLCONFIGDIR=/tmp/matplotlib-fastdiffsr /home/yikebe/.conda/envs/super_res310/bin/python
  make_dataset/run_fastdiffsr_infer_geopatches.py --dataset-root make_dataset/md_sentinel_geopatches_64_256_full --manifest
  make_dataset/md_sentinel_geopatches_64_256_full/patch_manifest.jsonl --config
  FastDiffSR/FastDiffSR/config/sr_fastdiffsr_infer_x4_sentinel_planet_clean_v5.json --infer-script FastDiffSR/FastDiffSR/infer.py --results-dir
  FastDiffSR/FastDiffSR/experiments/md_sentinel_planet_clean_v5_infer/results --reconstruction-out
  make_dataset/md_sentinel_planet_clean_v5_reconstruction_256 --gpu-ids 0
```

Outputs:
- `FastDiffSR/FastDiffSR/experiments/md_sentinel_planet_clean_v5_infer/results`
- `make_dataset/md_sentinel_planet_clean_v5_reconstruction_256/*.tif`
- `make_dataset/reconstruction_manifest.jsonl`

### 3. Build NOAA paired FastDiffSR train/val dataset

`build_fastdiffsr_planet_ortho_dataset.py` has a `reconstruction-ortho` mode for this step. It crops each 256x256 reconstruction into 64x64 LR patches, renders matching NOAA/Lexington orthophoto HR patches at 256x256, and writes FastDiffSR train/val folders.
The reconstruction-ortho mode also applies Sentinel-Planet-style RGB quality filters for LR and HR patches: black ratio, mean brightness, bright-pixel ratio, white/low-saturation ratio, dark-pixel ratio, and HR valid-pixel ratio.
Use `--max-ortho-day-diff` to keep only NOAA orthophotos close to the source Sentinel date. In the current Maryland NOAA metadata, the closest available day difference is 9 days, so a Sentinel-Planet-style 5 day limit yields no samples. A 10 day limit keeps only the closest July Sentinel/NOAA pairs; 30 days can be used only when more samples are needed for debugging.

```bash
cd /home/yikebe/research/AR_Crop_Identification

PYTHONWARNINGS=ignore /home/yikebe/.conda/envs/super_res310/bin/python make_dataset/build_fastdiffsr_planet_ortho_dataset.py reconstruction-ortho \
  --reconstruction-root make_dataset/md_sentinel_planet_clean_v5_reconstruction_256 \
  --reconstruction-manifest make_dataset/reconstruction_manifest.jsonl \
  --ortho-root /home/yikebe/NOAA_NGS/LexingtonPark_RGBN_2018_10140 \
  --ortho-root /home/yikebe/NOAA_NGS/NOAA_NGS_2019 \
  --train-out FastDiffSR/FastDiffSR/dataset/train_md_sentinel_planet_noaa_64_256 \
  --val-out FastDiffSR/FastDiffSR/dataset/val_md_sentinel_planet_noaa_64_256 \
  --lr-patch-size 64 \
  --hr-patch-size 256 \
  --scale 4 \
  --reconstruction-crop-stride 64 \
  --val-ratio 0.1 \
  --max-ortho-day-diff 10 \
  --max-black-ratio 0.5 \
  --min-valid-hr-ratio 0.95 \
  --no-save-debug-png \
  --num-workers 8 \
  --manifest-path FastDiffSR/FastDiffSR/dataset/md_sentinel_planet_noaa_pairs_manifest.jsonl
```

Output layout:
- `FastDiffSR/FastDiffSR/dataset/train_md_sentinel_planet_noaa_64_256/lr_64`
- `FastDiffSR/FastDiffSR/dataset/train_md_sentinel_planet_noaa_64_256/sr_64_256`
- `FastDiffSR/FastDiffSR/dataset/train_md_sentinel_planet_noaa_64_256/hr_256`
- `FastDiffSR/FastDiffSR/dataset/val_md_sentinel_planet_noaa_64_256/lr_64`
- `FastDiffSR/FastDiffSR/dataset/val_md_sentinel_planet_noaa_64_256/sr_64_256`
- `FastDiffSR/FastDiffSR/dataset/val_md_sentinel_planet_noaa_64_256/hr_256`

Full debug run with all reconstructed patches, debug PNGs, 10 day close-date filtering, and parallel workers:

```bash
cd /home/yikebe/research/AR_Crop_Identification

LD_PRELOAD=/home/yikebe/.conda/envs/super_res310/lib/libgomp.so.1 \
MKL_THREADING_LAYER=GNU \
PYTHONWARNINGS=ignore \
GTIFF_MAX_CUMULATED_MEM_USAGE=1000000000 \
GTIFF_USE_DEFER_STRILE_LOADING=YES \
/home/yikebe/.conda/envs/super_res310/bin/python make_dataset/build_fastdiffsr_planet_ortho_dataset.py reconstruction-ortho \
  --reconstruction-root make_dataset/md_sentinel_planet_clean_v5_reconstruction_256 \
  --reconstruction-manifest make_dataset/reconstruction_manifest.jsonl \
  --ortho-root /home/yikebe/NOAA_NGS/LexingtonPark_RGBN_2018_10140 \
  --ortho-root /home/yikebe/NOAA_NGS/NOAA_NGS_2019 \
  --train-out FastDiffSR/FastDiffSR/dataset/train_md_sentinel_planet_noaa_64_256_debug_full_10d \
  --val-out FastDiffSR/FastDiffSR/dataset/val_md_sentinel_planet_noaa_64_256_debug_full_10d \
  --lr-patch-size 64 \
  --hr-patch-size 256 \
  --scale 4 \
  --reconstruction-crop-stride 64 \
  --val-ratio 0.1 \
  --max-ortho-day-diff 10 \
  --black-threshold 30 \
  --max-black-ratio 0.2 \
  --min-valid-hr-ratio 0.95 \
  --max-lr-mean-brightness 230 \
  --max-lr-bright-ratio 0.35 \
  --max-lr-white-ratio 0.6 \
  --min-lr-mean-brightness 40 \
  --max-lr-dark-ratio 0.4 \
  --max-hr-mean-brightness 230 \
  --max-hr-bright-ratio 0.25 \
  --max-hr-white-ratio 0.6 \
  --min-hr-mean-brightness 60 \
  --max-hr-dark-ratio 0.4 \
  --ortho-stats-mode local \
  --save-debug-png \
  --num-workers 16 \
  --manifest-path FastDiffSR/FastDiffSR/dataset/md_sentinel_planet_noaa_pairs_manifest_debug_full_10d.jsonl
```

### 4. Train FastDiffSR on the NOAA paired dataset

Use the training config:

```bash
FastDiffSR/FastDiffSR/config/sr_fastdiffsr_train_64_256_md_sentinel_planet_noaa.json
```

Then run:

```bash
cd /home/yikebe/research/AR_Crop_Identification/FastDiffSR/FastDiffSR

/home/yikebe/.conda/envs/super_res310/bin/python sr_mfe.py \
  -c config/sr_fastdiffsr_train_64_256_md_sentinel_planet_noaa.json
```
