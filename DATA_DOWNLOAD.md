# Phase 1: Data download

This phase downloads the Sentinel-2 imagery used by crop identification, FastDiffSR, and harvest estimation. Harvest estimation also requires a matching USDA Cropland Data Layer (CDL) raster for each tile.

Skip this phase if the required data already exists. Downloads require time and storage, so begin with one month and a small region before downloading a statewide or whole-year dataset.

## Prerequisites

- A Google Earth Engine project and local authentication.
- The deployment environment from `super_res310.yml`.
- `Rscript` with the `terra`, `CropScapeR`, and `sf` packages when downloading CDL data.

Create and activate the environment if needed:

```bash
conda env create -f super_res310.yml
conda activate super_res310
```

## Download Sentinel-2

Use `--auth` for the first interactive Earth Engine run only. Omit it after credentials are available locally.

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

The `all` band preset supplies the complete Harvest band set and the `B2`, `B3`, `B4`, and `SCL` subset used by FastDiffSR. It creates this layout:

```text
<DATA_ROOT>/
└── <year>_AR/
    └── <grid id>/
        └── <YYYY-MM-DD>/
            ├── B2_<YYYY-MM-DD>.tif
            ├── B3_<YYYY-MM-DD>.tif
            ├── B4_<YYYY-MM-DD>.tif
            ├── B5_<YYYY-MM-DD>.tif
            ├── B6_<YYYY-MM-DD>.tif
            ├── B7_<YYYY-MM-DD>.tif
            ├── B8_<YYYY-MM-DD>.tif
            ├── B8A_<YYYY-MM-DD>.tif
            ├── B11_<YYYY-MM-DD>.tif
            ├── B12_<YYYY-MM-DD>.tif
            └── SCL_<YYYY-MM-DD>.tif
```

Replace `--month 7` with `--whole-year` to download a complete year. Repeating the monthly command with the same `--data-dir` adds or verifies another month without replacing existing files. Do not mix grids generated with different `--roi-bbox` values in the same `<year>_AR` directory.

The `LD_LIBRARY_PATH` setting applies only to the downloader terminal. The inference services use their own runtime settings.

See [data_download/README.md](data_download/README.md) for every downloader option.

## Select a region

`--roi-bbox` accepts `min_lon,min_lat,max_lon,max_lat` in WGS84 coordinates.

| Region | Bounds |
| --- | --- |
| FastDiffSR PlanetScope training-source envelope | `-92.3017581016,34.7471457771,-89.8853046962,36.5201148678` |
| Harvest supported region | `-92.2800605553,32.9038172827,-89.8804944923,34.7531509549` |
| Smallest rectangle containing both | `-92.3017581016,32.9038172827,-89.8804944923,36.5201148678` |
| Default Arkansas deployment region | `-94.7610,32.8376,-89.5522,36.6652` |

Omit `--roi-bbox` to use the default Arkansas deployment region. The FastDiffSR envelope contains the 500 PlanetScope source scenes used by its dataset-preparation workflow; it does not imply complete source coverage within that rectangle.

## Download and align USDA CDL

Install the required R packages once:

```r
install.packages(c("terra", "CropScapeR", "sf"))
```

Test the download and alignment on one tile:

```bash
Rscript data_download/download_cdl.r \
  --base-dir /local/data/sentinel2/2023_AR \
  --year 2023 \
  --limit 1
```

The script writes the aligned raster to:

```text
/local/data/sentinel2/2023_AR/<grid id>/cdl.tif
```

Remove `--limit 1` to process every tile after the test succeeds. Use `--overwrite` only when existing CDL files should be replaced.

## Outputs and next phase

The download scripts create Sentinel-2 and CDL source rasters. They do not create crop prediction GeoTIFFs, a training-ready dataset, or the `harvest_summary_all_crops.xlsx` workbooks consumed by the Harvest API.

- Continue to [Training](TRAINING.md) to train crop-identification models.
- Continue to [Inference](INFERENCE.md) to use existing models, generate predictions, or prepare Harvest workbooks.
