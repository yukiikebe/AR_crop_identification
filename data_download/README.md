# Arkansas data download scripts

This directory contains the source-data utilities needed outside the project
HPC. Generated imagery and CDL rasters must remain outside Git.

## Which command should I use?

| Target | Recommended command | Reason |
| --- | --- | --- |
| New FastDiffSR month | `python ar_deploy.py download ...` | Uses the deployment CLI and writes `AR_<year>_raw` |
| New Harvest year | `python data_download/download_sentinel2.py ... --data-dir <year>_AR` | Writes the exact `<year>_AR` layout expected by Harvest |
| Harvest CDL | `Rscript data_download/download_cdl.r ...` | Downloads and aligns CDL after Sentinel-2 exists |

Use `--auth` for the first Earth Engine run only. Omit it after credentials are
available locally.

## 1. Download Sentinel-2

`download_sentinel2.py` downloads `COPERNICUS/S2_SR_HARMONIZED` imagery from
Google Earth Engine. Use the repository Conda environment and authenticate with
an Earth Engine-enabled Google Cloud project.

FastDiffSR only needs B2/B3/B4/SCL:

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

This creates:

```text
/local/data/sentinel2/AR_2024_raw/<grid>/<YYYY-MM-DD>/
```

Run with `--auth` once for interactive Earth Engine authentication. Omit it on
later runs.

For a Harvest-compatible year directory, invoke the downloader directly and
set `--data-dir`:

```bash
python data_download/download_sentinel2.py \
  --project <earth-engine-project> \
  --auth \
  --year 2023 \
  --whole-year \
  --band-preset all \
  --data-dir /local/data/sentinel2/2023_AR
```

Use `--roi-bbox=min_lon,min_lat,max_lon,max_lat` to test a smaller region before
starting a statewide or whole-year download.

The `LD_LIBRARY_PATH` line ensures that geemap and matplotlib load the Conda
C++ runtime rather than an older system copy. Set it only in the downloader
terminal; the API startup section uses its own environment instructions.

## 2. Download and align USDA CDL

The CDL script requires an `Rscript` executable plus the `terra`,
`CropScapeR`, and `sf` packages:

```r
install.packages(c("terra", "CropScapeR", "sf"))
```

After Sentinel-2 exists, download the matching year's CDL and align it to every
tile's Sentinel raster:

```bash
Rscript data_download/download_cdl.r \
  --base-dir /local/data/sentinel2/2023_AR \
  --year 2023
```

Each tile receives:

```text
/local/data/sentinel2/2023_AR/<grid>/cdl.tif
```

Use `--limit 1` for a one-tile smoke test and `--overwrite` to replace existing
CDL files.

## Harvest preprocessing boundary

These scripts create the Sentinel-2 and CDL inputs. They do not create
`harvest_summary_all_crops.xlsx`; Harvest estimation still requires those
prepared workbooks under:

```text
<outputs root>/<year>_AR/<grid>/harvest_summary_all_crops.xlsx
```
