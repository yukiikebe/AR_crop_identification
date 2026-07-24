# Arkansas data download scripts

This directory contains the source-data utilities needed outside the project
HPC. Generated imagery and CDL rasters must remain outside Git.

## Which command should I use?

Use `download_sentinel2.py` for every Sentinel-2 download. `ar_deploy.py` no
longer exposes download arguments.

Use `--auth` for the first Earth Engine run only. Omit it after credentials are
available locally.

## 1. Download Sentinel-2

`download_sentinel2.py` downloads `COPERNICUS/S2_SR_HARMONIZED` imagery from
Google Earth Engine. `--band-preset all` includes the full Harvest band set and
the B2/B3/B4/SCL subset used by FastDiffSR, so both services can share one
`<year>_AR` directory.

```bash
conda activate super_res310
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python data_download/download_sentinel2.py \
  --project <earth-engine-project> \
  --auth \
  --year 2026 \
  --month 7 \
  --band-preset all \
  --data-dir /local/data/sentinel2/2026_AR
```

This creates:

```text
/local/data/sentinel2/2026_AR/<grid>/<YYYY-MM-DD>/
```

Replace `--month 7` with `--whole-year` for a complete Harvest input year.
Repeating the monthly command with the same `--data-dir` adds or verifies a new
FastDiffSR month without creating a second copy.

### ROI reference

`--roi-bbox` uses
`min_lon,min_lat,max_lon,max_lat` in WGS84:

| ROI | `--roi-bbox` value |
| --- | --- |
| FastDiffSR PlanetScope training-source envelope | `-92.3017581016,34.7471457771,-89.8853046962,36.5201148678` |
| Harvest training/support footprint | `-92.2800605553,32.9038172827,-89.8804944923,34.7531509549` |
| Smallest rectangle containing both | `-92.3017581016,32.9038172827,-89.8804944923,36.5201148678` |
| Default Arkansas deployment bbox | `-94.7610,32.8376,-89.5522,36.6652` |

The FastDiffSR value is the union envelope of 500 georeferenced PlanetScope
source scenes recorded by the dataset-preparation workflow. Its active model
was trained as downsampled Planet → Planet; it was not trained on a
Sentinel-2 ROI. The Harvest value comes from `right_bottom_ar.json` and the
source GeoTIFF used by that workflow.

Omit `--roi-bbox` to use the larger default Arkansas bbox. To download only the
smallest area containing both recorded training footprints, append:

```bash
--roi-bbox=-92.3017581016,32.9038172827,-89.8804944923,36.5201148678
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
