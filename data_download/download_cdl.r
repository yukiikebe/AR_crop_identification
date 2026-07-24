#!/usr/bin/env Rscript

# Download USDA CDL per Sentinel-2 tile and align it to the Sentinel raster grid.

suppressPackageStartupMessages({
  library(terra)
  library(CropScapeR)
  library(sf)
})

parse_args <- function(args) {
  out <- list(base_dir = NULL, year = NULL, overwrite = FALSE, limit = NULL)
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--base-dir") {
      i <- i + 1
      out$base_dir <- args[[i]]
    } else if (a == "--year") {
      i <- i + 1
      out$year <- args[[i]]
    } else if (a == "--overwrite") {
      out$overwrite <- TRUE
    } else if (a == "--limit") {
      i <- i + 1
      out$limit <- as.integer(args[[i]])
    } else {
      stop(paste("Unknown argument:", a))
    }
    i <- i + 1
  }
  if (is.null(out$base_dir) || out$base_dir == "") stop("--base-dir is required")
  if (is.null(out$year) || out$year == "") stop("--year is required")
  return(out)
}

find_template <- function(meta_dir) {
  # Prefer 10m_rgb_*.tif if present; fallback to any B2_*.tif.
  files_rgb <- list.files(meta_dir, pattern = "10m_rgb_.*\\.tif$", full.names = TRUE, recursive = TRUE)
  if (length(files_rgb) > 0) return(files_rgb[[1]])
  files_b2 <- list.files(meta_dir, pattern = "B2_.*\\.tif$", full.names = TRUE, recursive = TRUE)
  if (length(files_b2) > 0) return(files_b2[[1]])
  return(NA)
}

extent_as_sf <- function(rast_obj) {
  e <- terra::ext(rast_obj)
  v <- terra::vect(e)
  v <- terra::set.crs(v, terra::crs(rast_obj))
  return(sf::st_as_sf(v))
}

download_cdl_for_meta_patch <- function(meta_dir, year, overwrite) {
  out_fp <- file.path(meta_dir, "cdl.tif")
  if (!overwrite && file.exists(out_fp)) {
    return(list(ok = TRUE, skipped = TRUE, out_fp = out_fp))
  }

  template_fp <- find_template(meta_dir)
  if (is.na(template_fp)) {
    return(list(ok = FALSE, skipped = FALSE, error = "No template raster found (expected 10m_rgb_*.tif or B2_*.tif)."))
  }

  sentinel <- terra::rast(template_fp)
  study_extent <- extent_as_sf(sentinel)

  # Download CDL via CropScape API and align to Sentinel grid.
  cdl_path <- CropScapeR::GetCDLData(aoi = study_extent, year = as.character(year), type = "b", format = "raster")
  cdl_rast <- terra::rast(cdl_path)

  # CropScapeR typically returns Layer_1; fallback to the first layer.
  cdl_layer <- tryCatch(cdl_rast$Layer_1, error = function(e) cdl_rast[[1]])
  cdl_proj <- terra::project(
    cdl_layer,
    y = terra::crs(sentinel),
    method = "near",
    mask = FALSE,
    align = FALSE,
    use_gdal = FALSE,
    by_util = TRUE
  )

  # Preserve class colour-table if available (not required for training).
  ctab <- tryCatch(as.data.frame(terra::coltab(cdl_rast)), error = function(e) NULL)
  if (!is.null(ctab)) {
    try(terra::coltab(cdl_proj) <- ctab, silent = TRUE)
  }

  cdl_resampled <- terra::resample(cdl_proj, sentinel, method = "near")
  terra::writeRaster(cdl_resampled, filename = out_fp, overwrite = TRUE)
  return(list(ok = TRUE, skipped = FALSE, out_fp = out_fp))
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  base_dir <- opts$base_dir
  year <- opts$year
  overwrite <- opts$overwrite
  limit <- opts$limit

  if (!dir.exists(base_dir)) stop(paste("Base dir not found:", base_dir))
  sub_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
  if (!is.null(limit) && is.finite(limit)) sub_dirs <- head(sub_dirs, limit)

  n_ok <- 0
  n_skip <- 0
  n_fail <- 0

  for (dir in sub_dirs) {
    res <- tryCatch(
      download_cdl_for_meta_patch(dir, year, overwrite),
      error = function(e) list(ok = FALSE, skipped = FALSE, error = as.character(e))
    )
    if (isTRUE(res$ok)) {
      if (isTRUE(res$skipped)) {
        n_skip <- n_skip + 1
      } else {
        n_ok <- n_ok + 1
      }
    } else {
      n_fail <- n_fail + 1
      msg <- if (!is.null(res$error)) res$error else "unknown error"
      message("[fail] ", basename(dir), ": ", msg)
    }
  }

  message("Done. ok=", n_ok, " skipped=", n_skip, " failed=", n_fail)
  if (n_fail > 0) quit(status = 1)
}

main()
