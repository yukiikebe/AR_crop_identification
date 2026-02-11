library(terra)
library(CropScapeR)
library(dplyr)
library(sf)
library(readxl)


# Base directory containing all the subdirectories
base_dir <- "/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all"

# List all subdirectories in the base directory
sub_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)

list_tiff_files <- function(directory) {
  return(list.files(directory, pattern = "10m_rgb_.*\\.tif$", full.names = TRUE, recursive = TRUE))
}

load_and_plot_sentinel <- function(filepath) {
  sentinel.image <- terra::rast(filepath)
  print(filepath)
  plot.new() 
  plotRGB(sentinel.image, r=1, g=2, b=3, stretch="lin")
  Sys.sleep(5)
  return(sentinel.image)
}

get_and_plot_extent <- function(image) {
  study.extent <- ext(image) %>%
    vect(.) %>%
    set.crs(image) %>%
    st_as_sf(.)
  # plot.new() 
  # plot(study.extent)
  return(study.extent)
}

get_file_name<-function(path){
  name<-basename(path)
  file_name_no_ext <- sub("\\.tif$", "", name)
  return(file_name_no_ext)
}

get_cdl<-function(year, extent, name, output){
  cdl.sa <- GetCDLData(aoi = extent, year = "2022", type = "b",format = "raster")
  cdl.sa.rast <- rast(cdl.sa)
  cdl.sa.rast.proj <- project(cdl.sa.rast$Layer_1, y="EPSG:3857", method="near", mask=FALSE, align=FALSE, use_gdal=FALSE, by_util=TRUE)
  cdl.colortable <- as.data.frame(coltab(cdl.sa.rast))
  coltab(cdl.sa.rast.proj) <- cdl.colortable
  
  cdl.sa.rast.resampled <- resample(cdl.sa.rast.proj, sentinel, method="near")
  print("ok")
  
  plot(cdl.sa.rast.resampled)
  output_path_cdl =paste0(output,"/cdl.tif") 
  print(output_path_cdl)
  writeRaster(cdl.sa.rast.resampled, filename=output_path_cdl, overwrite=TRUE)
  return(cdl.sa.rast.resampled)
}

# Loop through each subdirectory and process the files
for (dir in sub_dirs) {
  setwd(dir)
  files <- list_tiff_files(dir)
  if (length(files) > 0) {
    item <- files[1]
    sentinel <- load_and_plot_sentinel(item)
    extent <- get_and_plot_extent(sentinel)
    path<-get_file_name(item)
    output<- getwd()
    cdl<-get_cdl('2023',extent,path,output)
  }
}

# print(paste("path:", path))
# for (item in files) {
#   print(item)
#   print(class(item))
# }
