import xarray as xr
import numpy as np
import glob
from skimage import exposure
import os
import rasterio
from typing import List,Dict
import xarray as xr
import cv2
import re
from datetime import datetime



SELECTED_BANDS = {
    "10m": ["B2", "B3", "B4", "B8"], # 10m resolution
    "20m": ["B5", "B6", "B7", "B8A", "B11", "B12"], # 20m resolution
    "60m": ["B1", "B9", "QA60"], # 60m resolution
    "SCL": ["SCL"], # 20m resolution
}

def create_splits(img, input_shape, pad_mode="reflect"):
    # A function to create splits out of a particular size from a given image.
    # images are split up row wise, i.e - row1 split up, row2 split up and so on
    # NOTE - padding is added in case the image can't be split into equal parts
    #       padding is added on the right and the bottom of the image, padding type
    #       is reflected by default
    splits = []

    # calculate pad length  
    pad_len_y = (0 - img.shape[0]) % input_shape[0]  
    pad_len_x = (0 - img.shape[1]) % input_shape[1]

    # option to pad the xarray
    if isinstance(img, xr.DataArray):
        img = img.pad(pad_width={
            "y": (0, pad_len_y), 
            "x": (0, pad_len_x)}, 
            mode=pad_mode
        )

    # option to pad normal numpy array
    else:
        img = np.pad(
            img, 
            [(0, pad_len_y), (0, pad_len_x), (0, 0)], 
            pad_mode
        )

    # loop through the indices with a step of input_shape
    # and create spltis for each index
    for i in range(0, img.shape[0], input_shape[0]):
        for j in range(0, img.shape[1], input_shape[1]):
          # use array slicing to select split size from the whole image
          splits.append(img[i : i + input_shape[0], j : j + input_shape[1]].copy())
    print("Done: create_splits  ", len(splits), img.shape)
    return splits, img

def stack_bands(satellite_image, cdl_image)->dict:
    """
    Stack satellite bands and cdl image
    Args:
    satellite_image : dict : dictionary of satellite image
    cdl_image : xarray : cdl image
    Returns:
    dict : dictionary of stacked image
    """
    stacked_image = {}

    # band 10m
    B2_image = satellite_image["10m"]["B2"]
    h10,w10 = B2_image["image"].shape #(1908, 2842)
    x10 = B2_image["x"]
    y10 = B2_image["y"]

    SCL_image = satellite_image["SCL"]["SCL"]["image"] #(954, 1421)
    upsampled_SCL_image = cv2.resize(SCL_image, (w10,h10), interpolation=cv2.INTER_NEAREST) #(1908, 2842)

    stacked_image_10m = np.dstack([
        satellite_image["10m"]["B2"]["image"],
        satellite_image["10m"]["B3"]["image"],
        satellite_image["10m"]["B4"]["image"],
        satellite_image["10m"]["B8"]["image"],
        upsampled_SCL_image,
        cdl_image, #(1908, 2842)
        ]) #(1908, 2842, 6)
    
    stacked_image["10m"] = xr.DataArray(
        stacked_image_10m,
        dims=['y', 'x', 'bands'], 
        coords={
            'y': y10[:, 0],  # y coordinates of each pixel
            'x': x10[0, :],
            'bands': list(range(stacked_image_10m.shape[2]))

        })

    # band 20m
    stacked_image_20m = np.dstack([
        satellite_image["20m"]["B5"]["image"],
        satellite_image["20m"]["B6"]["image"],
        satellite_image["20m"]["B7"]["image"],
        satellite_image["20m"]["B8A"]["image"],
        satellite_image["20m"]["B11"]["image"],
        satellite_image["20m"]["B12"]["image"],
        ]) #(954, 1421, 6)

    x20 = satellite_image["20m"]["B5"]["x"]
    y20 = satellite_image["20m"]["B5"]["y"]
    stacked_image["20m"] = xr.DataArray(
        stacked_image_20m,
        dims=['y', 'x', 'bands'], 
        coords={
            'y': y20[:, 0],  # y coordinates of each pixel
            'x': x20[0, :],
            'bands': list(range(stacked_image_20m.shape[2]))
        })

    # band 60m
    stacked_image["60m"] = np.dstack([
        satellite_image["60m"]["B1"]["image"],
        satellite_image["60m"]["B9"]["image"],
        satellite_image["60m"]["QA60"]["image"]]) #(318, 475, 3)
    x60 = satellite_image["60m"]["B1"]["x"]
    y60 = satellite_image["60m"]["B1"]["y"]
    stacked_image["60m"] = xr.DataArray(
        stacked_image["60m"],
        dims=['y', 'x', 'bands'], 
        coords={
            'y': y60[:, 0],  # y coordinates of each pixel
            'x': x60[0, :],
            'bands': list(range(stacked_image["60m"].shape[2]))
        })

    return stacked_image


def tiling_image(image_stacked)->List:
    """
    Tiling image
    Args:
    image_stacked : dict : dictionary of stacked image
    Returns:
    list : list of tiled image
    """
    tiled_image = {}
    
    N = 24 #@param {type:"number"}
    splits_10, padded_img_10 = create_splits(image_stacked["10m"], [N, N])
    N = 12 #@param {type:"number"}
    splits_20, padded_img_20 = create_splits(image_stacked["20m"], [N, N])
    N = 4
    splits_60, padded_img_60 = create_splits(image_stacked["60m"], [N, N])

    if len(splits_10) != len(splits_20) or len(splits_20) != len(splits_60):
        raise ValueError("Number of splits of 10m, 20m and 60m are not equal")
    print(padded_img_10.shape, padded_img_20.shape, padded_img_60.shape)
    tiled_image["10m"] = splits_10
    tiled_image["20m"] = splits_20
    tiled_image["60m"] = splits_60
    return tiled_image


def read_satellite_image(data_dir,date)->dict:
    """
    Read satellite image
    Args:
    data_dir : str : directory of the satellite images
    date : str : date of the image
    Returns:
    dict : dictionary of satellite image
    """
    statellite_image = {}
    for resolution, bands in SELECTED_BANDS.items():
        statellite_image[resolution] = {}
        # print("bands: ",bands)
        for band in bands:
            # print("band: ",band)
            file_path = os.path.join(data_dir,date, f"{band}_{date}.tif")
            with rasterio.open(file_path) as src:
                # image[band] = src.read(1)
                image = src.read(1)
                width = src.width  # Width of the raster in pixels
                height = src.height
                # print("image shape: ",image.shape)
                # print("width: ",width)
                # print("height: ",height)
                # exit()
                transform = src.transform # Affine transform of the raster
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x, y = (x * transform[0]) + transform[2], (y * transform[4]) + transform[5]
                statellite_image[resolution][band] = {}
                statellite_image[resolution][band]["image"] = image
                statellite_image[resolution][band]["x"] = x
                statellite_image[resolution][band]["y"] = y
    return statellite_image

def read_cdl_image(data_dir,date):
    """
    Read CDL image
    Args:
    data_dir : str : directory of the CDL images
    date : str : date of the image
    Returns:
    cdl image
    """
    file_path = os.path.join(data_dir, f"rgb_{date}_cdl.tif")
    with rasterio.open(file_path) as src:
        cdl_image = src.read(1)
        num_bands = src.count
    return cdl_image

def read_series_satellite_and_cdl(satellite_image_dir:str, cdl_image_dir:str)->dict:
    """
    Read series of satellite images
    Args:
    satellite_image_dir : str : directory of the satellite images
    cdl_image_dir : str : directory of the CDL images

    Returns:
    dict : dictionary of satellite images
    """
    dates = os.listdir(satellite_image_dir)
    sorted_dates = sorted(dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    print(f"No. dates: {len(sorted_dates)}")
    print(sorted_dates)

    # satellite_images = {}
    tiled_images = {}
    for date in sorted_dates:
        print(f"Reading image at date: {date}")
        satellite_images = read_satellite_image(satellite_image_dir,date)
        cdl_image = read_cdl_image(cdl_image_dir, "2023-03-02")
        stacked_image = stack_bands(satellite_images, cdl_image)
        tiled_image = tiling_image(stacked_image)
        tiled_images[date] = tiled_image

    return tiled_images

if __name__ == "__main__":

    satellite_image_dir = "/home/khoavo/Desktop/workplace/satelite/arkansas/"
    cdl_image_dir = "/home/khoavo/Desktop/workplace/satelite/cdl_arkansas"
    list_statellite_image = read_series_satellite_and_cdl(satellite_image_dir,cdl_image_dir)
    print(list_statellite_image.keys())
    # date = "2023-01-03"
    # statellite_image = read_satellite_image(data_dir,date)
    # print(statellite_image.keys())
