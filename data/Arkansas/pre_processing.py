import xarray as xr
import pickle as pkl
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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

import pandas as pd


SELECTED_BANDS = {
    "10m": ["B2", "B3", "B4", "B8"], # 10m resolution
    "20m": ["B5", "B6", "B7", "B8A", "B11", "B12"], # 20m resolution
    "SCL": ["SCL"], # 20m resolution
}

def scl_mapping(img, data_dir, date):
    # Define the colors in HEX and convert them to RGB
    hex_colors = [
        "#ff0004",  # Saturated or defective
        "#868686",  # Dark Area Pixels
        "#774b0a",  # Cloud Shadows
        "#10d22c",  # Vegetation
        "#ffff52",  # Bare Soils
        "#0000ff",  # Water
        "#818181",  # Clouds Low Probability / Unclassified
        "#c0c0c0",  # Clouds Medium Probability
        "#f1f1f1",  # Clouds High Probability
        "#bac5eb",  # Cirrus
        "#52fff9"   # Snow / Ice
    ]

    # Convert HEX to RGB
    colors = [tuple(int(h[i:i+2], 16) / 255.0 for i in (1, 3, 5)) for h in hex_colors]

    # Create a ListedColormap
    cmap = ListedColormap(colors)

    # Map the array values to the color map
    mapped_array = cmap(img)  # Normalize array values to the range 0-1

    # Display the image
    filename = f'{data_dir}/scl_{date}.png'  # Replace this with the desired filename
    plt.imsave(filename, mapped_array)

    print(f"Image saved as {filename}")



def create_splits(imgs, input_shape, pad_mode="reflect"):
    # A function to create splits out of a particular size from a given image.
    # images are split up row wise, i.e - row1 split up, row2 split up and so on
    # NOTE - padding is added in case the image can't be split into equal parts
    #       padding is added on the right and the bottom of the image, padding type
    #       is reflected by default
    splits = []

    # calculate pad length  
    pad_len_y = (0 - imgs.shape[1]) % input_shape[0]  
    pad_len_x = (0 - imgs.shape[2]) % input_shape[1]

    # option to pad the xarray
    if isinstance(imgs, xr.DataArray):
        imgs = imgs.pad(pad_width={
            "y": (0, pad_len_y), 
            "x": (0, pad_len_x)}, 
            mode=pad_mode
        )

    # option to pad normal numpy array
    else:
        imgs = np.pad(
            imgs, 
            [(0, 0), (0, pad_len_y), (0, pad_len_x), (0, 0)], 
            pad_mode
        )

    # loop through the indices with a step of input_shape
    # and create spltis for each index
    patch_ids = []
    for i in range(0, imgs.shape[1], input_shape[0]):
        for j in range(0, imgs.shape[2], input_shape[1]):
          # use array slicing to select split size from the whole image
          splits.append(imgs[:, i : i + input_shape[0], j : j + input_shape[1]].copy())
          patch_ids.append((i, j))
    print("Done: create_splits  ", len(splits), imgs.shape)
    return splits, patch_ids


def stack_bands(satellite_image)->dict:
    """
    Stack satellite bands and cdl image
    Args:
    satellite_image : dict : dictionary of satellite image
    cdl_image : xarray : cdl image
    Returns:
    dict : dictionary of stacked image
    """
    # It's the default order of Sentinel-2 bands (B01 to B12) except that B01,B09, and B10 are removed.

    stacked_image = np.dstack([
        satellite_image['10m']['B2'],
        satellite_image['10m']['B3'],
        satellite_image['10m']['B4'],
        satellite_image['20m']['B5'],
        satellite_image['20m']['B6'],
        satellite_image['20m']['B7'],
        satellite_image['10m']['B8'],
        satellite_image['20m']['B8A'],
        satellite_image['20m']['B11'],
        satellite_image['20m']['B12'],
        satellite_image['SCL']['SCL'],
    ])
    return stacked_image


def read_satellite_image(data_dir,date)->dict:
    """
    Read satellite image
    Args:
    data_dir : str : directory of the satellite images
    date : str : date of the image
    Returns:
    dict : dictionary of satellite image
    """
    satellite_image = {}
    for resolution, bands in SELECTED_BANDS.items():
        satellite_image[resolution] = {}
        for band in bands:
            file_path = os.path.join(data_dir,date, f"{band}_{date}.tif")
            with rasterio.open(file_path) as src:
                data = src.read(1)
                satellite_image[resolution][band] = data
                #print(resolution, band, data.dtype, data.shape, data.min(), data.max())
                if band == 'SCL':
                    occlusion_ratio = np.sum(data > 7) / float(data.size)
                    # print(np.around(occlusion_ratio, 4))
                    if  occlusion_ratio > 0.1:
                        return None
                    #scl_mapping(data, data_dir, date)

    height_10m, width_10m = satellite_image['10m']['B2'].shape
    for resolution, bands in SELECTED_BANDS.items():
        for band in bands:
            if resolution == '10m':
                assert satellite_image[resolution][band].shape == (height_10m, width_10m)
            else:
                orig_im = np.copy(satellite_image[resolution][band])
                upsampled_im = cv2.resize(orig_im, (width_10m, height_10m), interpolation=cv2.INTER_NEAREST) #(1908, 2842)
                assert upsampled_im.shape == (height_10m, width_10m)
                satellite_image[resolution][band] = upsampled_im

    return satellite_image


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
    # print("cdl_image ",np.unique(cdl_image))
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
    stacked_images, doys = [], []
    for date in sorted_dates:
        satellite_images = read_satellite_image(satellite_image_dir,date)
        if satellite_images is None:
            continue
        cdl_image = read_cdl_image(cdl_image_dir, "27classes")
        cdl_image_expanded = np.expand_dims(cdl_image, axis=-1)

        stacked_image = stack_bands(satellite_images)
        # stacked_image = np.concatenate([stacked_image, cdl_image_expanded], axis=-1)
        # print("stacked_image ", stacked_image.shape)
        # print("cdl_image ", cdl_image.shape)
        stacked_images.append(stacked_image)
        doys.append(datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday)

    N = 24
    stacked_images = np.stack(stacked_images, axis=0)
    # print("stacked_images ", stacked_images.shape)
    # exit()
    tiled_images, patch_ids = create_splits(stacked_images, (N, N))
    return tiled_images, patch_ids, doys
def preprocess_AR(satellite_dir, cdl_dir, output_dir):

    tiled_satellite_series, patch_ids, doys = read_series_satellite_and_cdl(satellite_dir,cdl_dir)

    for series, patch_idx in zip(tiled_satellite_series, patch_ids):
        # print("series ", series.shape)
        series_image = series[:,:,:,:-1]
        labels = series[:1,:,:, -1]

        pkl_data = {
            'img': series,
            'doy': doys,
            'labels': #np.array(labels, dtype=np.uint8),
        }
        with open(f'{output_dir}/{patch_idx[0]}_{patch_idx[1]}.pickle', 'wb') as f:
            pkl.dump(pkl_data, f)

def split_data(pickle_dir, split_dir):
    files = os.listdir(pickle_dir)
    latest_folder = pickle_dir.split('/')[-1]
    print(latest_folder)
    pickle_files = [os.path.join(latest_folder,f) for f in os.listdir(pickle_dir) if f.endswith('.pickle')]
    train_data, temp_data = train_test_split(pickle_files, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert lists to DataFrames
    train_df = pd.DataFrame(train_data, columns=['file_path'])
    val_df = pd.DataFrame(val_data, columns=['file_path'])
    test_df = pd.DataFrame(test_data, columns=['file_path'])

    # Save DataFrames to CSV files
    train_df.to_csv(os.path.join(split_dir, 'train_data.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(split_dir, 'val_data.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(split_dir, 'test_data.csv'), index=False, header=False)

if __name__ == "__main__":

    satellite_image_dir = "/home/vuonghn/research/dataset/satellite/arkansas/satellite_images/2023/"
    cdl_image_dir = "/home/vuonghn/research/dataset/satellite/arkansas/org_maral/cdl/"
    pickle_dir = "/home/vuonghn/research/dataset/satellite/arkansas/arkansas24/pickle24x24"
    split_dir = "/home/vuonghn/research/dataset/satellite/arkansas/arkansas24/fold-paths/"

    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    preprocess_AR(satellite_image_dir, cdl_image_dir, pickle_dir)
    # split_data(pickle_dir, split_dir)
    print("Done pre-processing Arkansas")


