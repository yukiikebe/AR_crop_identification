import os
from glob import glob
from datetime import datetime
import json
import pickle as pkl
from collections import Counter
import multiprocessing
from functools import partial

import rasterio
import xarray as xr
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


CDL_CLASSES = {
    0: 'Background', 1: 'Corn', 2: 'Cotton', 3: 'Rice', 4: 'Sorghum', 5: 'Soybeans', 6: 'Sunflowers', 10: 'Peanuts', 12: 'Sweet Corn',
    13: 'Pop or Orn Corn', 21: 'Barley', 23: 'Spring Wheat', 24: 'Winter Wheat', 26: 'Dbl Crop WinWht/Soybeans', 27: 'Rye', 28: 'Oats',
    29: 'Millet', 36: 'Alfalfa', 37: 'Other Hay/Non Alfalfa', 43: 'Potatoes', 44: 'Other Crops', 45: 'Sugarcane', 46: 'Sweet Potatoes',
    48: 'Watermelons', 49: 'Onions', 53: 'Peas', 54: 'Tomatoes', 57: 'Herbs', 58: 'Clover/Wildflowers', 59: 'Sod/Grass Seed', 60: 'Switchgrass',
    61: 'Fallow/Idle Cropland', 67: 'Peaches', 68: 'Apples', 69: 'Grapes', 71: 'Other Tree Crops', 72: 'Citrus', 74: 'Pecans', 92: 'Aquaculture',
    111: 'Open Water', 121: 'Developed/Open Space', 122: 'Developed/Low Intensity', 123: 'Developed/Medium Intensity',
    124: 'Developed/High Intensity', 131: 'Barren', 141: 'Deciduous Forest', 142: 'Evergreen Forest', 143: 'Mixed Forest', 152: 'Shrubland',
    176: 'Grass/Pasture', 190: 'Woody Wetlands', 195: 'Herbaceous Wetlands', 205: 'Triticale', 211: 'Olives', 212: 'Oranges', 219: 'Greens',
    220: 'Plums', 222: 'Squash', 225: 'Dbl Crop WinWht/Corn', 226: 'Dbl Crop Oats/Corn', 228: 'Dbl Crop Triticale/Corn', 229: 'Pumpkins',
    236: 'Dbl Crop WinWht/Sorghum', 237: 'Dbl Crop Barley/Corn', 238: 'Dbl Crop WinWht/Cotton', 240: 'Dbl Crop Soybeans/Oats',
    241: 'Dbl Crop Corn/Soybeans', 242: 'Blueberries', 243: 'Cabbage', 254: 'Dbl Crop Barley/Soybeans', 255: 'Others',
}


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


def read_cdl_image(cdl_image_path, remap_classes):
    """
    Read CDL image
    Args:
    data_dir : str : directory of the CDL images
    date : str : date of the image
    Returns:
    cdl image
    """
    with rasterio.open(cdl_image_path) as src:
        cdl_image = src.read(1)

    # Remap classes to increasing ID
    remapped_cdl_image = np.full((cdl_image.shape[0], cdl_image.shape[1]), -1).astype(np.uint8)
    for class_id in remap_classes:
        new_class_id = remap_classes[class_id]['remapped_id']
        remapped_cdl_image[cdl_image == class_id] = new_class_id
    assert np.any(remapped_cdl_image >= 0), 'Found pixels that are not remapped.'
    return remapped_cdl_image


def read_series_satellite_and_cdl(satellite_image_dir:str, remap_classes: str)->dict:
    """
    Read series of satellite images
    Args:
    satellite_image_dir : str : directory of the satellite images
    cdl_image_dir : str : directory of the CDL images

    Returns:
    dict : dictionary of satellite images
    """
    dates = [v for v in os.listdir(satellite_image_dir) if os.path.isdir(os.path.join(satellite_image_dir, v))]
    cdl_image_path = os.path.join(satellite_image_dir, 'cdl.tif')
    sorted_dates = sorted(dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    #print(f"Processing {satellite_image_dir}")
    #print(f"No. dates: {len(sorted_dates)}")

    cdl_image = read_cdl_image(cdl_image_path, remap_classes)
    cdl_image_expanded = np.expand_dims(cdl_image, axis=-1)

    # satellite_images = {}
    stacked_images, doys = [], []
    for date in sorted_dates:
        satellite_images = read_satellite_image(satellite_image_dir,date)
        if satellite_images is None:
            continue

        stacked_image = stack_bands(satellite_images)
        stacked_image = np.concatenate([stacked_image, cdl_image_expanded], axis=-1)
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


def preprocess_sub_region(subregion_dir, output_dir, remap_classes):
    meta_patch_name = subregion_dir.split('/')[-1]
    #if os.path.isdir(f'{output_dir}/{meta_patch_name}'):
    #    continue
    #os.makedirs(f'{output_dir}/{meta_patch_name}')
    tiled_satellite_series, patch_ids, doys = read_series_satellite_and_cdl(subregion_dir, remap_classes)

    for series, patch_idx in zip(tiled_satellite_series, patch_ids):
        # print("series ", series.shape)
        series_image = series[:,:,:,:-1]
        labels = series[:1,:,:, -1]

        pkl_data = {
            'img': series_image,
            'doy': doys,
            'labels': np.array(labels, dtype=np.uint8),
        }
        # print("labels ", np.unique(pkl_data['labels']))
        pkl_path = f'{output_dir}/{meta_patch_name}/{patch_idx[0]}_{patch_idx[1]}.pickle'
        with open(pkl_path, 'wb') as f:
            pkl.dump(pkl_data, f)


def preprocess_AR(satellite_dir, output_dir, threshold=500):
    meta_patches = glob(os.path.join(satellite_dir, '*'))
    all_unique_classnames = Counter()
    for meta_patch_dir in tqdm(meta_patches):
        cdl_image_path = os.path.join(meta_patch_dir, 'cdl.tif')
        with rasterio.open(cdl_image_path) as src:
            cdl_image = src.read(1)
        class_ids, counts = np.unique(cdl_image, return_counts=True)
        for class_id, count in zip(class_ids.tolist(), counts.tolist()):
            all_unique_classnames[class_id] += count
    remap_classes, new_id = {}, 1
    for k, c in all_unique_classnames.items():
        if c < threshold or CDL_CLASSES[k] == 'Others':
            remap_classes[k] = {
                'remapped_id': 0,
                'count': c,
                'class_name': CDL_CLASSES[k], }
        else:
            remap_classes[k] = {
                'remapped_id': new_id,
                'count': c,
                'class_name': CDL_CLASSES[k], }
            new_id += 1

    for k, v in remap_classes.items():
        print('Class ID: ', k, 'Info: ', v)

    with multiprocessing.Pool(8) as pool:
        with tqdm(total=len(meta_patches)) as pbar:
            for _ in pool.imap_unordered(
                partial(preprocess_sub_region, output_dir=output_dir, remap_classes=remap_classes),
                meta_patches):
                pbar.update()

    return remap_classes


def split_data(pickle_dir, split_dir):
    latest_folder = pickle_dir.split('/')[-1]
    print(latest_folder)
    pickle_files = [os.path.join(latest_folder, f) for f in os.listdir(pickle_dir)]
    train_data, val_data = train_test_split(pickle_files, test_size=0.2, random_state=42)
    #val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert lists to DataFrames
    train_df = pd.DataFrame(train_data, columns=['dir_path'])
    val_df = pd.DataFrame(val_data, columns=['dir_path'])
    #test_df = pd.DataFrame(test_data, columns=['file_path'])

    # Save DataFrames to CSV files
    train_df.to_csv(os.path.join(split_dir, 'train_data.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(split_dir, 'val_data.csv'), index=False, header=False)
    #test_df.to_csv(os.path.join(split_dir, 'test_data.csv'), index=False, header=False)


if __name__ == "__main__":
    satellite_image_dir = "/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/"
    output_dir = "/home/khoavo/Desktop/workplace/satelite/AR23_all"
    pickle_dir = os.path.join(output_dir, 'pickle24x24')
    split_dir = os.path.join(output_dir, 'fold-paths')
    #output_dir = "/mnt/vhvkhoa_ssd2/datasets"
    #pickle_dir = os.path.join(output_dir, 'AR23_pickles_all')

    #if not os.path.exists(pickle_dir):
    #    os.makedirs(pickle_dir)

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    remap_class_info = preprocess_AR(satellite_image_dir, pickle_dir)
    with open(os.path.join(output_dir, 'classnames.json'), 'w') as f:
        json.dump(remap_class_info, f)
    #split_data(pickle_dir, split_dir)
    print("Done pre-processing Arkansas")
