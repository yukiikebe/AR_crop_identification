import os
from glob import glob
from datetime import datetime
import pickle as pkl
import multiprocessing
from functools import partial

import rasterio
import xarray as xr
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml

# Load the CDL config file
with open('configs/Arkansas/cdl.yaml', 'r') as file:
    cdl_data = yaml.safe_load(file)
num2class = cdl_data['num2class']

# Load config for Arkansas dataset
with open('configs/Arkansas/arkansas_data.yaml', 'r') as file:
    arkansas_data = yaml.safe_load(file)
sample_requirements = arkansas_data['sample_requirements']
selected_bands = arkansas_data['bands']

def visual_crop_distribution(satellite_dir, output_dir):
    meta_patches = glob(os.path.join(satellite_dir, '*'))
    aggregated_class_distribution = {}

    for meta_patch in tqdm(meta_patches):
        cdl_image_path = os.path.join(meta_patch, 'cdl.tif')
        with rasterio.open(cdl_image_path) as src:
            cdl_image = src.read(1)
            class_ids, counts = np.unique(cdl_image, return_counts=True)
            for class_id, count in zip(class_ids, counts):
                class_name = num2class[class_id]
                if class_name in aggregated_class_distribution:
                    aggregated_class_distribution[class_name] += count
                else:
                    aggregated_class_distribution[class_name] = count

    sorted_aggregated_class_distribution = dict(
        sorted(aggregated_class_distribution.items(), key=lambda item: item[1], reverse=True)
    )

    plt.figure(figsize=(20, 10))
    plt.bar(sorted_aggregated_class_distribution.keys(), sorted_aggregated_class_distribution.values())
    plt.xlabel('Class Names')
    plt.ylabel('Counts')
    plt.title('Crop Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'crop_distribution.png')
    plt.savefig(save_path)

def resample_dates(dates):
    df = pd.DataFrame(dates, columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    resampled_dates = []
    
    for month, group in df.groupby('month'):
        sample_size = sample_requirements.get(month, 0)
        if sample_size > 0:
            group = group.sort_values(by='date')
            if sample_size == 1:
                middle_date = group.iloc[(group['day'] - 15).abs().argsort()[:1]]
                resampled_dates.extend(middle_date['date'].dt.strftime('%Y-%m-%d').tolist())
            elif sample_size == 2:
                beginning_date = group.iloc[:1]
                ending_date = group.iloc[-1:]
                resampled_dates.extend(beginning_date['date'].dt.strftime('%Y-%m-%d').tolist())
                resampled_dates.extend(ending_date['date'].dt.strftime('%Y-%m-%d').tolist())
    
    return resampled_dates

def create_splits(imgs, input_shape, pad_mode="reflect"):
    splits = []

    pad_len_y = (0 - imgs.shape[1]) % input_shape[0]  
    pad_len_x = (0 - imgs.shape[2]) % input_shape[1]

    if isinstance(imgs, xr.DataArray):
        imgs = imgs.pad(pad_width={"y": (0, pad_len_y), "x": (0, pad_len_x)}, mode=pad_mode)
    else:
        imgs = np.pad(imgs, [(0, 0), (0, pad_len_y), (0, pad_len_x), (0, 0)], pad_mode)

    patch_ids = []
    for i in range(0, imgs.shape[1], input_shape[0]):
        for j in range(0, imgs.shape[2], input_shape[1]):
            splits.append(imgs[:, i : i + input_shape[0], j : j + input_shape[1]].copy())
            patch_ids.append((i, j))    
    return splits, patch_ids

def stack_bands(satellite_image):
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

def read_satellite_image(data_dir, date):
    satellite_image = {}
    for resolution, bands in selected_bands.items():
        satellite_image[resolution] = {}
        for band in bands:
            file_path = os.path.join(data_dir, date, f"{band}_{date}.tif")
            with rasterio.open(file_path) as src:
                data = src.read(1)
                satellite_image[resolution][band] = data
                if band == 'SCL':
                    occlusion_ratio = np.sum(data > 7) / float(data.size)
                    if occlusion_ratio > 0.1:
                        return None

    height_10m, width_10m = satellite_image['10m']['B2'].shape
    for resolution, bands in selected_bands.items():
        for band in bands:
            if resolution == '10m':
                assert satellite_image[resolution][band].shape == (height_10m, width_10m)
            else:
                orig_im = np.copy(satellite_image[resolution][band])
                upsampled_im = cv2.resize(orig_im, (width_10m, height_10m), interpolation=cv2.INTER_NEAREST)
                assert upsampled_im.shape == (height_10m, width_10m)
                satellite_image[resolution][band] = upsampled_im

    return satellite_image

def read_cdl_image(cdl_image_path):
    with rasterio.open(cdl_image_path) as src:
        cdl_image = src.read(1)
    return cdl_image

def read_series_satellite_and_cdl(satellite_image_dir):
    dates = [v for v in os.listdir(satellite_image_dir) if os.path.isdir(os.path.join(satellite_image_dir, v))]
    dates = resample_dates(dates)
    cdl_image_path = os.path.join(satellite_image_dir, 'cdl.tif')
    sorted_dates = sorted(dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    cdl_image = read_cdl_image(cdl_image_path)
    cdl_image_expanded = np.expand_dims(cdl_image, axis=-1)

    stacked_images, doys = [], []
    for date in sorted_dates:
        satellite_images = read_satellite_image(satellite_image_dir, date)
        if satellite_images is None:
            continue

        stacked_image = stack_bands(satellite_images)
        stacked_image = np.concatenate([stacked_image, cdl_image_expanded], axis=-1)
        stacked_images.append(stacked_image)
        doys.append(datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday)

    N = 24
    stacked_images = np.stack(stacked_images, axis=0)
    tiled_images, patch_ids = create_splits(stacked_images, (N, N))
    return tiled_images, patch_ids, doys

def preprocess_sub_region(subregion_dir, output_dir):
    meta_patch_name = subregion_dir.split('/')[-1]
    if not os.path.isdir(f'{output_dir}/{meta_patch_name}'):
        os.makedirs(f'{output_dir}/{meta_patch_name}')
    
    tiled_satellite_series, patch_ids, doys = read_series_satellite_and_cdl(subregion_dir)

    for series, patch_idx in zip(tiled_satellite_series, patch_ids):
        series_image = series[:, :, :, :-1]
        labels = series[:1, :, :, -1]

        pkl_data = {
            'img': series_image,
            'doy': doys,
            'labels': np.array(labels, dtype=np.uint8),
        }
        pkl_path = f'{output_dir}/{meta_patch_name}/{patch_idx[0]}_{patch_idx[1]}.pickle'
        with open(pkl_path, 'wb') as f:
            pkl.dump(pkl_data, f)

def preprocess_satellite(satellite_dir, output_dir, num_cpus=1):
    meta_patches = glob(os.path.join(satellite_dir, '*'))
    with multiprocessing.Pool(num_cpus) as pool:
        with tqdm(total=len(meta_patches)) as pbar:
            for _ in pool.imap_unordered(
                partial(preprocess_sub_region, output_dir=output_dir),
                meta_patches):
                pbar.update()

    return True

def split_data(pickle_dir, split_dir):
    latest_folder = pickle_dir.split('/')[-1]
    pickle_files = [os.path.join(latest_folder, f) for f in os.listdir(pickle_dir)]
    train_data, val_data = train_test_split(pickle_files, test_size=0.2, random_state=42)

    train_df = pd.DataFrame(train_data, columns=['dir_path'])
    val_df = pd.DataFrame(val_data, columns=['dir_path'])

    train_df.to_csv(os.path.join(split_dir, 'train_sub_data.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(split_dir, 'val_sub_data.csv'), index=False, header=False)

if __name__ == "__main__":
    satellite_image_dir = "/data/datasets/satellite/raw_arkansas_2023/2023_all"
    output_dir = "/data/vuonghn/datasets/satellite/AR23_preprocessed"
    pickle_dir = os.path.join(output_dir, 'pickle24x24')
    split_dir = os.path.join(output_dir, 'fold-paths')

    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    
    print("\nStep 1: Visualize crop distribution")
    visual_crop_distribution(satellite_image_dir, output_dir)

    print("\nStep 2: Preprocess satellite images")
    preprocess_satellite(satellite_image_dir, pickle_dir, num_cpus=8)

    print("\nStep 3: Split data")
    split_data(pickle_dir, split_dir)
    print("Done pre-processing Arkansas")