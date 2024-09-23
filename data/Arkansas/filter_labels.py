import pickle as pkl
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict 
from tqdm import tqdm
import rasterio
import random
import xarray as xr


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
        remapped_cdl_image[cdl_image == int(class_id)] = new_class_id
    assert np.any(remapped_cdl_image >= 0), 'Found pixels that are not remapped.'
    return remapped_cdl_image


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
    return splits, patch_ids


if __name__ == '__main__':
    raw_root_dir = '/data/datasets/satellite/raw_arkansas_2023/2023_all'
    root_dir = '/data/vuonghn/datasets/satellite/AR23_processed'
    csv_file = '/data/vuonghn/datasets/satellite/AR23_processed/fold-paths/train_data.csv'
    filtered_csv_file = '/data/vuonghn/datasets/satellite/AR23_processed/fold-paths/train_data_filtered.csv'
    max_num = 100

    cls_interest = [
        'Corn',
        'Rice',
        'Sorghum',
        'Soybeans',
        'Winter Wheat',
        'Dbl Crop WinWht/Soybeans',
        'Other Hay/Non Alfalfa',
        'Sod/Grass Seed',
        'Fallow/Idle Cropland',
        'Pecans',
        'Open Water',
        'Developed/Open Space',
        'Developed/Low Intensity',
        'Developed/Medium Intensity',
        'Developed/High Intensity',
        'Barren',
        'Deciduous Forest',
        'Evergreen Forest',
        'Mixed Forest',
        'Shrubland',
        'Grass/Pasture',
        'Woody Wetlands',
        'Herbaceous Wetlands',
        'Cotton',
        'Peangguts',
        'Sweet Potatoes',
        'Aquaculture',
        'Pop or Orn Corn',
    ]

    class_dict = json.load(open(os.path.join(root_dir, 'classnames.json'), 'r'))
    dominant_ids = []
    dominant_names = []
    for k, v in class_dict.items():
        if v['class_name'] in cls_interest:
            dominant_ids.append(v['remapped_id'])
            dominant_names.append(v['class_name'])

    if type(csv_file) == str:
        data_paths = pd.read_csv(csv_file, header=None)

    cls_counter = Counter()
    cls_files = defaultdict(list)
    filtered_files = 0
    total_files = 0

    for idx in tqdm(range(len(data_paths))):
        subdir = data_paths.iloc[idx, 0]
        cdl_file = os.path.join(raw_root_dir, subdir.split('/')[-1], 'cdl.tif')
        cdl_image = read_cdl_image(cdl_file, class_dict)
        cdl_image_expanded = np.expand_dims(cdl_image, axis=-1)
        cdl_image_expanded = np.expand_dims(cdl_image_expanded, axis=0)
        tiles, patch_ids = create_splits(cdl_image_expanded, (24, 24))
        total_files += len(tiles)
        for tile, patch_id in zip(tiles, patch_ids):
            class_ids, counts = np.unique(tile, return_counts=True)
            if counts[0] < 24 * 24:
                filtered_files += 1
            for class_id, count in zip(class_ids.tolist(), counts.tolist()):
                cls_counter[class_id] += count
    total_pixels = 0
    for class_id, count in cls_counter.items():
        class_name = None
        for v in class_dict.values():
            if class_id == v['remapped_id']:
                class_name = v['class_name']
        print(class_name, count)
        total_pixels += count
    
    with open('/home/khoavo/Desktop/workplace/satelite/filtered_pixels.json', 'w') as f:
        json.dump(cls_counter, f)

    print('Total files before filtering: ', total_files)
    print('Total files after filtering: ', filtered_files)
    print('Total pixels after filtering: ', total_pixels)

    # Convert the list to a DataFrame
    #df = pd.DataFrame(filtered_files)

    # Save the DataFrame to a CSV file
    #df.to_csv(filtered_csv_file, index=False, header=False)
