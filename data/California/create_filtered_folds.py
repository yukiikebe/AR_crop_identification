import os
import json
from glob import glob
import rasterio
import cv2

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter


REMOVED_CLASSES = [
    1,  # Unkown
    2,  # Land not cropped the current or previous crop season, but cropped within the past three years
    4,  # Long term, land consistently idle for four or more years
    23, # Almonds
]

SEASONAL_MAJOR_CLASSES = [
    12,  # 5055     # Corn,Sorghum or Sudan grouped for remote sensing only
    13,  # 4748     # Alfalfa & alfalfa mixtures
    3,   # 3214     # Mixed pasture
    5,   # 2355     # Miscellaneous grain and hay
    51,  # 2145     # Rice
    14,  # 2439     # Wheat
    37,  # 2527     # Tomato (processing and market)
    15,  # 1345     # Miscellaneous grasses
    16,  # 1492     # Native pasture
    44,  # 1440     # Cotton
]

SEASONAL_MINOR_CLASSES = [
    27,  # 560      # Lettuce or Leafy Greens grouped for remote sensing only
    #28,  # Cole crops
    38,  # 637      # Onions & garlic
    20,  # Melons, squash, and cucumbers (all types)
    33,  # Safflower
    31,  # Strawberries
    40,  # Carrots
    49,  # Sunflower
    17,  # Potatoes
    29,  # Bush berries
    46,  # Sweet potatoes
    54,  # Sugar beets
    41,  # Beans (dry)
    42,  # Peppers (chili, bell, etc.)
    #32,  # Miscellaneous fields
]        

class_names = {
    1: 'Unkown',
    12: 'Corn,Sorghum or Sudan grouped for remote sensing only',
    13: 'Alfalfa & alfalfa mixtures',
    3: 'Mixed pasture',
    5: 'Miscellaneous grain and hay',
    51: 'Rice',
    14: 'Wheat',
    37: 'Tomato (processing and market)',
    15: 'Miscellaneous grasses',
    16: 'Native pasture',
    44: 'Cotton',
    27: 'Lettuce or Leafy Greens grouped for remote sensing only',
    38: 'Onions & garlic',
    20: 'Melons, squash, and cucumbers (all types)',
    33: 'Safflower',
    31: 'Strawberries',
    40: 'Carrots',
    49: 'Sunflower',
    17: 'Potatoes',
    29: 'Bush berries',
    46: 'Sweet potatoes',
    54: 'Sugar beets',
    41: 'Beans (dry)',
    42: 'Peppers (chili, bell, etc.)', 
}

# Define explicit deterministic colormaps for the class names as an ndarray
colormaps = {
    1:  np.array((0, 0, 0)),
    12: np.array((255, 0, 0)),
    13: np.array((0, 255, 0)),
    3:  np.array((0, 0, 255)),
    5:  np.array((255, 255, 0)),
    51: np.array((255, 165, 0)),
    14: np.array((128, 0, 128)),
    37: np.array((0, 255, 255)),
    15: np.array((128, 128, 0)),
    16: np.array((255, 192, 203)),
    44: np.array((0, 128, 128)),
    27: np.array((128, 0, 0)),
    38: np.array((0, 128, 0)),
    20: np.array((0, 0, 128)),
    33: np.array((192, 192, 192)),
    31: np.array((255, 20, 147)),
    40: np.array((255, 140, 0)),
    49: np.array((255, 215, 0)),
    17: np.array((139, 69, 19)),
    29: np.array((75, 0, 130)),
    46: np.array((210, 105, 30)),
    54: np.array((0, 100, 0)),
    41: np.array((70, 130, 180)),
    42: np.array((220, 20, 60)),
}

def vis_cdl_seasonal(cdl_image, save_path):
    unique_classes = np.unique(cdl_image)
    rgb_image = np.zeros((cdl_image.shape[0], cdl_image.shape[1], 3), dtype=np.uint8)

    for cls_val in unique_classes:
        # Get the mask where the class is cls_val
        mask = (remapped_cdl_image == cls_val)
        if cls_val not in colormaps:
            continue
        
        # Assign the color from colormaps[cls_val]
        # Ensure colormaps has at least cls_val rows
        rgb_image[mask] = colormaps[cls_val]

    # Create figure with two subplots (equal width)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    # Left subplot: colorized CDL image
    ax[0].imshow(rgb_image)
    ax[0].set_title("Colorized CDL Image")
    ax[0].axis('off')
    
    # Right subplot: legend
    ax[1].axis('off')
    ax[1].set_title("Color Legend")

    # For readability, let's gather only the classes that appear in the image:
    present_classes = [cls_val for cls_val in unique_classes
                       if cls_val in class_names.keys()]
    
    # Sort them so legend is consistent if needed
    present_classes.sort()
    
    # We'll display them from top (first class) to bottom (last class)
    for idx, cls_val in enumerate(present_classes):
        # Y-position in descending order so first class is at the top
        y_pos = len(present_classes) - idx - 1
        color = colormaps[cls_val]  # (R, G, B)

        # Draw a colored rectangle
        ax[1].add_patch(
            plt.Rectangle(
                (0, y_pos),       # bottom-left corner in "data" coords
                1.5,              # width
                1.0,              # height
                color=(color / 255.0)  # convert 0..255 -> 0..1
            )
        )
        
        # Label next to the rectangle
        ax[1].text(
            1.8,                # some padding to the right of the rectangle
            y_pos + 0.5,        # vertical center of the rectangle
            class_names[cls_val],
            va='center',
            fontsize=10
        )

    # Adjust axes for the legend
    ax[1].set_xlim(0, 6)  # Enough space for the colored box & text
    ax[1].set_ylim(0, len(present_classes))
    
    plt.tight_layout()
        
    # Save to file if a path was provided
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")


def create_splits(imgs, input_shape, pad_mode="reflect"):
    splits = []

    # calculate pad length  
    pad_len_y = (0 - imgs.shape[1]) % input_shape[0]  
    pad_len_x = (0 - imgs.shape[2]) % input_shape[1]

    imgs = np.pad(
        imgs, 
        [(0, 0), (0, pad_len_y), (0, pad_len_x)], 
        mode=pad_mode
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


if __name__ == "__main__":
    satellite_dir = '/mnt/data/mzarvani/new_out/out/pickle24x24'
    meta_patches = glob(os.path.join(satellite_dir, '*'))
    cdl_dir ='/home/mzarvani/ca/out1/'
    class_map = json.load(open(os.path.dirname(satellite_dir) + '/classnames.json'))
    label_dir = '/mnt/data/mzarvani/new_out/out/all_labels'

    total, counter = 0, 0
    patch_paths = []
    minor_ids, major_ids, all_unique_classes = [], [], []

    for meta_patch_dir in tqdm(meta_patches):
        cdl_name  = 'roi_mask_' + os.path.basename(meta_patch_dir)+'.tiff'
        cdl_image_path =os.path.join(cdl_dir, cdl_name)

        with rasterio.open(cdl_image_path) as src:
            cdl_image = src.read(1)

        remapped_cdl_image = np.zeros_like(cdl_image) - 1
        for k, v in class_map.items():
            orig_idx = int(k)
            new_idx = int(v['remapped_id'])
            remapped_cdl_image[cdl_image == orig_idx] = new_idx

        vis_cdl_seasonal(remapped_cdl_image, os.path.join(label_dir, 'vis_cdl', os.path.basename(meta_patch_dir) + '.png'))

        splits, patch_ids = create_splits(np.expand_dims(remapped_cdl_image, 0), (24, 24))
        for split, patch_id in zip(splits, patch_ids):
            total += 1
            split = np.squeeze(split)
            unique_classes = np.unique(split).tolist()
            ## if unique_classes are all in the removed classes, skip this patch
            #if len(set(unique_classes).intersection(REMOVED_CLASSES)) == len(unique_classes):
            #    continue

            # if any one entry of unique_classes is in REMOVED_CLASSES, skip this patch
            if any([x in REMOVED_CLASSES for x in unique_classes]):
                continue

            # if none of the unique classes are in SEASONAL_CLASSES, skip this patch
            if not any([x in SEASONAL_MINOR_CLASSES + SEASONAL_MAJOR_CLASSES for x in unique_classes]):
                continue

            for i in range(len(unique_classes)):
                if unique_classes[i] not in SEASONAL_MINOR_CLASSES + SEASONAL_MAJOR_CLASSES:
                    unique_classes[i] = 0

            patch_path = os.path.join(
                'pickle24x24',
                os.path.basename(meta_patch_dir),
                f'{patch_id[1]}_{patch_id[0]}.pickle'
            )
            patch_paths.append(patch_path)
            all_unique_classes.extend(unique_classes)
            if len(set(unique_classes).intersection(SEASONAL_MAJOR_CLASSES)) == len(unique_classes):
                major_ids.append(counter)
            else:
                minor_ids.append(counter)

            counter += 1

    selected_patches = [patch_paths[i] for i in minor_ids]
    # Randomly select N patches from major_patches and add them to selected_patches:
    np.random.seed(42)
    N = 1000
    selected_major_ids = [] #  np.random.choice(major_ids, N, replace=False)
    selected_patches.extend([patch_paths[i] for i in selected_major_ids])

    selected_unique_classes = [all_unique_classes[i] for i in minor_ids] + [all_unique_classes[i] for i in selected_major_ids]
    class_counter = Counter(selected_unique_classes)
    for k, v in class_counter.most_common():
        print(f'{k}: {v}')

    # Split the selected patches into training and validation with a ratio of 80:20
    np.random.shuffle(selected_patches)
    train_size = int(0.8 * len(selected_patches))
    train_patches = selected_patches[:train_size]
    val_patches = selected_patches[train_size:]

    # Save train and val patches to csv files:
    with open(os.path.join(os.path.dirname(satellite_dir), 'fold-paths', 'train_seasonal_filtered.csv'), 'w') as f:
        for patch in train_patches:
            f.write(f'{patch}\n')
    with open(os.path.join(os.path.dirname(satellite_dir), 'fold-paths', 'val_seasonal_filtered.csv'), 'w') as f:
        for patch in val_patches:
            f.write(f'{patch}\n')

    class_map = {}
    for i, k in enumerate(SEASONAL_MAJOR_CLASSES + SEASONAL_MINOR_CLASSES, start=1):
        class_map[int(k)] = i
    with open('class_mapping_filtered.json', 'w') as f:
        json.dump(class_map, f)

    print(f'Number of patches: {len(selected_patches)} out of {total}')
