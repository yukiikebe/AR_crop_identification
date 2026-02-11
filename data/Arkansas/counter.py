import pickle as pkl
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict 
from tqdm import tqdm
import random


if __name__ == '__main__':
    root_dir = '/home/khoavo/Desktop/workplace/satelite/AR23_all/'
    csv_file = '/home/khoavo/Desktop/workplace/satelite/AR23_all/fold-paths/train_data.csv'
    filtered_csv_file = '/home/khoavo/Desktop/workplace/satelite/AR23_all/fold-paths/train_data_filtered.csv'
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
    filtered_files = []
    total_files = 0

    for idx in tqdm(range(len(data_paths))):
        subdir = data_paths.iloc[idx, 0]
        pkl_files = os.listdir(os.path.join(root_dir, subdir))
        pkl_files = [os.path.join(root_dir, subdir, pf) for pf in pkl_files]
        total_files += len(pkl_files)
        for pkl_file in pkl_files:
            sample = pkl.load(open(pkl_file, 'rb'))
            lbl = sample['labels']
            class_ids, counts = np.unique(lbl, return_counts=True)
            for class_id, count in zip(class_ids.tolist(), counts.tolist()):
                cls_counter[class_id] += count
    
    for k, v in cls_counter.items():
        print('Class name: ', k, '. Duplicate patches: ', v)
