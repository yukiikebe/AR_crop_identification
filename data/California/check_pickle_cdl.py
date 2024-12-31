import os
import rasterio
import numpy as np
import pickle


path_cdl = "/home/vuonghn/research/dataset/satellite/arkansas/2023_all/data_check/cdl.tif"
path_pickle = "/home/vuonghn/research/dataset/satellite/arkansas/2023_all/data_check/17_10/"

with rasterio.open(path_cdl) as src:
    cdl_image = src.read(1)

list_pickle = os.listdir(path_pickle)
for pickle_file in list_pickle:
    h_idx, w_idx = [int(s) for s in pickle_file.split(".pickle")[0].split('_')]

    pickle_path = os.path.join(path_pickle, pickle_file)
    with open(pickle_path, 'rb') as handle:
        sample = pickle.load(handle, encoding='latin1') # (['img', 'labels', 'doy'])
        label_pickle = sample['labels'].squeeze()
    labels_sample_cdl = cdl_image[h_idx:h_idx+24, w_idx:w_idx+24]

    print("label_pickle ",pickle_file, label_pickle)
    print("labels_sample_cdl",pickle_file, labels_sample_cdl)
    # are_equal = np.array_equal(label_pickle, labels_sample_cdl)
    # print("Are label_pickle and labels_sample_cdl equal?", are_equal)


