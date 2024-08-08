#read pickle file
import pickle
import numpy as np
path = "/home/vuonghn/research/code/Agriculture/DeepSatModels/datasets/AR24/pickle24x24"
import os
list_file = os.listdir(path)
for file in list_file:
    file_path = os.path.join(path, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("label shape ", data['labels'].shape, np.unique(data['labels']))