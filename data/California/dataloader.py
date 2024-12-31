from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
import warnings
import numpy as np
import yaml

warnings.filterwarnings("ignore")

# Load the classes for mapping the labels
with open('configs/California/california_data.yaml', 'r') as file:
    california_data = yaml.safe_load(file)
california_classes = california_data['classes']
class_mappings = {key: value for key, value in california_classes.items()}

def normalize_classes(labels): # this belongs to 2 class that convert other labels  to 1 and 0
    """
    Normalize the labels based on the provided dictionary.
    """
    normalized_labels = torch.zeros_like(labels)
    for new_class, original_classes in class_mappings.items():
        mask = torch.isin(labels, torch.tensor(list(original_classes)))
        normalized_labels[mask] = new_class
    return normalized_labels.int()

def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    Return a distributed dataloader.
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader

def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    """
    Return a dataloader.
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader

class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if isinstance(csv_file, str):
            data_paths = pd.read_csv(csv_file, header=None)
        elif isinstance(csv_file, (list, tuple)):
            data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths
        self.data_paths = []

        if 'filtered' in csv_file:
            for idx in range(len(data_paths)):
                pkl_file = os.path.join(self.root_dir, data_paths.iloc[idx, 0])
                self.data_paths.append(pkl_file)
        else:
            for idx in range(len(data_paths)):
                subdir = data_paths.iloc[idx, 0]
                pkl_files = os.listdir(os.path.join(self.root_dir, subdir))
                pkl_files = [os.path.join(self.root_dir, subdir, pf) for pf in pkl_files]
                self.data_paths.extend(pkl_files)
        
        if 'val' in csv_file: 
            self.data_paths = self.data_paths[5000:]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_paths[idx]

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')

            if sample['img'].shape[-1] == 11:
                sample['img'] = sample['img'][..., :-1]
                sample['img'] = np.transpose(sample['img'].astype(np.float32), (0, 3, 1, 2))

        if self.transform:
            sample = self.transform(sample)

        # sample["labels"] = normalize_classes(sample["labels"])
        print("**************************")
        print(torch.unique(sample["labels"] ))

        if self.return_paths:
            return sample, img_name

        return sample