from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import warnings
import yaml

warnings.filterwarnings("ignore")

# Load the classes for mapping the labels
with open('configs/Arkansas/arkansas_data.yaml', 'r') as file:
    arkansas_data = yaml.safe_load(file)
arkansas_classes = arkansas_data['classes']
class_mappings = {key: list(value.keys()) for key, value in arkansas_classes.items()}

def normalize_classes(labels):
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
    dataset = SatImDataset(data_paths=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader

def get_dataloader(paths, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    """
    Return a dataloader.
    """
    dataset = SatImDataset(data_paths=paths, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader

class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, data_paths, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            data_paths (list): List of paths to the data files.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            multilabel (bool, optional): Whether the dataset is multilabel.
            return_paths (bool, optional): Whether to return the paths of the images.
        """
        self.data_paths = data_paths
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_paths[idx])
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')  # (['img', 'labels', 'doy'])
            if sample['img'].shape[-1] == 11:
                sample['img'] = sample['img'][..., :-1]
                sample['img'] = np.transpose(sample['img'].astype(np.float32), (0, 3, 1, 2))
        
        if self.transform:
            sample = self.transform(sample)
        
        sample["labels"] = normalize_classes(sample["labels"])
        
        if self.return_paths:
            return sample, img_name.split('/')[-1].split('.')[0]
        
        return sample