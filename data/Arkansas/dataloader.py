from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import yaml

# Load the YAML file
with open('configs/Arkansas/cdl.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Extract keys from crop_type and non_crop_type and convert them to integers
crop_labels = list(map(int, data['crop_type'].keys()))
non_crop_labels = list(map(int, data['non_crop_type'].keys()))

# Load the classes

# with open('configs/Arkansas/arkansas_data.yaml', 'r') as file:
#     arkansas_data = yaml.safe_load(file)

# arkansas_classes = arkansas_data['classes']
# print("Arkansas classes ", arkansas_classes)
# exit()


def convert_to_crop_non_crop_labels(labels):
    """
    Convert the labels to crop/non-crop labels
    """
    crop_mask = torch.isin(labels, torch.tensor(crop_labels))
    labels = torch.zeros_like(labels)  # Set all values to 0
    labels[crop_mask] = 1  # Set crop labels to 1
    return labels

def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):

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
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv_file) == str:
            data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
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
            # print("csv file ", csv_file)
            # exit()
            for idx in range(len(data_paths)):
                subdir = data_paths.iloc[idx, 0]
                pkl_files = os.listdir(os.path.join(self.root_dir, subdir))
                pkl_files = [os.path.join(self.root_dir, subdir, pf) for pf in pkl_files]
                self.data_paths.extend(pkl_files)
        if 'val' in csv_file: 
            self.data_paths = self.data_paths[5000:]

        # print('CSV file: ', csv_file, '. Dataset size: ', len(self.data_paths))
        # exit()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_paths[idx]
        # print("dataloader: img_name ", img_name)

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1') # (['img', 'labels', 'doy'])
            # print("dataloader: pickle labels ", sample["labels"])


            if sample['img'].shape[-1] == 11:
                sample['img'] = sample['img'][..., :-1]
                sample['img'] = np.transpose(sample['img'].astype(np.float32), (0, 3, 1, 2))
        #         img_temp = sample['img']
        #         doy_temp = sample['doy']
        # sample["labels"] = np.full((1, 24, 24), 19).astype(np.uint8)

        # # sample["labels"] = np.zeros((1, 24, 24), dtype=np.uint8)#sample["labels"].astype(np.int64)
        # # print("before transform" , sample['img'].shape) #torch.Size([60, 24, 24, 11])
        # # print("before transform labels ", np.unique(sample["labels"]))
        # for i in range(27):
        #     sample["img"] = img_temp
        #     sample["doy"] = doy_temp
        #     sample["labels"] = np.full((1, 24, 24), i).astype(np.uint8)
        #     print("before transform labels ", np.unique(sample["labels"]))
        #     if self.transform:
        #         sample = self.transform(sample)  #dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
        #         print("after transform labels ", np.unique(sample["labels"]), "\n\n\n")
        # exit()

        if self.transform:
            sample = self.transform(sample)  #dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
        # print("labels in dataloader 0 ", sample["labels"].shape, np.unique(sample["labels"]))
        sample["labels"] = convert_to_crop_non_crop_labels(sample["labels"])
        # print("labels in dataloader 1 ", sample["labels"].shape, np.unique(sample["labels"]))
        # print("after transform" , sample['inputs'].shape) #torch.Size([60, 24, 24, 11])
        # print(img_name , sample['inputs'].shape) #torch.Size([60, 24, 24, 11])
        # exit()
        
        # print("self.return_paths ", self.return_paths)

        # before transform (43, 10, 24, 24)
        # aafter transform torch.Size([60, 24, 24, 11])
        # print("after transform labels ", np.unique(sample["labels"]), "\n")
        if self.return_paths:
            return sample, img_name
        
        # print("sample ", sample.keys())
        # exit()
        return sample

#     def read(self, idx, abs=False):
#         """
#         read single dataset sample corresponding to idx (index number) without any data transform applied
#         """
#         if type(idx) == int:
#             img_name = os.path.join(self.root_dir,
#                                     self.data_paths.iloc[idx, 0])
#         if type(idx) == str:
#             if abs:
#                 img_name = idx
#             else:
#                 img_name = os.path.join(self.root_dir, idx)
#         with open(img_name, 'rb') as handle:
#             sample = pickle.load(handle, encoding='latin1')
#         return sample
    
    
# def my_collate(batch):

#     "Filter out sample where mask is zero everywhere"
#     idx = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in batch]
#     batch = [b for i, b in enumerate(batch) if idx[i]]
#     return torch.utils.data.dataloader.default_collate(batch)
