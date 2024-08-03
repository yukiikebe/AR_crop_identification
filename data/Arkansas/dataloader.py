from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data
import pickle
import warnings
warnings.filterwarnings("ignore")



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


def get_dataloader(paths, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    dataset = SatImDataset(data_paths=paths, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, data_paths, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
        # print("img_name ", img_name)
        # exit()

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1') # (['img', 'labels', 'doy'])
            if sample['img'].shape[-1] == 11:
                sample['img'] = sample['img'][..., :-1]
                sample['img'] = np.transpose(sample['img'].astype(np.float32), (0, 3, 1, 2))
        # print("before transform" , sample['img'].shape) #torch.Size([60, 24, 24, 11])
        if self.transform:
            sample = self.transform(sample)  #dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
        # print("sample ", sample.keys())
        # print("after transform" , sample['inputs'].shape) #torch.Size([60, 24, 24, 11])
        # print(img_name , sample['inputs'].shape) #torch.Size([60, 24, 24, 11])
        # exit()

        # print("self.return_paths ", self.return_paths)

        # before transform (43, 10, 24, 24)
        # aafter transform torch.Size([60, 24, 24, 11])
        
        if self.return_paths:
            return sample, img_name.split('/')[-1].split('.')[0]
        
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
