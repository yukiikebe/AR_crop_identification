from __future__ import print_function, division
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset
import warnings
import numpy as np
import yaml
import blosc2


warnings.filterwarnings("ignore")


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

def get_dataloader(paths_file, root_dir, max_doy, split, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    """
    Return a dataloader.
    """
    dataset = SatImDataset(csv_file=paths_file, split=split, root_dir=root_dir, max_doy=max_doy,
                           transform=transform, return_paths=return_paths)
    print("**************************")
    print("Number of entries: ", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader

class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, split, root_dir, max_doy=366, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            df = csv_file

        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths
        self.max_doy = max_doy
        self.data_paths = []

        count_skipped = 0
        for _, row in df.iterrows():
            if row["split"] != split:
                continue
            img_fp = Path(self.root_dir) / row["meta_patch"] / "img" / f"{row['tile_id']}_img.b2frame"
            label_fp = Path(self.root_dir) / row["meta_patch"] / "label_remap" / f"{row['tile_id']}_label.b2frame"
            doy_fp = Path(self.root_dir) / row["meta_patch"] / "doy" / f"{row['tile_id']}_doy.b2frame"
            if max_doy <= 365:
                doy_arr = self._read_b2frame(doy_fp, dtype=np.int16)
                if doy_arr.min() > max_doy:
                    count_skipped += 1
                    continue
            self.data_paths.append([img_fp, label_fp, doy_fp])

        if count_skipped > 0:
            print(f"Skipped {count_skipped} samples with doy > {max_doy}")

    def __len__(self):
        return len(self.data_paths)

    def _read_b2frame(self, fp: str, dtype) -> np.ndarray:
        """
        Generic Blosc-2 reader that respects the stored shape.
        `dtype` must match how the frame was written.
        """
        sch     = blosc2.open(fp, mode="r")
        shape   = np.frombuffer(sch.vlmeta["shape"], dtype=np.int32)
        out_arr = np.empty(shape, dtype=dtype)
        sch.get_slice(out=out_arr)
        return out_arr

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ------------------------------------------------------------------
        # Our `data_paths` now point to the *image* frame:
        #   .../<X_Y>/img/<iy>_<ix>_img.b2frame
        # Derive the matching label frame by string substitution.
        # ------------------------------------------------------------------
        img_path, label_path, doy_path = self.data_paths[idx]

        # ---- read frames --------------------------------------------------
        img_arr   = self._read_b2frame(img_path,   dtype=np.uint16)  # (T,24,24,11)
        label_arr = self._read_b2frame(label_path, dtype=np.uint8)   # (24,24)
        doy_arr = self._read_b2frame(doy_path, dtype=np.int16)       # (T,)

        if self.max_doy <= 365:
            doy_mask = (doy_arr < self.max_doy)
            img_arr = img_arr[doy_mask]
            doy_arr = doy_arr[doy_mask]

        # ---- replicate old pickle processing -----------------------------
        if img_arr.shape[-1] == 11:                      # drop SCL channel
            img_arr = img_arr[..., :-1]
            img_arr = np.transpose(img_arr.astype(np.float32), (0, 3, 1, 2))
                                                    # (T,C,H,W) for DL
        label_arr = label_arr[np.newaxis, ...]  # (1,H,W) for DL
        sample = {"img": img_arr, "labels": label_arr, "doy": doy_arr}

        # ---- user transforms ---------------------------------------------
        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, str(img_path)

        return sample
