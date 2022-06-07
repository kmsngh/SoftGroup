import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .scannetv2 import ScanNetV2Dataset
from .s3dis import S3DISDataset

def build_dataset(data_cfg):
    _data_cfg = data_cfg.copy()
    data_type = _data_cfg.pop('type')
    if data_type == 's3dis':
        return S3DISDataset(**_data_cfg)
    elif data_type == 'scannetv2':
        return ScanNetV2Dataset(**_data_cfg)
    else:
        raise ValueError(f'Unknown {data_type}')

def build_dataloader(dataset, batch_size=1, training=True, num_workers=1):
    shuffle = training
    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True)
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True)