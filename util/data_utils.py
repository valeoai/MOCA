# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import numpy as np

from PIL import Image
from torchvision import datasets
import torchvision.transforms as T


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def subset_of_ImageNet_train_split(dataset_train, subset):
    assert isinstance(subset, int)
    assert subset > 0

    all_indices = []
    for _, img_indices in buildLabelIndex(dataset_train.targets).items():
        assert len(img_indices) >= subset
        all_indices += img_indices[:subset]

    dataset_train.imgs = [dataset_train.imgs[idx] for idx in all_indices]
    dataset_train.samples = [dataset_train.samples[idx] for idx in all_indices]
    dataset_train.targets = [dataset_train.targets[idx] for idx in all_indices]
    assert len(dataset_train) == (subset * 1000)

    return dataset_train


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def prepare_knn_imagenet_datasets(
    data_path,
    global_rank,
    num_tasks, 
    input_size=224,
    subset=None,
    batch_size=32,
    num_workers=4,
    pin_mem=True):

    transform_knn = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    dataset_train_knn = ReturnIndexDataset(os.path.join(data_path, "train"), transform=transform_knn)
    dataset_val_knn = ReturnIndexDataset(os.path.join(data_path, "val"), transform=transform_knn)

    if (subset is not None) and (subset >= 1):
        dataset_train_knn = subset_of_ImageNet_train_split(dataset_train_knn, subset)
    
    sampler_knn = torch.utils.data.DistributedSampler(
        dataset_train_knn, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train_knn = torch.utils.data.DataLoader(
        dataset_train_knn,
        sampler=sampler_knn,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False)
    data_loader_val_knn = torch.utils.data.DataLoader(
        dataset_val_knn,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False)

    return dataset_train_knn, data_loader_train_knn, dataset_val_knn, data_loader_val_knn

def prepare_imagenet_datasets(
    data_path,
    global_rank,
    num_tasks, 
    input_size=224,
    subset=None,
    batch_size=32,
    num_workers=4,
    pin_mem=True,
    dist_eval=True):

    train_transform = T.Compose([
        T.RandomResizedCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=train_transform)
    dataset_val = datasets.ImageFolder(os.path.join(data_path, "val"), transform=val_transform)

    if (subset is not None) and (subset >= 1):
        dataset_train = subset_of_ImageNet_train_split(dataset_train, subset)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    if dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    return dataset_train, train_loader, dataset_val, val_loader