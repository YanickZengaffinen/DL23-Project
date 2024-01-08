""" Standardized benchmark datasets that do a consistent train/val/test split """

from tkinter import NO
from typing import Tuple
import random
from learn2learn.vision.benchmarks import BenchmarkTasksets

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)
from PIL.Image import LANCZOS

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels, FusedNWaysKShots




class TasksetWrapper():
    def __init__(self, taskset, ways: int, shots: int, num_tasks: int) -> None:
        self.taskset = taskset
        self.ways = ways
        self.shots = shots
        self.num_tasks = num_tasks

    def sample(self) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        data,labels = self.taskset.sample()

        support_indices = np.zeros(data.size(0), dtype=bool)
        support_indices[np.arange(self.shots*self.ways) * 2] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support_x, support_lbls = data[support_indices], labels[support_indices]
        query_x, query_lbls = data[query_indices], labels[query_indices]

        return (support_x, support_lbls), (query_x, query_lbls)



# Adapted from learn2learn.vision.benchmarks.omniglot_benchmark
# Deterministic class split
def omniglot_tasksets(
    train_ways,
    train_samples,
    test_ways,
    test_samples,
    root,
    seed=42,
    device=None,
    **kwargs,
):
    """
    Benchmark definition for Omniglot.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(28, interpolation=LANCZOS),
        transforms.ToTensor(),
        lambda x: 1.0 - x,
    ])
    omniglot = l2l.vision.datasets.FullOmniglot(
        root=root,
        transform=data_transforms,
        download=True,
    )
    if device is not None:
        dataset = l2l.data.OnDeviceDataset(omniglot, device=device)
    dataset = l2l.data.MetaDataset(omniglot)

    classes = list(range(1623))
    random.Random(seed).shuffle(classes)
    train_dataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    validation_datatset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
    test_dataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])

    train_transforms = [
        FusedNWaysKShots(dataset, n=train_ways, k=train_samples),
        LoadData(dataset),
        RemapLabels(dataset),
        ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    validation_transforms = [
        FusedNWaysKShots(dataset, n=test_ways, k=test_samples),
        LoadData(dataset),
        RemapLabels(dataset),
        ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_transforms = [
        FusedNWaysKShots(dataset, n=test_ways, k=test_samples),
        LoadData(dataset),
        RemapLabels(dataset),
        ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]

    _datasets = (train_dataset, validation_datatset, test_dataset)
    _transforms = (train_transforms, validation_transforms, test_transforms)
    return _datasets, _transforms

# Adapted from learn2learn.vision.benchmarks.mini_imagenet
# Actually it already does a deterministic class split but it's good to show this has been considered ;)
def mini_imagenet_tasksets(
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    root='~/data',
    data_augmentation=None,
    device=None,
    **kwargs,
):
    """Tasksets for mini-ImageNet benchmarks."""
    data_augmentation = None
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'lee2019':
        normalize = Normalize(
            mean=[120.39586422/255.0, 115.59361427/255.0, 104.54012653/255.0],
            std=[70.68188272/255.0, 68.27635443/255.0, 72.54505529/255.0],
        )
        train_data_transforms = Compose([
            ToPILImage(),
            RandomCrop(84, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        test_data_transforms = Compose([
            ToPILImage(), # this code was buggy but this fixes it
            ToTensor(),
            normalize,
        ])
    else:
        raise ValueError('Invalid data_augmentation argument.')

    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='train',
        download=True,
    )
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='validation',
        download=True,
    )
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='test',
        download=True,
    )
    if device is None:
        train_dataset.transform = train_data_transforms
        valid_dataset.transform = test_data_transforms
        test_dataset.transform = test_data_transforms
    else:
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_data_transforms,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            transform=test_data_transforms,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_data_transforms,
            device=device,
        )
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms



def get_benchmark_tasksets(name: str, ways: int, shots: int, num_tasks: int, seed: int = 42, root: str = '~/data') -> BenchmarkTasksets:
    root = os.path.expanduser(root)

    if name == 'omniglot':
        datasets, transforms = omniglot_tasksets(
            train_ways=ways,
            train_samples=2*shots,
            test_ways=ways,
            test_samples=2*shots,
            num_tasks=num_tasks,
            root=root,
            seed=seed
        )
    elif name == 'mini-imagenet':
        datasets, transforms = mini_imagenet_tasksets(
            train_ways=ways,
            train_samples=2*shots,
            test_ways=ways,
            test_samples=2*shots,
            num_tasks=num_tasks,
            root=root,
            seed=seed,
            data_augmentation='lee2019'
            
        )

    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = l2l.data.Taskset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.Taskset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = l2l.data.Taskset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    
    return BenchmarkTasksets(
        TasksetWrapper(train_tasks, ways, shots, num_tasks), 
        TasksetWrapper(validation_tasks, ways, shots, num_tasks), 
        TasksetWrapper(test_tasks, ways, shots, num_tasks)
    )