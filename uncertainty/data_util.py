from typing import List
import os
import random
import learn2learn as l2l

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)
from PIL.Image import LANCZOS

import matplotlib.pyplot as plt

from collections import namedtuple


ClassSplitDatasets = namedtuple('ClassSplitDatasets', ['train_classes', 'val_classes', 'test_classes'])
SampleSplitDatasets = namedtuple('SampleSplitDatasets', ['train_samples', 'val_samples'])


class IndexMapDataset(Dataset):
    def __init__(self, dataset: Dataset, index_map: List[int]):
        self.dataset = dataset
        self.index_map = index_map
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        return self.dataset[self.index_map[index]]

class FilterClassesDataset(Dataset):
    def __init__(self, dataset: Dataset, whitelist_classes: List[int]) -> None:
        super().__init__()

        self.dataset = dataset

        self.index_map = []
        for i,(_,y) in enumerate(dataset):
            if y in whitelist_classes:
                self.index_map.append(i)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        return self.dataset[self.index_map[index]]
    
class ConsecutiveLabelRemapDataset(Dataset):
    def __init__(self, dataset: Dataset, classes: List[int]) -> None:
        super().__init__()

        self.dataset = dataset

        self.class_remap = {cls:i for i,cls in enumerate(classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x,y = self.dataset[index]
        return (x, self.class_remap[y])

class FewShotDataset(Dataset):
    def __init__(self, dataset: Dataset, ways: int, shots: int, nr_of_tasks: int) -> None:
        super().__init__()

        self.dataset = dataset
        self.ways = ways
        self.shots = shots

        self.nr_of_tasks = nr_of_tasks

        self.classes = set([]) # all classes that can be sampled
        self.label_to_indices = {}
        for i,(_,y) in enumerate(dataset):
            self.classes.add(y)
            self.label_to_indices[y] = self.label_to_indices.get(y, []) + [i]

    def __len__(self):
        return self.nr_of_tasks

    def __getitem__(self, index):
        rand = random.Random(index)
        class_ids = rand.sample(list(self.classes), k=self.ways) # shape = ways
        consec_lbl_remap = {cls:i for i,cls in enumerate(class_ids)}

        selection_per_class = {cls: rand.sample(list(range(len(self.label_to_indices[cls]))), k=self.shots) for cls in class_ids}

        selected_classes = class_ids.copy()
        all_shots_x = []
        all_shots_y = []
        for shot in range(self.shots):
            xs = []
            ys = []
            rand.shuffle(selected_classes) # reshuffle, s.t. the i-th image does not always correspond to the i-th class
            for cls in selected_classes:
                x,_ = self.dataset[self.label_to_indices[cls][selection_per_class[cls][shot]]]
                xs.append(x)
                ys.append(consec_lbl_remap[cls])

            all_shots_x.append(torch.stack(xs, dim=0))
            all_shots_y.append(torch.tensor(ys))

        X = torch.stack(all_shots_x, dim=0) # shape = shots x ways x ...
        Y = torch.stack(all_shots_y, dim=0) # shape = shots x ways
        return (X, Y, class_ids) # class id at index 0 maps to label 0
    
    def sample(self, class_ids: List[int]):
        consec_lbl_remap = {cls:i for i,cls in enumerate(class_ids)}

        selection_per_class = {cls: random.sample(list(range(len(self.label_to_indices[cls]))), k=self.shots) for cls in class_ids}

        selected_classes = class_ids.copy()
        all_shots_x = []
        all_shots_y = []
        for shot in range(self.shots):
            xs = []
            ys = []
            random.shuffle(selected_classes) # reshuffle, s.t. the i-th image does not always correspond to the i-th class
            for cls in selected_classes:
                x,_ = self.dataset[self.label_to_indices[cls][selection_per_class[cls][shot]]]
                xs.append(x)
                ys.append(consec_lbl_remap[cls])

            all_shots_x.append(torch.stack(xs, dim=0))
            all_shots_y.append(torch.tensor(ys))

        X = torch.stack(all_shots_x, dim=0) # shape = shots x ways x ...
        Y = torch.stack(all_shots_y, dim=0) # shape = shots x ways
        return (X, Y, class_ids) # class id at index 0 maps to label 0
    
class NegativeSampleDataset:
    def __init__(self, dataset: Dataset, seed: int = 42) -> None:
        super().__init__()
        self.dataset = dataset
        self.rand = random.Random(seed)

        self.label_to_indices = {}
        for i,(_,y) in enumerate(dataset):
            self.label_to_indices[y] = self.label_to_indices.get(y, []) + [i]
        self.classes = list(self.label_to_indices.keys())

    
    def sample(self, half_batch_size: int, class_id: int = None):
        # samples positives and negatives for one specific class
        positive_class_id = class_id
        if positive_class_id is None:
            positive_class_id = self.rand.choice(self.classes)

        positive_candidates = self.label_to_indices[positive_class_id]
        positive_indices = self.rand.choices(positive_candidates, k=half_batch_size)
        positive_X = [self.dataset[i][0] for i in positive_indices]
        
        negative_class_ids = self.rand.choices([self.classes[i] for i in range(len(self.classes)) if self.classes[i] != positive_class_id], k=half_batch_size)
        negative_indices = [self.rand.choice(self.label_to_indices[class_id]) for class_id in negative_class_ids]
        negative_X = [self.dataset[i][0] for i in negative_indices]

        return torch.stack(positive_X + negative_X, dim=0), torch.cat([torch.ones(half_batch_size), torch.zeros(half_batch_size)], dim=0), positive_class_id
    

class TrainValSplitDataset:
    # train-val split over samples and not classes
    def __init__(self, dataset: Dataset, val_split: float = 0.33, seed: int = 42):
        rand = random.Random(seed)

        self.dataset = dataset
        
        label_to_indices = {}
        for i,(_,y) in enumerate(dataset):
            indices = label_to_indices.get(y, [])
            indices.append(i)
            label_to_indices[y] = indices
        self.classes = list(label_to_indices.keys())

        train_indices = []
        val_indices = []

        for class_id in self.classes:
            indices = label_to_indices[class_id]
            train_split = round(len(indices) * (1.0 - val_split))
            train_indices.extend(indices[:train_split])
            val_indices.extend(indices[train_split:])

        rand.shuffle(train_indices)
        rand.shuffle(val_indices)

        self.train = IndexMapDataset(dataset, train_indices)
        self.val = IndexMapDataset(dataset, val_indices)

class AugmentDataset:
    def __init__(self, dataset: Dataset, transforms) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x,y = self.dataset[index]
        return (self.transforms(x), y)
    

def get_omniglot_datasets(seed: int = 42, train_cls_val_split: float = 0.33, val_cls_val_split: float = 0.33, augment: bool = True) -> ClassSplitDatasets:
    # note: for a given seed, this class split is consistent with the one from the evaluation code

    root = '~/data'
    root = os.path.expanduser(root)
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

    classes = list(range(1623))
    random.Random(seed).shuffle(classes)
    train_classes = classes[:1100]
    #print(f'Train Classes: {train_classes}')
    val_classes = classes[1100:1200]
    #print(f'Val Classes: {val_classes}')
    test_classes = classes[1200:]

    train_cls_dataset = ConsecutiveLabelRemapDataset(FilterClassesDataset(omniglot, whitelist_classes=train_classes), train_classes)
    val_cls_dataset = ConsecutiveLabelRemapDataset(FilterClassesDataset(omniglot, whitelist_classes=val_classes), val_classes)
    test_cls_dataset = ConsecutiveLabelRemapDataset(FilterClassesDataset(omniglot, whitelist_classes=test_classes), test_classes)

    train_split = TrainValSplitDataset(train_cls_dataset, train_cls_val_split)
    val_split = TrainValSplitDataset(val_cls_dataset, val_cls_val_split)
    test_split = TrainValSplitDataset(test_cls_dataset, 0.5)

    if augment:
        augmentation = transforms.Compose([
            transforms.RandomRotation(degrees=(-15,15)),
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1), ratio=(0.9, 1.1)),
        ])
        train_split.train = AugmentDataset(train_split.train, augmentation)
        val_split.train = AugmentDataset(val_split.train, augmentation)
        test_split.train = AugmentDataset(test_split.train, augmentation)


    return ClassSplitDatasets(
        train_classes=train_split,
        val_classes=val_split,
        test_classes=test_split # ensure test-data does not get used before evaluation
    )

def get_miniimagenet_datasets(seed: int = 42, train_cls_val_split: float = 0.33, val_cls_val_split: float = 0.33, augment: bool = True) -> ClassSplitDatasets:
    root = '~/data'
    root = os.path.expanduser(root)

    train_cls_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='train',
        download=True,
    )
    val_cls_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='validation',
        download=True,
    )

    train_split = TrainValSplitDataset(train_cls_dataset, train_cls_val_split)
    val_split = TrainValSplitDataset(val_cls_dataset, val_cls_val_split)

    if augment:
        train_data_transforms = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-10,10)),
            transforms.RandomResizedCrop(size=(84, 84), scale=(0.7, 1), ratio=(0.9, 1.1)),
            lambda x: x / 255.0,
        ])
        test_data_transforms = Compose([
            lambda x: x / 255.0,
        ])

        train_split.train = AugmentDataset(train_split.train, train_data_transforms)
        train_split.val = AugmentDataset(train_split.val, test_data_transforms)
        val_split.train = AugmentDataset(val_split.train, train_data_transforms)
        val_split.val = AugmentDataset(val_split.val, test_data_transforms)

    
    return ClassSplitDatasets(
        train_classes=train_split,
        val_classes=val_split,
        test_classes=None # ensure test-data does not get used before evaluation
    )
    

def binary_tasks_from_fewshot(X: torch.Tensor, y: torch.Tensor, class_id: int, half_batch_size: int):
    """ 
        Constructs binary one-vs-rest classification tasks from the given data,
        of which half_batch_size are positive and half_batch_size are negative (so it's balanced).
        Notice that for small shots this can lead to the same image being present multiple times.

        class_id is the positive label, all others are negatives
    """

    positive_ids = [i for i in range(X.shape[0]) if y[i].item() == class_id]
    negative_ids = [i for i in range(X.shape[0]) if y[i].item() != class_id]

    positives = random.choices(positive_ids, k=half_batch_size)
    negatives = random.choices(negative_ids, k=half_batch_size)

    positive_X = [X[i] for i in positives]
    negative_X = [X[i] for i in negatives]

    return torch.stack(positive_X + negative_X, dim=0), torch.cat([torch.ones(half_batch_size), torch.zeros(half_batch_size)], dim=0)
