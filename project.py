# Import libraries.
import os
import random
import functools
import math
from typing import Tuple

import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision as tv
from torchvision import transforms
# from torchattacks.attacks.pgd import PGD

import learn2learn as l2l
from learn2learn.data.transforms import NWays
from learn2learn.data.transforms import TaskTransform

import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import LANCZOS

# Import other files.
from my_pgd import myPGD

from mmpretrain.models import ResNet
# pip install -U openmim && mim install "mmpretrain>=1.0.0rc8"
# pip install mmpretrain

import sklearn
from sklearn.preprocessing import StandardScaler

train_batch_size = 32 # Define batch size for dataset construction.

def buildData():
    # Dataset contruction. 
    data_transforms = tv.transforms.Compose([
                tv.transforms.Resize(32, interpolation=LANCZOS),
                tv.transforms.ToTensor(),
                lambda x: 1.0 - x,
            ])

    ### Omniglot
    omniglot_dataset = l2l.vision.datasets.FullOmniglot(root='~/data', download=True, transform=data_transforms)
    omniglot_dataset = l2l.data.MetaDataset(omniglot_dataset)

    # split the dataset into train-val-test classes
    classes = list(range(1623))
    # random.shuffle(classes) # No longer shuffling cause of index issue. After the goal would be to shuffle and re-name the indexes or something.

    # Train
    train_classes = classes[:1100]
    omni_train_ds = l2l.data.FilteredMetaDataset(omniglot_dataset, labels=train_classes) 

    # Val
    val_classes = classes[1100:1200]
    omni_val_ds = l2l.data.FilteredMetaDataset(omniglot_dataset, labels=val_classes) 

    # Test
    test_classes = classes[1200:]
    omni_test_ds = l2l.data.FilteredMetaDataset(omniglot_dataset, labels=test_classes)

    # Build N way K shot.
    test_trans = [
        l2l.data.transforms.FusedNWaysKShots(omni_test_ds, n=5, k=5*2),
        l2l.data.transforms.LoadData(omni_test_ds),
        l2l.data.transforms.RemapLabels(omni_test_ds), # map to 0 - ways
        l2l.data.transforms.ConsecutiveLabels(omni_test_ds),
        l2l.vision.transforms.RandomClassRotation(omni_test_ds, [0.0, 90.0, 180.0, 270.0])
    ]
    omni_test_tasks = l2l.data.TaskDataset(dataset=omni_test_ds, task_transforms=test_trans, num_tasks=10)

    ### Mini image net. 
    miniimagenet_dataset_train = l2l.vision.datasets.MiniImagenet(root='./data', mode='train', download=True)
    miniimagenet_dataset_train = l2l.data.MetaDataset(miniimagenet_dataset_train)

    miniimagenet_dataset_val = l2l.vision.datasets.MiniImagenet(root='./data', mode='validation', download=True)
    miniimagenet_dataset_val = l2l.data.MetaDataset(miniimagenet_dataset_val)

    miniimagenet_dataset_test = l2l.vision.datasets.MiniImagenet(root='./data', mode='test', download=True)
    miniimagenet_dataset_test = l2l.data.MetaDataset(miniimagenet_dataset_test)

    # Build N way K shot.
    test_trans = [
        l2l.data.transforms.FusedNWaysKShots(miniimagenet_dataset_test, n=5, k=5*2),
        l2l.data.transforms.LoadData(miniimagenet_dataset_test),
        l2l.data.transforms.RemapLabels(miniimagenet_dataset_test), # map to 0 - ways
        l2l.data.transforms.ConsecutiveLabels(miniimagenet_dataset_test),
        l2l.vision.transforms.RandomClassRotation(miniimagenet_dataset_test, [0.0, 90.0, 180.0, 270.0])
    ]
    mini_test_tasks = l2l.data.TaskDataset(dataset=miniimagenet_dataset_test, task_transforms=test_trans, num_tasks=10)

    return omni_train_ds, omni_val_ds, omni_test_ds, miniimagenet_dataset_test, miniimagenet_dataset_val, miniimagenet_dataset_test

### Define classes for sub-modules. 
# Classification head. Input = feature imbeddings.
class headC(nn.Module):
    def __init__(self, num_classes, feature_size):
        super(headC, self).__init__()
        self.num_classes = num_classes
        self.head  = nn.Sequential(
            nn.Linear(feature_size, num_classes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        if x.size()[-1] != 1:
            x = torch.flatten(x, start_dim=1)
        x = self.head(x.squeeze())
        return x

# Combines feature extractor and classification head to create a full end-to-end model for classification.
class net_plus_head(nn.Module):
    def __init__(self, net, head):
        super(net_plus_head, self).__init__()
        self.net = net
        self.head = head
        
    def forward(self, x):
        x = self.net(x)
        x = torch.cat(x)
        x = self.head(x)
        return x
    
# Binary classifier. Input = feature embeddings.
class binaryC(nn.Module):
    def __init__(self, feature_size):
        super(binaryC, self).__init__()
        self.linear = nn.Sequential(
            # nn.Linear(feature_size, feature_size),
            # nn.ReLU(),
            nn.Linear(feature_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.size()[-1] != 1:
            x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

# Purifier module. Input = feature embeddings.
class purifierC(nn.Module):
    def __init__(self, feature_size):
        super(purifierC, self).__init__()
        self.linear = nn.Linear(feature_size, feature_size)
        
    def forward(self, x):
        if x.size()[-1] != 1:
            x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
    
# This method does the training for a given feature extractor model and hyper-parameters.
# Input:
    # feature_extractor = backbone model (ResNet18) !Output must be feature embeddings!
    # feature_s = the feature embedding size after forward pas through model. 
    # train_ds = Dataset with training data. 
    # train_epochs = number of training epochs.
    # num_classes = number of classes of the training set (for classification).
def train_model(feature_extractor, feature_s, train_ds, train_epochs, num_classes, starting_lr):
    # Define parameters.
    # Overall loss params.
    lambda1 = 0.5
    lambda2 = 0.3

    # Re-weighting params.
    alpha = 0.5
    beta = 0.5
    T  = 7
    
    # Model.
    net = feature_extractor

    # 1. Binary classifier.
    binary = binaryC(feature_size=feature_s)
    # 2. Classification head.
    head = headC(num_classes=num_classes, feature_size=feature_s)
    # 3. Feature purifier. 
    purifier = purifierC(feature_s)

    # Define losses, optimizers and schedulers.
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = nn.MSELoss()

    overall_optim = optim.Adam(net.parameters(), lr=starting_lr)
    Laa_optim = optim.Adam(binary.parameters(), lr=starting_lr)
    Lar_optim = optim.Adam(head.parameters(), lr=starting_lr)
    Lfp_optim = optim.Adam(purifier.parameters(), lr=starting_lr)

    overall_scheduler = optim.lr_scheduler.MultiStepLR(overall_optim, milestones=[20, 40], gamma=0.1)
    Laa_scheduler = optim.lr_scheduler.MultiStepLR(Laa_optim, milestones=[20, 40], gamma=0.1)
    Lar_scheduler = optim.lr_scheduler.MultiStepLR(Lar_optim, milestones=[20, 40], gamma=0.1)
    Lfp_scheduler = optim.lr_scheduler.MultiStepLR(Lfp_optim, milestones=[20, 40], gamma=0.1)

    for epoch in range(train_epochs):

        # Get data. 
        train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

        for batch in train_loader:
            X, y = batch
            b_size = X.size()[0]

            # Re-define net plus head. 
            netplushead = net_plus_head(net, head)

            # Get adversarial examples.
            attack = myPGD(netplushead, eps = 8/255, alpha = 2/255, steps = 7, random_start= True)
            adv_X, loss_list, tx = attack(X, y)

            ## Forward pass.
            # Features on real.
            y_pred = torch.cat(net.forward(X)).squeeze()

            # Features on adversarial. 
            y_pred_adv = torch.cat(net.forward(adv_X)).squeeze()

            # Full prediction on advesarial.
            y_pred_adv_full = head(y_pred)

            # 1. 
            # Adversarial Classifier here. (Make that a function/class after).

            Tanh = nn.Tanh()

            # Classify real data. (0 is truth).
            ad_pred = binary(y_pred)
            ad_truth =  torch.zeros((b_size,1))
            Laa_real = bce_loss(ad_pred, ad_truth)

            # Classify adversarial data. (1 is truth).
            ad_pred = binary(y_pred_adv)
            ad_truth =  torch.ones((b_size,1))
            Laa_adv = bce_loss(ad_pred, ad_truth)

            Laa = Laa_real + Laa_adv

            # 2. 
            # Cassification head here for adversarial re-weighted training.
            # These 3 values below idk what they are. Leave as 1 for now.

            # Get omega0 parameters. 
            var_list = []
            loss_list = torch.column_stack(loss_list)
            for i in range(loss_list.size()[0]):
                max = torch.max(loss_list[i])
                min = torch.min(loss_list[i])
                diff = max - min
                var_list.append(diff)
            var_list = torch.tensor(var_list)
            maxV = torch.max(var_list).item()
            # Prevent division by 0  which returns NaN. 
            if maxV == 0:
                maxV = torch.tensor(0.0000001)

            # Omega is calculated during the adversarial generation I am pretty sure. 
            omega0 = alpha * 0.5 *(1 + Tanh(4 - 10*(tx/T))) + beta * (var_list/maxV)
            Lar = torch.mean(torch.mul(omega0, ce_loss(y_pred_adv_full, y)))

            # 3.
            # Feature purifier here.
            # Local variable for dealing with 3 channels.
            tmp_feature_size = int(feature_s/9)
            # Feature purifier on real data.
            fp_pred = purifier(y_pred)
            if y_pred.size()[-1] != feature_s:
                fp_pred = torch.unflatten(fp_pred, dim=-1, sizes=(tmp_feature_size,3,3))
            Lfp_real = mse_loss(fp_pred, y_pred) # un flatten

            # Feature purifier on adversarial data.
            fp_pred = purifier(y_pred_adv)
            if y_pred.size()[-1] != feature_s:
                fp_pred = torch.unflatten(fp_pred, dim=-1, sizes=(tmp_feature_size,3,3))
            Lfp_adv = mse_loss(fp_pred, y_pred)

            Lfp = Lfp_real + Lfp_adv

            # Overall loss.
            overall_loss = Lar + lambda1*Laa + lambda2*Lfp
            print("Epoch: ", epoch, " Overall loss: ", overall_loss.item(), " Laa loss: ", Laa.item(), " Lar loss: ", Lar.item(), " Lfp loss: ", Lfp.item())

            # Optims.
            overall_optim.zero_grad()
            overall_loss.backward()
            
            overall_optim.step()
            Laa_optim.step()
            Lar_optim.step()
            Lfp_optim.step()

        overall_scheduler.step()
        Laa_scheduler.step()
        Lar_scheduler.step()
        Lfp_scheduler.step()

        # Return trained network. 
        return net

###### Main ######

# Get the data.    
o_train_ds, o_val_ds, o_test_ds, m_train_ds, m_val_ds, m_test_ds = buildData()

# Train Omniglot.
feature_extractor = ResNet(depth=18, in_channels=1)
feature_s = 512
train_epochs = 50
num_classes = 1100
lr = 0.0005

print("Training on Omniglot.")
# omni_model = train_model(feature_extractor, feature_s, o_train_ds, train_epochs, num_classes, lr)

# Train MiniImageNet
feature_extractor = ResNet(depth=18, in_channels=3)
feature_s = 512*3*3
train_epochs = 50
num_classes = 64
lr = 0.0005

print("Training on MiniImageNet.")
mini_model = train_model(feature_extractor, feature_s, m_train_ds, train_epochs, num_classes, lr)