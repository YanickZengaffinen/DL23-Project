from typing import List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypnettorch.hnets.structured_hmlp_examples import resnet_chunking
from hypnettorch.hnets import ChunkedHMLP, HMLP, StructuredHMLP
from hypnettorch.mnets import ResNet, LeNet, ResNetIN

from torchvision.transforms import Resize

class NOTEModel(nn.Module):
    def __init__(self, nr_of_classes: int, task_emb_size: int, softmax_temperature: float = 0.1, architecture: str = 'omniglot', verbose: bool = True):
        super().__init__()

        self.architecture = architecture
        self.task_emb_size = task_emb_size
        self.adv_protection_enabled = False
        self.fixed_noise = None
        self.softmax_temperature = softmax_temperature
        self.task_representations = nn.Parameter(torch.randn((nr_of_classes, task_emb_size)))

        if architecture == 'omniglot': 
            self.resize = None
            self.mnet = LeNet(in_shape=(28, 28, 1), num_classes=1, no_weights=True, verbose=verbose)
            self.hnet = ChunkedHMLP(self.mnet.param_shapes, layers=[100, 100, 100], chunk_emb_size=128, chunk_size=1024, uncond_in_size=task_emb_size, 
                            num_cond_embs=0, no_cond_weights=True, cond_in_size=0, verbose=verbose)
        elif architecture == 'mini-imagenet':
            self.resize = Resize((64,64))
            self.mnet = ResNetIN(in_shape=(64, 64, 3), num_classes=1, use_batch_norm=True, num_feature_maps=(32,32,64,128,128), no_weights=True, verbose=verbose)
            self.hnet = ChunkedHMLP(self.mnet.param_shapes, layers=[256, 256, 256], chunk_emb_size=64, chunk_size=4096, uncond_in_size=task_emb_size, 
                            num_cond_embs=0, no_cond_weights=True, cond_in_size=0, verbose=verbose)

    def enable_adversarial_protection(self, noise: float = 0.05, nr_of_samples: int = 5):
        self.adv_protection_enabled = True
        self.noise = noise
        self.nr_of_samples = nr_of_samples

    def disable_adversarial_protection(self):
        self.adv_protection_enabled = False

    def fix_noise(self):
        # fixes the noise, such that the same models can be sampled many times
        self.fixed_noise = torch.randn([self.nr_of_samples, *self.task_representations.shape], device=self.task_representations.device) * self.noise
    
    def free_noise(self):
        self.fixed_noise = None

    def forward_binary(self, X: torch.Tensor, class_id: int) -> torch.Tensor:
        # for performance reasons, we only allow training of one class at once 
        # (otherwise we need multiple instantiations of the mnet)
        if self.resize is not None:
            X = self.resize(X)

        task_representation = self.task_representations[class_id]
        W = self.hnet.forward(uncond_input=task_representation.view(1,-1))
        y = self.mnet.forward(X, weights=W).squeeze()
        return y

    def forward(self, X: torch.Tensor, class_ids: List[int] = None) -> torch.Tensor:
        # if no class_ids are specified, we assume we are doing inference and hence use all classifiers
        if self.resize is not None:
            X = self.resize(X)

        if class_ids is None:
            class_ids = range(self.task_representations.shape[0])

        if self.adv_protection_enabled:
            return self._forward_protected(X, class_ids)
        else:
            return self._forward_unprotected(X, class_ids)

    def _forward_protected(self, X: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
        samples = []
        # gather nr of samples
        if self.fixed_noise is not None:
            noise_sample = self.fixed_noise.detach().clone()
        else:
            noise_sample = torch.randn([self.nr_of_samples, *self.task_representations.shape], device=self.task_representations.device) * self.noise    
        
        for i in range(self.nr_of_samples):
            ys = []
            # sample for each binary classifier
            for class_id in class_ids:
                task_representation = self.task_representations[class_id] + noise_sample[i, class_id]
                
                W = self.hnet.forward(uncond_input=task_representation.view(1,-1))
                y = self.mnet.forward(X, weights=W).squeeze().unsqueeze(-1)  # ways * shots x 1
                y = F.sigmoid(y)
                ys.append(y)

            ys = torch.cat(ys, dim=-1) # ways * shots x ways
            ys = torch.softmax(ys / self.softmax_temperature, dim=-1)
            samples.append(ys)
        
        return torch.stack(samples, dim=0).mean(dim=0)
    
    def _forward_unprotected(self, X: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
        ys = []
        for class_id in class_ids:
            task_representation = self.task_representations[class_id]
            W = self.hnet.forward(uncond_input=task_representation.view(1,-1))
            y = self.mnet.forward(X, weights=W).squeeze().unsqueeze(-1)  # ways * shots x 1
            y = F.sigmoid(y)
            ys.append(y)

        ys = torch.cat(ys, dim=-1) # ways * shots x ways

        return torch.softmax(ys / self.softmax_temperature, dim=-1)
    
    def adapt_parameters(self):
        # parameters that should be adapted to the given task
        return [self.task_representations]
    

def get_omniglot_model(nr_of_classes: int, task_emb_size: int, softmax_temperature: float = 0.1):
    return NOTEModel(
        nr_of_classes=nr_of_classes, 
        task_emb_size=task_emb_size, 
        softmax_temperature=softmax_temperature, 
        architecture='omniglot'
    )

def get_miniimagenet_model(nr_of_classes: int, task_emb_size: int, softmax_temperature: float = 0.1):
    return NOTEModel(
        nr_of_classes=nr_of_classes, 
        task_emb_size=task_emb_size, 
        softmax_temperature=softmax_temperature, 
        architecture='mini-imagenet'
    )