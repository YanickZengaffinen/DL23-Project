import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from evaluation.model_wrapper import ModelWrapper

from uncertainty.model import NOTEModel
from uncertainty.uncertainty_util import prepare_model_for_adaptation, adapt as fast_adapt

class Uncertainty(ModelWrapper):
    def init_model(self, dataset_name: str, ways: int, shots: int):
        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots

        if dataset_name == "Omniglot":
            self.augmentation = transforms.Compose([
                transforms.RandomRotation(degrees=(-15,15)),
                transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1), ratio=(0.9, 1.1)),
            ])
            self._model = NOTEModel(1100, 256, 0.1, 'omniglot')
            self._model.load_state_dict(torch.load('models/omniglot/best-binary-classifiers-256.pt', map_location='cpu'))
            self._model.enable_adversarial_protection(noise=0.75, nr_of_samples=10)
        elif dataset_name == "MiniImageNet":
            self._model = NOTEModel(64, 1024, 0.1, 'omniglot')
            self._model.load_state_dict(torch.load('models/mini-imagenet/best-binary-classifiers-1024.pt', map_location='cpu'))
            self._model.enable_adversarial_protection(noise=0.75, nr_of_samples=10)

    def reset_model(self):
        if self.dataset_name == 'Omniglot':
            prepare_model_for_adaptation(self._model, 'models/omniglot/best-meta-embedding-maml-256.pt', self.ways, device='cpu')
        elif self.dataset_name == 'MiniImageNet':
            prepare_model_for_adaptation(self._model, 'models/mini-imagenet/best-meta-embedding-maml-1024.pt', self.ways, device='cpu')
        

    def adapt(self, x_support: Tensor, y_support: Tensor):
        lbls = torch.argmax(y_support, dim=-1)
        self._model.train()

        lr = 0.75
        if self.dataset_name == 'Omniglot': # since we did this during training on the support data...
            x_support = self.augmentation(x_support)
        elif self.dataset_name == 'MiniImageNet':
            lr = 0.25
        
        fast_adapt(self._model, x_support, lbls, self.ways, nn.BCELoss(), adaptation_steps=5, lr=lr, half_batch_size=self.shots, device='cpu')

    def forward(self, x_query: Tensor) -> Tensor:
        if self.dataset_name == 'Omniglot':
            self._model.eval()
        elif self.dataset_name == 'MiniImageNet':
            self._model.train() # avoid bug with batchnorm buffers :/
        return self._model.forward(x_query, class_ids=range(self.ways))