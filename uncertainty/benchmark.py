import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from evaluation.model_wrapper import ModelWrapper

from uncertainty.model import NOTEModel
from uncertainty.uncertainty_util import prepare_model_for_adaptation, adapt as fast_adapt

TASK_EMBEDDING_SIZE_OMNIGLOT = 256
TASK_EMBEDDING_SIZE_MINIIMAGENET = 32

class Uncertainty(ModelWrapper):
    def init_model(self, dataset_name: str, ways: int, shots: int):
        if self._model is not None:
            del self._model

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots

        if dataset_name == "Omniglot":
            self.augmentation = transforms.Compose([
                transforms.RandomRotation(degrees=(-15,15)),
                transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1), ratio=(0.9, 1.1)),
            ])
            self._model = NOTEModel(1100, TASK_EMBEDDING_SIZE_OMNIGLOT, 0.1, 'omniglot')
            self._model.load_state_dict(torch.load(f'models/omniglot/best-binary-classifiers-{TASK_EMBEDDING_SIZE_OMNIGLOT}.pt', map_location=self.device))
            self._model.enable_adversarial_protection(noise=0.75, nr_of_samples=10)
        elif dataset_name == "MiniImageNet":
            print(f'Loading mini-ImageNet with task embedding size {TASK_EMBEDDING_SIZE_MINIIMAGENET}')
            self._model = NOTEModel(64, TASK_EMBEDDING_SIZE_MINIIMAGENET, 0.1, 'mini-imagenet')
            self._model.load_state_dict(torch.load(f'models/mini-imagenet/best-binary-classifiers-{TASK_EMBEDDING_SIZE_MINIIMAGENET}.pt', map_location=self.device))
            self._model.enable_adversarial_protection(noise=0.1, nr_of_samples=10)
        self._model.to(self.device)


    def reset_model(self):
        if self.dataset_name == 'Omniglot':
            prepare_model_for_adaptation(self._model, f'models/omniglot/best-meta-embedding-maml-{TASK_EMBEDDING_SIZE_OMNIGLOT}.pt', self.ways, device=self.device)
        elif self.dataset_name == 'MiniImageNet':
            prepare_model_for_adaptation(self._model, f'models/mini-imagenet/best-meta-embedding-maml-{TASK_EMBEDDING_SIZE_MINIIMAGENET}.pt', self.ways, device=self.device)
        

    def adapt(self, x_support: Tensor, y_support: Tensor):
        lbls = torch.argmax(y_support, dim=-1)
        self._model.train()

        lr = 0.75
        if self.dataset_name == 'Omniglot': # since we did this during training on the support data...
            x_support = self.augmentation(x_support)
        
        fast_adapt(self._model, x_support, lbls, self.ways, nn.BCELoss(), adaptation_steps=5, lr=lr, half_batch_size=self.shots, device=self.device)

    def forward(self, x_query: Tensor) -> Tensor:
        if self.dataset_name == 'Omniglot':
            self._model.eval()
        elif self.dataset_name == 'MiniImageNet':
            self._model.train() # avoid bug with batchnorm buffers :/
        
        x_query = x_query.to(self.device)
        return self._model.forward(x_query, class_ids=range(self.ways)).detach().cpu()