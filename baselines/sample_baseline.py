import torch
import torch.nn as nn
from torch import Tensor
from evaluation.model_wrapper import ModelWrapper

class SampleBaseline(ModelWrapper):
    def init_model(self, dataset_name: str, ways: int, shots: int):
        # reset method must be implemented on your own!
        # you will have to deal with 
        #   - dataset_name in {Omniglot, MiniImageNet}
        #   - ways,shots in {(5,1), (5,5)} 
        # this probably requires separate models for each scenario
        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots

        if dataset_name == "Omniglot":
            self._model = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1), # batch size = ways*shots during inference
                nn.Linear(28*28, ways),
                nn.Softmax(dim=-1)
            )
        elif dataset_name == "MiniImageNet":
            self._model = nn.Sequential(
                nn.MaxPool2d(kernel_size=4),
                nn.Flatten(start_dim=1, end_dim=-1), # batch size = ways*shots during inference
                nn.Linear(3*21*21, ways),
                nn.Softmax(dim=-1)
            )

        # self.model = torch.load(...)
        self.round = 0
        print(f"Initialize Sample model for {ways}-way x {shots}-shot scenario on {dataset_name} dataset")

    def reset_model(self):
        # reset method must be implemented on your own!
        # make sure you reset the last model that was loaded (so the same dataset_name, ways, shots)
        if self.round == 0:
            print(f"Resetting the model to the initial state (after training)")

        # model.weights = initial_weights

    def adapt(self, x_support: Tensor, y_support: Tensor):
        # adapt method must be implemented on your own!
        # should learn from the examples provided in the support set and change the model weights 
        # s.t. it will perform well on those tasks when forward() is called later
        if self.round == 0:
            print(f"Adapt to data of shape {x_support.shape} and labels are of shape {y_support.shape}")

        # model.weights += weights that make it better

    def forward(self, x_query: Tensor) -> Tensor:
        # here we choose to implement custom forward logic
        # the model has learned the tasks in the query during adaptation and should now try to predict them
        if self.round == 0:
            print(f"Run adapted model on data of shape {x_query.shape}")

        self.round += 1

        # return model(x_query)
        return self._model(x_query)