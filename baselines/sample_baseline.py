import torch
from torch import Tensor
from evaluation.model_wrapper import ModelWrapper

class SampleBaseline(ModelWrapper):
    def init_model(self, ways: int, shots: int):
        self.ways = ways
        self.shots = shots

        # self.model = torch.load(...)
        print(f"Initialize Sample model for {ways}-way x {shots}-shot scenario")

    def reset_model(self):
        print(f"Resetting the model to the initial state (after training)")

        # model.weights = initial_weights

    def adapt(self, x_support: Tensor, y_support: Tensor):
        print(f"Adapt to data of shape {x_support.shape} and labels are of shape {y_support.shape}")

        # model.weights += weights that make it better

    def forward(self, x_query: Tensor) -> Tensor:
        print(x_query.shape)

        # return model(x_query)
        return None