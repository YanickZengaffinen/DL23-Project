import torch
import torch.nn as nn
from torch import Tensor
from evaluation.model_wrapper import ModelWrapper

from project import net_plus_head, headC

class SampleBaseline(ModelWrapper):
    def init_model(self, dataset_name: str, ways: int, shots: int):
        # Init method must be implemented on your own!
        # you will have to deal with 
        #   - dataset_name in {Omniglot, MiniImageNet}
        #   - ways,shots in {(5,1), (5,5)} 
        # this probably requires separate models for each scenario
        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots

        if dataset_name == "Omniglot":

            # Load trained feature extractor.
            m = torch.load('./models/omni_model.pt')
            # Add classification head.
            h = headC(num_classes=5, feature_size=512)
            # Pass whole model (concatenated from the two above.) (512*3*3 is the from ResNet18)
            self._model = net_plus_head(m, h)

        elif dataset_name == "MiniImageNet":
            # Load trained feature extractor.
            m = torch.load('./models/mini_model.pt')
            # Add classification head. (Num of classes is 5 because it's always 5 way something.) (512*3*3 is the from ResNet18)
            h = headC(num_classes=5, feature_size=512*3*3)
            # Pass whole model (concatenated from the two above.)
            self._model = net_plus_head(m, h)

        # self.model = torch.load(...)
        self.round = 0
        print(f"Initialize Sample model for {ways}-way x {shots}-shot scenario on {dataset_name} dataset")

    def reset_model(self):
        # reset method must be implemented on your own!
        # make sure you reset the last model that was loaded (so the same dataset_name, ways, shots)
        if self.round == 0:
            # I think this works cause it justs reloads the models. 
            print(self.dataset_name)
            self.init_model(dataset_name=self.dataset_name, ways=self.ways, shots=self.shots)
            print(f"Resetting the model to the initial state (after training)")

        # model.weights = initial_weights

    def adapt(self, x_support: Tensor, y_support: Tensor):
        # adapt method must be implemented on your own!
        # should learn from the examples provided in the support set and change the model weights 
        # s.t. it will perform well on those tasks when forward() is called later
        if self.round == 0:
            print(f"Adapt to data of shape {x_support.shape} and labels are of shape {y_support.shape}")

            loss = nn.CrossEntropyLoss()
            optim = torch.optim.Adam(self._model.parameters(), lr=0.0005)
            
            predictions = self._model.forward(x_support) # Returns class probabilities.
            l = loss(predictions, y_support)
            optim.zero_grad()
            l.backward()
            optim.step()

    def forward(self, x_query: Tensor) -> Tensor:
        # here we choose to implement custom forward logic
        # the model has learned the tasks in the query during adaptation and should now try to predict them
        if self.round == 0:

            # Just inference. 
            predictions = self._model.forward(x_query)
            print(predictions.size())

            
            print(f"Run adapted model on data of shape {x_query.shape}")

        self.round += 1

        # return model(x_query)
        return self._model(x_query)