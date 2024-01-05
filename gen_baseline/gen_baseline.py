import torch
import torch.nn as nn
from torch import Tensor
from evaluation.model_wrapper import ModelWrapper

from gen_baseline.project import net_plus_head, headC, ProtoNet

class GenBaseline(ModelWrapper):
    def init_model(self, dataset_name: str, ways: int, shots: int):

        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots

        if dataset_name == "Omniglot":

            # Load trained feature extractor.
            m = torch.load('./gen_baseline/models/omni_model.pt', map_location=torch.device('cpu'))

            # Add classification head.
            # h = headC(num_classes=5, feature_size=512) # Old classfication head with just a linear layer.
            h = ProtoNet(x_dim=512, hid_dim=64, z_dim=64, ch=True, num_classes=5, o=64)

            # Pass whole model (concatenated from the two above.)
            self._model = net_plus_head(m, h)
            self._model.eval()

        elif dataset_name == "MiniImageNet":
            # Load trained feature extractor.
            m = torch.load('./gen_baseline/models/mini_model.pt', map_location=torch.device('cpu'))

            # Add classification head. 
            # h = headC(num_classes=5, feature_size=512*3*3) # Old classfication head with just a linear layer.
            h = ProtoNet(x_dim=512, hid_dim=64, z_dim=64, ch=True, num_classes=5, o=576)

            # Pass whole model (concatenated from the two above.)
            self._model = net_plus_head(m, h)
            self._model.eval()

        self.round = 0
        # print(f"Initialize Sample model for {ways}-way x {shots}-shot scenario on {dataset_name} dataset")

    def reset_model(self):

        if self.round == 0: 
            self.init_model(dataset_name=self.dataset_name, ways=self.ways, shots=self.shots)
            # print(f"Resetting the model to the initial state (after training)")

    def adapt(self, x_support: Tensor, y_support: Tensor):

        if self.round == 0:
            # print(f"Adapt to data of shape {x_support.shape} and labels are of shape {y_support.shape}")

            self._model.train()
            loss = nn.CrossEntropyLoss()
            optim = torch.optim.Adam(self._model.parameters(), lr=0.0001)
            
            # Loop in case adapting multiple times.
            for _ in range(1):
              predictions = self._model.forward(x_support)
              l = loss(predictions, y_support)
              optim.zero_grad()
              l.backward()
              optim.step()
            
            self._model.eval()

    def forward(self, x_query: Tensor) -> Tensor:
        # here we choose to implement custom forward logic
        # the model has learned the tasks in the query during adaptation and should now try to predict them
        # if self.round == 0:

        # Just inference. 
        predictions = self._model.forward(x_query)
        # print(f"Run adapted model on data of shape {x_query.shape}")

        # self.round += 1

        # return model(x_query)
        return predictions