import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# print(os.listdir(parent))

import torch
import torch.nn as nn
from torch import Tensor
# from evaluation.model_wrapper import ModelWrapper
from model_wrapper import ModelWrapper
from hypershot_model.hypershot import Hypershot

class HypershotBaseline(ModelWrapper):

    def init_model(self, dataset_name: str, ways: int, shots: int):
        # reset method must be implemented on your own!
        # you will have to deal with 
        #   - dataset_name in {Omniglot, MiniImageNet}
        #   - ways,shots in {(5,1), (5,5)} 
        # this probably requires separate models for each scenario
        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots
        self.round = 0

        if dataset_name == "Omniglot":
            self._model = Hypershot(kcnn_input_channels=1, z_length=256, kcnn_weights=None,
                              hnet_hidden_layers=1, hnet_hidden_size=256, hnet_weights=None,
                              mnet_hidden_layers=1, mnet_hidden_size=128, K=shots, W=ways, k_size=4, load_w = False)
            if shots == 1:
                # With adv training
                self.weights_file = f'hypershot_model/models/ResNet18/omniglot/1shot5way/HS_1Shot_5Way_8eps_0_91Acc_0_79Adv_20240107170711_19.pth'
                # Without adv training
                # self.weights_file = f'hypershot/models/ResNet12/omniglot/1shot5way/HS_1Shot_5Way_8eps_0_93Acc_0_71Adv_20240106220633_48.pth'
                self._model.load_state_dict(torch.load(self.weights_file))
            elif shots == 5:
                # With adv training
                self.weights_file = f'hypershot_model/models/ResNet18/omniglot/5shot5way/HS_5Shot_5Way_8eps_0_96Acc_0_9Adv_20240107221732_19.pth'
                # Without adv training
                # self.weights_file = f'hypershot/models/ResNet12/omniglot/5shot5way/HS_5Shot_5Way_8eps_0_97Acc_0_83Adv_20240107023339_48.pth'
                self._model.load_state_dict(torch.load(self.weights_file))
        elif dataset_name == "MiniImageNet":
            self._model = Hypershot(kcnn_input_channels=3, z_length=256, kcnn_weights=None,
                              hnet_hidden_layers=1, hnet_hidden_size=256, hnet_weights=None,
                              mnet_hidden_layers=1, mnet_hidden_size=128, K=shots, W=ways, i_dim=84, i_cha=3, k_size=7, load_w = False)
            if shots == 1:
                # With adv training
                self.weights_file = f'hypershot_model/models/ResNet18/miniimagenet/1shot5way/HS_1Shot_5Way_8eps_0_39Acc_0_31Adv_20240108220716_30.pth'
                self._model.load_state_dict(torch.load(self.weights_file))
            elif shots == 5:
                # With adv training
                self.weights_file = f'hypershot_model/models/ResNet18/miniimagenet/5shot5way/HS_5Shot_5Way_8eps_0_58Acc_0_49Adv_20240108180427_83.pth'
                self._model.load_state_dict(torch.load(self.weights_file))

        print(f"Initialize Sample model for {ways}-way x {shots}-shot scenario on {dataset_name} dataset")

    def reset_model(self):
        # reset method must be implemented on your own!
        # make sure you reset the last model that was loaded (so the same dataset_name, ways, shots)
        if self.round == 0:
            print(f"Resetting the model to the initial state (after training)")
        
        self._model.load_state_dict(torch.load(self.weights_file))

    def adapt(self, x_support: Tensor, y_support: Tensor):
        # adapt method must be implemented on your own!
        # should learn from the examples provided in the support set and change the model weights 
        # s.t. it will perform well on those tasks when forward() is called later
        if self.round == 0:
            print(f"Adapt to data of shape {x_support.shape} and labels are of shape {y_support.shape}")

        self._model.update_kernel(x_support, y_support)


    def forward(self, x_query: Tensor) -> Tensor:
        # here we choose to implement custom forward logic
        # the model has learned the tasks in the query during adaptation and should now try to predict them
        if self.round == 0:
            print(f"Run adapted model on data of shape {x_query.shape}")
        self.round += 1

        # return model(x_query)
        return self._model(x_query)