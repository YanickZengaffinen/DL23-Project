import os
import torch
from torch import Tensor

from evaluation.model_wrapper import ModelWrapper
from models import ResNet12


class AQBaseline(ModelWrapper):
    def __init__(self, add_noise: bool=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._add_noise = add_noise

    @property
    def model(self):
        return self._model
    
    def init_model(self, dataset_name: str, ways: int, shots: int, fast_lr: float = 0.5):
        """
        Initialize the model for the given dataset and scenario.
        Args:
            dataset_name (str): Name of the dataset.
            ways (int): Number of classes in the few-shot classification task.
            shots (int): Number of support samples per class.
            fast_lr (float, optional): Learning rate for the adaptation steps. Defaults to 0.5.
        """

        assert (dataset_name in ['Omniglot', 'MiniImageNet']), "Dataset name must be in {Omniglot, MiniImageNet}"
        assert ((ways, shots) in [(5,1), (5,5)]), "Only (5,1) and (5,5) are supported for ways and shots"

        self.dataset_name = dataset_name
        self.ways = ways
        self.shots = shots
        self.fast_lr = fast_lr

        self._model = None
        self._model_weights = None
        self._adapted_model_weights = None

        model_name = f'aq_model_{dataset_name}_{ways}w{shots}s.pth'
        model_path = os.path.join("adversarial_querying", "models", model_name)

        # if model name contains omniglot, load omniglot model
        if self.dataset_name in ['omniglot', 'Omniglot']:
            self._model = ResNet12(output_size=ways, hidden_size=64, channels=1, dropblock_dropout=0, avg_pool=False)
        # if model name contains miniimagenet, load miniimagenet model
        elif self.dataset_name in ['miniimagenet', 'MiniImageNet', 'mini-imagenet', 'Mini-ImageNet']:
            self._model = ResNet12(output_size=ways, hidden_size=128)
        else:
            raise ValueError('Model name must contain either "omniglot" or "miniimagenet"')
        
        print(f"Loading model from {model_path}")
        self._model_weights = torch.load(model_path, map_location=self.device) # load model weights
        self._model.load_state_dict(self._model_weights) # write model weights to model
        print("Number of parameters in the model:", sum(p.numel() for p in self._model.parameters() if p.requires_grad))
        self._model.eval() # set model to evaluation mode
        print(f"Initialize model for {ways}-way x {shots}-shot scenario on {dataset_name} dataset")
        if self._add_noise:
            print("Evaluate model with noise")

    def reset_model(self):
        # reset method must be implemented on your own!
        # make sure you reset the last model that was loaded (so the same dataset_name, ways, shots)
        # print(f"Resetting the model to the initial state (after training)") 
        self._model.load_state_dict(self._model_weights)

    def adapt(self, x_support: Tensor, y_support: Tensor):
        """
        Adapt the model to the support set. 
        Args:
            x_support (Tensor): Support set images.
            y_support (Tensor): Support set labels.
        """
        # Check if x_support and y_support have the same number of samples
        assert (x_support.shape[0] == y_support.shape[0]), "x_support and y_support must have the same number of samples."

        # Check if the model is initialized
        if self._model is None:
            raise ValueError("Model is not initialized.")

        self._model.train()  # Set the model to training mode

        # Define the optimizer (adaptation steps)
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.fast_lr)

        # Forward pass and compute loss
        optimizer.zero_grad()
        predictions = self._model(x_support)
        loss = torch.nn.functional.cross_entropy(predictions,  torch.argmax(y_support, dim=-1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        self._model.eval()  # Set the model back to evaluation mode

        self._adapted_model_weights = self._model.state_dict()


    def forward(self, x_query: Tensor) -> Tensor:
        """
        Forward pass of the model.
        Args:
            x_query (Tensor): Query set images.
        Returns:
            Tensor: Predicted labels.
        """
        if self._add_noise:
            outs = []
            samples = 5
            noise_multiplier = 0.01

            model = self._model

            for sample in range(samples):
                # Reset model to adapted parameters
                model.load_state_dict(self._adapted_model_weights)

                # Add some noise to the model parameters
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * noise_multiplier)

                # Forward pass
                outs.append(model(x_query))

            # Return mean of outputs
            return torch.stack(outs).mean(dim=0)
        else:
            return self._model(x_query)

        
        
