import torch
import torch.nn as nn

# Extend from this class to specify your own method
class ModelWrapper:
    def __init__(self, ways: int, shots: int) -> None:
        self._model: nn.Module = None
        self._ways = ways
        self._shots = shots

    def reset_model(self):
        """ Loads a completely fresh version of the model (= how it was directly after training) """
        # Note: potentially it may be sufficient to load the model once (if inference is side-effect free)
        self._model = None
        raise NotImplementedError()

    def adapt(self, x_support: torch.Tensor, y_support: torch.Tensor):
        """ Adapt the loaded model to distinguish the classes specified by the support-set which may or may not have been attacked """
        ### Q&A ###
        # Why do we attack only x but not y? 
        # => If we attacked both then the model wouldn't have the chance to learn the new tasks at all, which is an impossible challenge.
        # Does a scenario where only x is attacked make sense?
        # => Yes, e.g. if you are required to share your collected data with competitor. Changing the labels would be easily detectable.
        # Why is "may-or-may" not have been attacked important?
        # => Some methods, e.g. purification, can potentially benefit from knowing if the data has been attacked or not
        raise NotImplementedError()

    def forward(self, x_query: torch.Tensor) -> torch.Tensor:
        """ Predict the labels for the specified query set which may or may not have been attacked """
        self._model.eval()
        return self._model(x_query)