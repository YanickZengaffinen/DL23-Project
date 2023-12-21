""" Collection of attack scenarios """

import torch
import torchattacks

from evaluation.model_wrapper import ModelWrapper

class Attacker:
    def __init__(self) -> None:
        pass

    def attack(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass

class WhiteboxAttacker:
    def __init__(self, model_wrapper: ModelWrapper) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper

class Natural(Attacker):
    # Does not attack at all
    def attack(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return x

class PGDAttacker(WhiteboxAttacker):
    def __init__(self, model_wrapper: ModelWrapper, steps: int, epsilon: float, alpha: float) -> None:
        super().__init__(model_wrapper)

        self._steps = steps
        self._epsilon = epsilon
        self._alpha = alpha

    def attack(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._model_wrapper._model.eval()
        pgd_attack = torchattacks.PGD(self._model_wrapper._model, self._epsilon, self._alpha, self._steps, random_start=True)
        
        return pgd_attack(x, y_true).detach()


    