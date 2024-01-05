import random
from enum import Enum

import numpy as np
import torch

class RobustnessExperiment:
    def __init__(self, seed: int = None) -> None:
        self.seed = seed

    def train_nat(self):
        pass

    def train_adv(self):
        pass

    def test_nat(self):
        pass

    def test_adv(self):
        pass

    def reinitialize(self):
        """ Forgets everything the model has learned so far, s.t. the adversarial training can start from a clean state """
        pass

    def run(self):
        if self.seed is not None:
            self.fix_seed()
        self.reinitialize()

        self.train_nat()
        self.test_nat()
        self.test_adv() # we want both standard and adversarial accuracy

        if self.seed is not None:
            self.fix_seed()
        self.reinitialize()

        self.train_adv()
        self.test_nat()
        self.test_adv()

    def fix_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)