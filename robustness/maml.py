# Adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
# See here for an introduction to MAML: https://meta-learning.fastforwardlabs.com/

from typing import Callable
from dataclasses import dataclass

import random
import numpy as np
import torch
import torch.nn as nn
from learn2learn.algorithms import MAML
from learn2learn.vision.benchmarks import BenchmarkTasksets

from torchattacks.attack import Attack

from robustness.experiment import RobustnessExperiment

########
# UTIL #
########
@dataclass
class Stats:
    loss: torch.Tensor = 0
    accuracy: torch.Tensor = 0

    def __add__(self, o: 'Stats') -> 'Stats':
        return Stats(self.loss + o.loss, self.accuracy + o.accuracy)
    
    def __truediv__(self, o: float) -> 'Stats':
        return Stats(self.loss / o, self.accuracy / o)
    
    def __str__(self):
        return f'Nat. Loss = {self.loss.item():.3f}, Nat. Acc. = {self.accuracy.item():.3f}'

@dataclass
class CombinedStats:
    nat_loss: torch.Tensor = 0
    nat_accuracy: torch.Tensor = 0
    adv_loss: torch.Tensor = 0
    adv_accuracy: torch.Tensor = 0
    trades_loss: torch.Tensor = 0

    def __add__(self, o: 'CombinedStats') -> 'CombinedStats':
        return CombinedStats(self.nat_loss + o.nat_loss, self.nat_accuracy + o.nat_accuracy, 
                             self.adv_loss + o.adv_loss, self.adv_accuracy + o.adv_accuracy, 
                             self.trades_loss + o.trades_loss)
    
    def __truediv__(self, o: float) -> 'CombinedStats':
        return CombinedStats(self.nat_loss / o, self.nat_accuracy / o, 
                             self.adv_loss / o, self.adv_accuracy / o, self.trades_loss / o)

    def __str__(self):
        return f'TRADES Loss = {self.trades_loss.item():.3f}, Nat. Acc. = {self.nat_accuracy.item():.3f}, Adv. Acc. = {self.adv_accuracy.item():.3f}'


@dataclass
class Datasets:
    X_support: torch.Tensor
    y_support: torch.Tensor
    X_query: torch.Tensor
    y_query: torch.Tensor


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def split_support_query(X: torch.Tensor, y: torch.Tensor, shots: int, ways: int):
    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(X.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = X[adaptation_indices], y[adaptation_indices]
    evaluation_data, evaluation_labels = X[evaluation_indices], y[evaluation_indices]

    return Datasets(adaptation_data, adaptation_labels, evaluation_data, evaluation_labels)

##############
# EXPERIMENT #
##############
class MAMLRobustnessExperiment(RobustnessExperiment):
    # TODO - discuss/lit.-rev: currently attacking all training samples (so support and query set... 
    # iirc in one of the papers they only attacked the support for testing. Maybe we should test the effects of different attack vectors)
    def __init__(self, model_fn: Callable[[], nn.Module], 
                 maml_fn: Callable[[nn.Module], MAML], 
                 optim_fn: Callable[[MAML], torch.optim.Optimizer],
                 loss_fn: Callable[[], nn.Module],
                 tasksets_fn: Callable[[], BenchmarkTasksets],
                 nr_of_meta_epochs: int, meta_batchsize: int,
                 nr_of_adaptation_steps: int,
                 ways: int, shots: int,
                 attack_fn: Callable[[nn.Module], Attack],
                 trades_alpha: float,
                 seed: int = None) -> None:
        super().__init__(seed)
        self.model_fn = model_fn
        self.maml_fn = maml_fn
        self.optim_fn = optim_fn
        self.loss_fn = loss_fn
        self.tasksets_fn = tasksets_fn

        self.nr_of_meta_epochs = nr_of_meta_epochs
        self.meta_batchsize = meta_batchsize
        self.nr_of_adaptation_steps = nr_of_adaptation_steps
        self.ways = ways
        self.shots = shots

        self.attack_fn = attack_fn
        self.trades_alpha = trades_alpha

        self.maml: MAML = None
        self.optim: torch.optim.Optimizer = None
        self.loss: nn.Module = None
        self.tasksets: BenchmarkTasksets = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _fast_adapt(self, datasets: Datasets, learner: MAML) -> Stats:
        X_support, y_support = datasets.X_support.to(self.device), datasets.y_support.to(self.device)
        X_query, y_query = datasets.X_query.to(self.device), datasets.y_query.to(self.device)
        
        # Adapt the model using the support set
        for step in range(self.nr_of_adaptation_steps):
            support_loss = self.loss(learner(X_support), y_support)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        predictions = learner(X_query)
        query_loss = self.loss(predictions, y_query)
        query_acc = accuracy(predictions, y_query)
        return Stats(query_loss, query_acc)

    def _adapt_nat(self, learner: MAML, X: torch.Tensor, y: torch.Tensor, training: bool) -> Stats:
        learner_nat = learner.clone() # clones the inner module but keeps the graph => can compute grad wrt original parameters
    
        if training:
            learner_nat.train()
        else: 
            learner_nat.eval()
        
        datasets = split_support_query(X, y, self.shots, self.ways)
        stats = self._fast_adapt(datasets, learner_nat)
        
        return stats
    
    def _adapt_adv(self, learner: MAML, X: torch.Tensor, y: torch.Tensor, training: bool) -> CombinedStats:
        learner_nat = learner.clone()
        learner_adv = learner.clone()

        if training:
            learner_nat.train()
            learner_adv.train()
        else:
            learner_nat.eval()
            learner_adv.eval()

        datasets = split_support_query(X, y, self.shots, self.ways)

        # compute adversarial examples in eval mode
        learner_adv.module.eval()
        X_support_adv = self.attack_fn(learner_adv.module)(datasets.X_support,datasets.y_support)
        X_query_adv = self.attack_fn(learner_adv.module)(datasets.X_support,datasets.y_support)
        datasets_adv = Datasets(X_support_adv, datasets.y_support, X_query_adv, datasets.y_query)
        if training:
            learner_adv.module.train()

        # adapt on both (note that we need two separate learners for this!)
        stats = self._fast_adapt(datasets, learner_nat)
        stats_adv = self._fast_adapt(datasets_adv, learner_adv)
    
        # mix the natural and adversarial loss to keep high accuracy
        trades_loss = (1.0 - self.trades_alpha) * stats.loss + self.trades_alpha * stats_adv.loss

        return CombinedStats(stats.loss, stats.accuracy, stats_adv.loss, stats_adv.accuracy, trades_loss)
    
    def train_nat(self):
        print('Starting Standard Training...')
        self.maml.to(self.device)

        for iteration in range(self.nr_of_meta_epochs):
            self.optim.zero_grad()

            meta_train_stats = Stats()
            meta_val_stats = Stats()

            for task in range(self.meta_batchsize): 
                # Compute meta-training loss
                X_train, y_train = self.tasksets.train.sample()
                train_stats = self._adapt_nat(self.maml, X_train, y_train, training=True)
                train_stats.loss.backward()
                meta_train_stats += train_stats

                # Compute meta-validation loss
                X_val, y_val = self.tasksets.validation.sample()
                val_stats = self._adapt_nat(self.maml, X_val, y_val, training=False)
                meta_val_stats += val_stats

            meta_train_stats /= self.meta_batchsize
            meta_val_stats /= self.meta_batchsize

            # Print some metrics
            print('Iteration', iteration)
            print(f'\t Meta Train Stats: {meta_train_stats}')
            print(f'\t Meta Val Stats: {meta_val_stats}')

            # Average the accumulated gradients and optimize
            for p in self.maml.parameters():
                p.grad.data.mul_(1.0 / self.meta_batchsize)
            self.optim.step()

    def train_adv(self):
        print('Starting Adversarial Training...')
        self.maml.to(self.device)

        for iteration in range(self.nr_of_meta_epochs):
            self.optim.zero_grad()

            meta_train_stats = CombinedStats()
            meta_val_stats = CombinedStats()

            for task in range(self.meta_batchsize): 
                # Compute adversarial meta-training loss
                X_train, y_train = self.tasksets.train.sample()
                train_stats = self._adapt_adv(self.maml, X_train, y_train, training=True)
                train_stats.trades_loss.backward()
                meta_train_stats += train_stats

                # Compute adversarial meta-validation loss
                X_val, y_val = self.tasksets.validation.sample()
                val_stats = self._adapt_adv(self.maml, X_val, y_val, training=False)
                meta_val_stats += val_stats

            meta_train_stats /= self.meta_batchsize
            meta_val_stats /= self.meta_batchsize

            # Print some metrics
            print('Iteration', iteration)
            print(f'\t Meta Train Stats: {meta_train_stats}')
            print(f'\t Meta Val Stats: {meta_val_stats}')

            # Average the accumulated gradients and optimize
            for p in self.maml.parameters():
                p.grad.data.mul_(1.0 / self.meta_batchsize)
            self.optim.step()

    def test_nat(self):
        meta_test_stats = Stats()
        for task in range(self.meta_batchsize):
            # Compute meta-testing loss
            X_test, y_test = self.tasksets.test.sample()
            test_stats = self._adapt_nat(self.maml, X_test, y_test, training=False)
            meta_test_stats += test_stats

        meta_test_stats /= self.meta_batchsize
        print(f'Meta Nat. Test Stats: {meta_test_stats}')

    def test_adv(self):
        meta_test_stats = CombinedStats()
        for task in range(self.meta_batchsize):
            # Compute adversarial meta-testing loss
            X_test, y_test = self.tasksets.test.sample()
            test_stats = self._adapt_adv(self.maml, X_test, y_test, training=False)
            meta_test_stats += test_stats

        meta_test_stats /= self.meta_batchsize
        print(f'Meta Adv. Test Stats: {meta_test_stats}')

    def reinitialize(self):
        """ Forgets everything the model has learned so far """
        model = self.model_fn()
        self.maml = self.maml_fn(model)
        self.optim = self.optim_fn(self.maml)
        self.loss = self.loss_fn()
        self.tasksets = self.tasksets_fn()

