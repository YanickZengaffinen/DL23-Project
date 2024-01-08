import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import learn2learn as l2l
import torchattacks

from datasets import get_benchmark_tasksets, BenchmarkTasksets
from models import ResNet12

# tensorboard
from torch.utils.tensorboard import SummaryWriter


class AdversarialQuerying:
    def __init__(
        self,
        model_fn: callable,
        ways: int,
        shots: int,
        meta_lr: float,
        fast_lr: float,
        meta_batch_size: int,
        adaptation_steps: int,
        num_iterations: int,
        cuda: bool,
        seed: int,
        epsilon: float,
        alpha: float,
        steps: int,
        taskset_name: str,
        num_tasks: int,
    ):
        self.model_fn = model_fn
        self.ways = ways
        self.shots = shots
        self.meta_lr = meta_lr
        self.fast_lr = fast_lr
        self.meta_batch_size = meta_batch_size
        self.adaptation_steps = adaptation_steps
        self.num_iterations = num_iterations
        self.cuda = cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.seed = seed
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.taskset_name = taskset_name
        self.num_tasks = num_tasks

        self._trained_model = None

    def save_model(self, path: str) -> None:
        """Saves the model to the given path."""
        if self._trained_model is None:
            raise ValueError("Model is not trained yet!")
        
        path = os.path.realpath(path)
        print(f"Saving model to {path}")
        torch.save(self._trained_model.state_dict(), path)

    @staticmethod
    def load_model(model: torch.nn.Module, path: str) -> None:
        """Loads the model from the given path."""
        model.load_state_dict(torch.load(path))

    @staticmethod
    def _accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the accuracy for the given predictions and targets.
        Args:
            predictions (torch.Tensor): Predictions of the model.
            targets (torch.Tensor): Targets.
        Returns:
            torch.Tensor: Accuracy value.
        """
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def _get_tasksets(self) -> BenchmarkTasksets:
        """Returns taskset containing train/validation/test splits"""
        # check if dataset is omniglot or mini-imagenet
        assert self.taskset_name in [
            "omniglot",
            "mini-imagenet",
        ], "Dataset not supported! Use either omniglot or mini-imagenet."

        taskset = get_benchmark_tasksets(
            name=self.taskset_name,
            ways=self.ways,
            shots=self.shots,
            num_tasks=self.num_tasks,
        )

        return BenchmarkTasksets(
            train=taskset.train.taskset,
            validation=taskset.validation.taskset,
            test=taskset.test.taskset,
        )

    def _create_adversarial_examples(
        self, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, epsilon: float = None
    ) -> torch.Tensor:
        """Creates adversarial examples using the given model and inputs.
        Args:
            model (torch.nn.Module): Model to be used.
            inputs (torch.Tensor): Inputs.
            labels (torch.Tensor): Labels.
            epsilon (float, optional): Epsilon value. Default to use self.epsilon.
        Returns:
            torch.Tensor: Adversarial examples.
        """
        if epsilon is None: # Use default epsilon value if not specified
            epsilon = self.epsilon
        elif epsilon == 0:  # Standard training
            return inputs
        else:  # Adversarial training
            adversary = torchattacks.PGD(
                model, eps=epsilon, alpha=self.alpha, steps=self.steps
            )
            adv_inputs = adversary(inputs, labels)
            return adv_inputs

    def _fast_adapt_aq(
        self, batch: tuple, learner: torch.nn.Module, loss: torch.nn.Module, epsilon: float = None
    ) -> tuple:
        """Adapts the model using the given batch for the given number of adaptation steps.
        Args:
            batch (tuple): Batch of data.
            learner (torch.nn.Module): Model to be adapted.
            loss (torch.nn.Module): Loss function.
        Returns:
            tuple: Validation error and accuracy.
        """

        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = (
            data[adaptation_indices],
            labels[adaptation_indices],
        )
        evaluation_data, evaluation_labels = (
            data[evaluation_indices],
            labels[evaluation_indices],
        )

        for step in range(self.adaptation_steps):
            adaptation_data_adv = self._create_adversarial_examples(
                learner, adaptation_data, adaptation_labels, epsilon=epsilon
            )
            if self.epsilon == 0:  # Standard training
                train_error = loss(learner(adaptation_data), adaptation_labels)
            else:  # Adversarial training
                train_error = loss(learner(adaptation_data), adaptation_labels) + loss(
                    learner(adaptation_data_adv), adaptation_labels
                )
            learner.adapt(train_error)

        # Create adversarial examples for evaluation data. If epsilon is 0, then this will return the original evaluation data
        evaluation_data_adv = self._create_adversarial_examples(
            learner, evaluation_data, evaluation_labels, epsilon=epsilon
        )

        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_accuracy = self._accuracy(predictions, evaluation_labels)
        predictions_adv = learner(evaluation_data_adv)
        valid_error_adv = loss(predictions_adv, evaluation_labels)

        return valid_error + valid_error_adv, valid_accuracy

    def train(self):
        assert self.model_fn is not None, "Please provide a model_fn to create the model"

        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        device = torch.device("cpu")
        if cuda:
            torch.cuda.manual_seed(self.seed)
            device = torch.device("cuda")

        tasksets = self._get_tasksets()
        
        print(f"Training on {self.taskset_name} dataset with {self.ways}-way {self.shots}-shot tasks")
        model = model_fn()  # Model to be used
        model.to(device)
        maml = l2l.algorithms.MAML(
            model, lr=fast_lr, first_order=False
        )  # Use learn2learn's MAML model
        # print(maml)
        opt = torch.optim.Adam(maml.parameters(), meta_lr)
        loss = nn.CrossEntropyLoss(reduction="mean")

        validation_accuracies = []  # List to store validation accuracies

        # Meta-training loop
        for iteration in range(self.num_iterations):
            # Slowly increase the epsilon value from 0 to epsilon until halfway through the training
            epsilon_value = min(self.epsilon * (iteration / (self.num_iterations / 2)), self.epsilon)

            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone()
                batch = tasksets.train.sample()
                evaluation_error, evaluation_accuracy = self._fast_adapt_aq(
                    batch,
                    learner,
                    loss,
                    epsilon=epsilon_value,
                )
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                batch = tasksets.validation.sample()
                evaluation_error, evaluation_accuracy = self._fast_adapt_aq(
                    batch,
                    learner,
                    loss,
                    epsilon=epsilon_value,
                )
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            avg_valid_accuracy = meta_valid_accuracy / self.meta_batch_size
            validation_accuracies.append(avg_valid_accuracy)

            print(f"\nIteration {iteration}/{self.num_iterations-1}")
            print("Epsilon", epsilon_value)
            print("Meta Train Error", meta_train_error / self.meta_batch_size)
            print("Meta Train Accuracy", meta_train_accuracy / self.meta_batch_size)
            print("Meta Valid Error", meta_valid_error / self.meta_batch_size)
            print("Meta Valid Accuracy", avg_valid_accuracy)

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            opt.step()


        # Compute the meta-testing loss
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.meta_batch_size):
            learner = maml.clone()
            batch = tasksets.test.sample()
            evaluation_error, evaluation_accuracy = self._fast_adapt_aq(
                batch,
                learner,
                loss,
                epsilon=epsilon_value,
            )
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
        print("Meta Test Error", meta_test_error / self.meta_batch_size)
        print("Meta Test Accuracy", meta_test_accuracy / self.meta_batch_size)

        # Store the trained model
        self._trained_model = maml.module
        # return trained model
        return maml


if __name__ == "__main__":
    # Dataset to use, either "omniglot" or "mini-imagenet"
    dataset: str = "omniglot"
    num_tasks: int = 10000

    # MAML parameters
    ways: int = 5
    shots: int = 1
    meta_lr: float = 0.003
    fast_lr: float = 0.5
    meta_batch_size: int = 32
    adaptation_steps: int = 1
    num_iterations: int = 100

    # CUDA parameters
    cuda: bool = False

    # Adversarial Querying parameters, PGD attack
    epsilon: float = 8/255
    alpha: float = 2/255
    steps: int = 20

    if dataset == "omniglot":
        # model_fn = lambda: l2l.vision.models.OmniglotFC(28 ** 2, ways)
        model_fn = lambda: ResNet12(output_size=ways, hidden_size=64, channels=1, dropblock_dropout=0, avg_pool=False)
    elif dataset == "mini-imagenet":
        # model_fn = lambda: l2l.vision.models.MiniImagenetCNN(output_size=ways)
        model_fn = lambda: ResNet12(output_size=ways, hidden_size=128)
    else:
        raise NotImplementedError(
            "Dataset not supported! Use either omniglot or mini-imagenet."
        )

    trainer = AdversarialQuerying(
        model_fn=model_fn,
        ways=ways,
        shots=shots,
        meta_lr=meta_lr,
        fast_lr=fast_lr,
        meta_batch_size=meta_batch_size,
        adaptation_steps=adaptation_steps,
        num_iterations=num_iterations,
        cuda=cuda,
        seed=42,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        taskset_name=dataset,
        num_tasks=num_tasks,
    )
    trainer.train()
    trainer.save_model(f'adversarial_querying/models/aq_model_{dataset}_{ways}w{shots}s.pth')
