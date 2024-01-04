import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from uncertainty.model import get_omniglot_model
from uncertainty.data_util import NegativeSampleDataset, get_omniglot_datasets, get_miniimagenet_datasets
from uncertainty.uncertainty_util import *

def train_omniglot_binary(task_emb_size: int, epochs: int, best_model_file: str, last_model_file: str):
    """ Trains the binary classifiers and saves two models (the one with the best validation loss and the last one) """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    datasets = get_omniglot_datasets()
    train_ds = NegativeSampleDataset(datasets.train_classes.train)
    val_ds = NegativeSampleDataset(datasets.train_classes.val)

    model = get_omniglot_model(nr_of_classes=1100, task_emb_size=task_emb_size, softmax_temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCELoss()

    train_binary_classifiers(
        train_samples=train_ds,
        val_samples=val_ds,
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        best_model_file=best_model_file,
        last_model_file=last_model_file,
        epochs=epochs,
        iterations_per_epoch=32,
        nr_of_classes_per_epoch=32,
        nr_of_samples_per_class=4,
        device=device
    )

def train_omniglot_meta_embedding(task_emb_size: int, trained_model_file: str, method: str, 
                             best_embedding_file: str, last_embedding_file: str):
    """ 
        Prepares an embedding that allows for easy adaptation to any task. 
        Requires the binary classifiers to be trained and all task embeddings for the training classes to be available.

        Parameters:
            - method:
                - mean: computes the mean over existing embeddings
                - maml: uses maml to find an easy-to-adapt-from embedding
    """

    model = get_omniglot_model(nr_of_classes=1100, task_emb_size=task_emb_size, softmax_temperature=0.1)
    model.load_state_dict(torch.load(trained_model_file))

    if method == 'mean':
        compute_mean_meta_embedding(model, best_embedding_file, last_embedding_file)
    elif method == 'maml':
        datasets = get_omniglot_datasets()    
        # probably want to use the same adaptation parameters as during inference here
        compute_maml_meta_embedding(model, best_embedding_file, last_embedding_file, 
                                    train_classes=datasets.train_classes,
                                    val_classes=datasets.val_classes,
                                    meta_lr=0.003, fast_lr=0.1, meta_batch_size=32, adaptation_steps=5, num_iterations=60000)

def val_omniglot_original(task_emb_size: int, model_file: str, nr_of_samples_per_class: int = 16):
    """ 
        Measure how well the binary classifiers perform on the learned classes.
        Will report the mean accuracy as well as the standard deviation over classes.
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'

    datasets = get_omniglot_datasets()

    model = get_omniglot_model(nr_of_classes=1100, task_emb_size=task_emb_size, softmax_temperature=0.1)
    model.load_state_dict(torch.load(model_file, map_location=device))
    
    val_original(model, datasets.train_classes, nr_of_samples_per_class, device)


def val_omniglot_binary_adaptability(task_emb_size: int, model_file: str, meta_embedding_file: str, 
                                     adaptation_steps: int = 5, fast_lr: float = 0.05, 
                                     nr_of_samples_per_class: int = 8, nr_of_samples_per_adaptation: int = 6):
    """ 
        Measure how well the binary models can adapt to seen classes vs unseen classes using a specific meta-embedding
    
        Parameters:
            - task_emb_size: the embedding size that was used during training
            - model_file: where to load the trained model weights from
            - meta_embedding_file: path to the file that holds the computed meta-embedding
            - adaptation_steps: how many steps should be taken during adaptation
            - fast_lr: size of the step-size during adaptation
            - nr_of_samples_per_class: how many measurements should be made for each class
            - nr_of_samples_per_adaptation: how many positive + negative samples should be drawn for each adaptation
                e.g. 6 will result in 3 positive examples and 3 negative examples
    """
    # TODO: potentially also report average gradients on the embedding
    device='cuda' if torch.cuda.is_available() else 'cpu'

    datasets = get_omniglot_datasets()
    model = get_omniglot_model(nr_of_classes=1100, task_emb_size=task_emb_size, softmax_temperature=0.1)
    model.load_state_dict(torch.load(model_file, map_location=device))

    # performance when adapting to seen classes
    print('Evaluating seen classes...')
    # we just look at the first 100 classes of the task-embeddings to speed things up (+ val only has 100 classes too anyway)
    val_binary_adaptability(model, meta_embedding_file, 100, datasets.train_classes, 
                            nr_of_samples_per_class, nr_of_samples_per_adaptation, 
                            adaptation_steps, fast_lr, device)

    # performance when adapting to unseen classes
    print('Evaluating unseen classes...')
    val_binary_adaptability(model, meta_embedding_file, 100, datasets.val_classes, 
                            nr_of_samples_per_class, nr_of_samples_per_adaptation, 
                            adaptation_steps, fast_lr, device)
    

def val_omniglot_fewshot(task_emb_size: int, model_file: str, meta_embedding_file: str, 
                         iterations: int, ways: int, shots: int,
                         adaptation_steps: int = 5, fast_lr: float = 0.05):
    """ 
        Adapts the binary classifiers to few-shot tasksets and measures how well they perform as an n-way classifer
    
        Parameters:
            - task_emb_size: the task embedding size that was used during training
            - model_file: where to load the trained model weights from
            - meta_embedding_file: where to load the computed meta-embedding from
            - iterations: how many different tasksets should be evaluated on
            - ways: nr of classes that should be distinguished
            - shot: nr of examples provided for each class
            - adaptation_steps: nr of steps taken during adaptation
            - fast_lr: step_size during adaptation
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'

    datasets = get_omniglot_datasets()
    model = get_omniglot_model(nr_of_classes=1100, task_emb_size=task_emb_size, softmax_temperature=0.1)
    model.load_state_dict(torch.load(model_file, map_location=device))

    val_fewshot(model, meta_embedding_file, iterations, datasets.val_classes, ways, shots, adaptation_steps, fast_lr, device)

def val_omniglot_defense(task_emb_size: int, model_file: str, meta_embedding_file: str, 
                         iterations: int, ways: int, shots: int,
                         adaptation_steps: int = 5, fast_lr: float = 0.05, 
                         noise: float = 0.05, samples: int = 5, is_noise_public: bool = False,
                         pgd_epsilon: float = 2/255, pgd_stepsize: float = 2/255, pgd_steps: int = 20):
    """ 
        Adapts the binary classifiers to few-shot tasksets and measures how well they perform as an n-way classifer 
        when subjected to adversarial attacks but using uncertainty as defense mechanism
    
        Parameters:
            - task_emb_size: the task embedding size that was used during training
            - model_file: where to load the trained model weights from
            - meta_embedding_file: where to load the computed meta-embedding from
            - iterations: how many different tasksets should be evaluated on
            - ways: nr of classes that should be distinguished
            - shot: nr of examples provided for each class
            - adaptation_steps: nr of steps taken during adaptation
            - fast_lr: step_size during adaptation
            - noise: the uncertainty/noise that is applied to the task-embedding
            - is_noise_public: does the attacker know the noise that is going to be sampled by the model during adaptation
            - samples: how many samples to approximate the distribution over outputs
            - pgd_epsilon: maximum perturbation PGD attack is allowed to do
            - pgd_stepsize: stepsize during PGD attack
            - pgd_steps: nr of steps PGD attack performs
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'

    datasets = get_omniglot_datasets()
    model = get_omniglot_model(nr_of_classes=1100, task_emb_size=task_emb_size, softmax_temperature=0.1)
    model.load_state_dict(torch.load(model_file, map_location=device))

    model.enable_adversarial_protection(noise, samples)

    val_defense(model, meta_embedding_file, iterations, datasets.val_classes, ways, shots, adaptation_steps, fast_lr, is_noise_public, pgd_epsilon, pgd_stepsize, pgd_steps, device)

def train_miniimagenet():
    raise NotImplementedError()