import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD

from uncertainty.maml_util import MAML, fast_adapt
from uncertainty.model import NOTEModel
from uncertainty.data_util import NegativeSampleDataset, TrainValSplitDataset, FewShotDataset, binary_tasks_from_fewshot

def train_binary_classifiers(train_samples: NegativeSampleDataset, val_samples: NegativeSampleDataset, 
                             model: NOTEModel, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
                             best_model_file: str, last_model_file: str,
                             epochs: int, iterations_per_epoch: int, nr_of_classes_per_epoch: int = 32, nr_of_samples_per_class: int = 4,
                             device: str = 'cpu'):
    """ 
        Trains the binary classifiers of the NOTEModel on the given dataset.

        Note that since we cannot run multiple classifiers at the same time we instead resort to 
        aggregating the gradients over many different classes (since hnet parameters are shared).

        Parameters
            - TODO: your normal training parameters
            - iterations_per_epoch: Basically, how many batches to sample in each epoch
            - nr_of_classes_per_epoch: How many classes to sample from before doing one step (Note: we draw uniform with replacement). 
                                       This should be large enough s.t. the model does not collapse to a single class everytime.
            - nr_of_samples_per_class: How many positive and negative samples should be sampled from each class
    """
    print(f"Training Binary Classifiers")
    
    model.to(device)
    best_validation_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        # Training
        model.train()
        for iteration in range(iterations_per_epoch):
            optimizer.zero_grad()
            # randomly sample x times from a random class
            for i in range(nr_of_classes_per_epoch):
                X,y,class_id = train_samples.sample(nr_of_samples_per_class)
                X,y = X.to(device),y.to(device)

                out = F.sigmoid(model.forward_binary(X, class_id))
                loss = loss_fn(out, y)
                loss.backward()

                train_losses.append(loss.item())
                train_accs.append((torch.round(out) == y).sum().item() / y.shape[0])

                del X,y

            optimizer.step()

        print(f"\t Training: Loss = {np.array(train_losses).mean():.5f} ± {np.array(train_losses).std():.5f}, Acc = {np.array(train_accs).mean():.5f}")

        # Validation
        model.eval()
        with torch.no_grad():
            for iteration in range(iterations_per_epoch):
                for i in range(nr_of_classes_per_epoch):
                    X,y,class_id = val_samples.sample(nr_of_samples_per_class)
                    X,y = X.to(device),y.to(device)

                    out = F.sigmoid(model.forward_binary(X, class_id))
                    loss = loss_fn(out, y)

                    val_losses.append(loss.item())
                    val_accs.append((torch.round(out) == y).sum().item() / y.shape[0])

                    del X,y

        # Save the model
        current_val_loss = np.array(val_losses).mean()
        if current_val_loss < best_validation_loss:
            best_validation_loss = current_val_loss
            torch.save(model.state_dict(), best_model_file)
        torch.save(model.state_dict(), last_model_file)

        print(f"\t Validation: Loss = {current_val_loss:.5f} ± {np.array(val_losses).std():.5f}, Acc = {np.array(val_accs).mean():.5f}")
        
    print("Finished Training")


def compute_mean_meta_embedding(model: NOTEModel, best_embedding_file: str, last_embedding_file: str):
    """ 
        Prepares the ideal embedding that can be used for fast-adaptation during few-shot learning 
        by simply computing the mean over the existing embeddings.

        (ideal embedding in terms of euclidean distance but might be many gradient steps away from the ideal adapted embedding)
    """
    mean_embedding = model.task_representations.mean(dim=0)
    torch.save(mean_embedding, best_embedding_file)
    torch.save(mean_embedding, last_embedding_file)

def compute_maml_meta_embedding(model: NOTEModel, best_embedding_file: str, last_embedding_file: str,
                        train_classes: TrainValSplitDataset, val_classes: TrainValSplitDataset, meta_lr: float = 0.003, fast_lr: float = 0.5, 
                        meta_batch_size: int = 32, adaptation_steps: int = 2, num_iterations: int = 60000):
    """ 
        Prepares the ideal embedding that can be used for fast-adaptation during few-shot learning
        by using MAML to find a point in space from where we can adapt to all embeddings in only a few gradient steps.
    """
    # start of from mean embedding
    mean_embedding = model.task_representations.mean(dim=0)
    model.task_representations = nn.Parameter(mean_embedding.clone().detach().unsqueeze(0))

    maml = MAML(model, lr=fast_lr, first_order=False)
    opt = torch.optim.Adam(maml.parameters(), meta_lr) # note: maml.parameters() returns only the adaptable params of the model
    binary_loss = nn.BCELoss()

    train_cls_train = NegativeSampleDataset(train_classes.train)
    train_cls_val = NegativeSampleDataset(train_classes.val)

    val_cls_train = NegativeSampleDataset(val_classes.train)
    val_cls_val = NegativeSampleDataset(val_classes.val)

    best_val_loss = float('inf')
    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        for task in range(meta_batch_size): 
            # Meta-Training
            learner = maml.clone()
            X_support,y_support,class_id = train_cls_train.sample(2)
            X_query,y_query,_ = train_cls_val.sample(2, class_id) 
            query_loss, query_acc = fast_adapt(X_support, y_support, X_query, y_query, learner, binary_loss, adaptation_steps)
            query_loss.backward()
            meta_train_error += query_loss.item()
            meta_train_accuracy += query_acc.item()

            # Meta-Validation
            learner = maml.clone()
            X_support,y_support,class_id = val_cls_train.sample(2)
            X_query,y_query,_ = val_cls_val.sample(2, class_id) 
            query_loss, query_acc = fast_adapt(X_support, y_support, X_query, y_query, learner, binary_loss, adaptation_steps)
            meta_valid_error += query_loss.item()
            meta_valid_accuracy += query_acc.item()

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        current_val_loss = meta_valid_error / meta_batch_size
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.task_representations[0], best_embedding_file)
        torch.save(model.task_representations[0], last_embedding_file)

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

def val_original(model: NOTEModel, classes_dataset: TrainValSplitDataset, nr_of_samples_per_class: int, device: str = 'cpu'):
    val_ds = NegativeSampleDataset(classes_dataset.val)

    model.to(device)
    model.eval()

    class_accs = []
    with torch.no_grad():
        nr_of_classes = model.task_representations.shape[0]
        for class_id in range(nr_of_classes):
            if class_id % 10 == 0:
                print(f'Evaluating class {class_id} of {nr_of_classes}')

            X,y,_ = val_ds.sample(nr_of_samples_per_class // 2, class_id=class_id)
            X,y = X.to(device), y.to(device)

            out = F.sigmoid(model.forward_binary(X, class_id=class_id))

            class_accs.append((torch.round(out) == y).sum().item() / y.shape[0])

            del X,y
        
    print('Accuracy:')
    print(f'Mean over classes = {np.array(class_accs).mean()}')
    print(f'Standard deviation over classes = {np.array(class_accs).std()}')

def val_binary_adaptability(model: NOTEModel, meta_embedding_file: str, 
                            nr_of_classes: int, classes_dataset: TrainValSplitDataset, 
                            nr_of_samples_per_class: int, nr_of_samples_per_adaptation: int,
                            adaptation_steps: int, fast_lr: float,
                            device: str = 'cpu'):
    """ Adapts the binary classifiers to data from the train set and measures how well it performs on the validation set """
    train_ds = NegativeSampleDataset(classes_dataset.train)
    val_ds = NegativeSampleDataset(classes_dataset.val)

    model.to(device)
    loss = nn.BCELoss()

    class_accs = [] # mean accuracies of all classes
    for class_id in range(nr_of_classes):
        accs_of_class = []
        for i in range(nr_of_samples_per_class):
            prepare_model_for_adaptation(model, meta_embedding_file, 1, device)

            # adaptation
            model.train()
            X_support,y_support,_ = train_ds.sample(nr_of_samples_per_adaptation // 2, class_id=class_id)
            X_support,y_support = X_support.to(device), y_support.to(device)
            adapt_binary(model, X_support, y_support, loss, 
                adaptation_steps=adaptation_steps, 
                lr=fast_lr)

            del X_support, y_support

            # evaluation
            model.eval()
            X_query,y_query,_ = val_ds.sample(nr_of_samples_per_adaptation // 2, class_id=class_id)
            X_query,y_query = X_query.to(device), y_query.to(device)

            out = F.sigmoid(model.forward_binary(X_query, class_id=0)) # we are only using 1 embedding as it gets reset everytime

            accs_of_class.append((torch.round(out) == y_query).sum().item() / y_query.shape[0])

            del X_query, y_query
        
        class_accs.append(np.array(accs_of_class).mean())

    print('Accuracy of adapted model:')
    print(f'Mean over classes = {np.array(class_accs).mean()}')
    print(f'Standard deviation over classes = {np.array(class_accs).std()}')

def val_fewshot(model: NOTEModel, meta_embedding_file: str, 
                iterations: int, classes_dataset: TrainValSplitDataset, ways: int, shots: int,
                adaptation_steps: int, fast_lr: float,
                device: str = 'cpu'):
    """ Samples few-shot tasksets and constructs, constructs binary tasks to adapt the embeddings and evaluates model performance on the n-way problem """
    support_ds = FewShotDataset(classes_dataset.train, ways, shots, iterations)
    query_ds = FewShotDataset(classes_dataset.val, ways, shots, iterations)

    model.to(device)
    loss = nn.BCELoss()

    accs = []
    for i in range(iterations):
        prepare_model_for_adaptation(model, meta_embedding_file, ways, device)

        # adaptation
        model.train()
        X_support,y_support,class_ids = support_ds[i]
        X_support,y_support = torch.flatten(X_support, end_dim=1), torch.flatten(y_support, end_dim=1) # first dim is shots
        X_support,y_support = X_support.to(device), y_support.to(device)

        adapt(model, X_support, y_support, ways, loss, 
                adaptation_steps=adaptation_steps, lr=fast_lr, half_batch_size=shots*ways)

        del X_support, y_support

        # evaluation
        model.eval()
        X_query,y_query,_ = query_ds.sample(class_ids) # need to sample from the same classes as we did in the support set
        X_query,y_query = torch.flatten(X_query, end_dim=1), torch.flatten(y_query, end_dim=1) # first dim is shots
        X_query,y_query = X_query.to(device), y_query.to(device)

        out = model.forward(X_query, range(ways))

        accs.append((torch.argmax(out, dim=-1) == y_query).sum().item() / y_query.shape[0])

        del X_query, y_query
        
    print(f'{ways}-way accuracy of the adapted model:')
    print(f'Mean over classes = {np.array(accs).mean()}')
    print(f'Standard deviation over classes = {np.array(accs).std()}')

def val_defense(model: NOTEModel, meta_embedding_file: str, 
                iterations: int, classes_dataset: TrainValSplitDataset, ways: int, shots: int,
                adaptation_steps: int, fast_lr: float, is_noise_public: bool,
                pgd_epsilon: float, pgd_stepsize: float, pgd_steps: int,
                device: str = 'cpu'):
    """ Test how the adapted model performs against PGD attacker when defense is on """
    support_ds = FewShotDataset(classes_dataset.train, ways, shots, iterations)
    query_ds = FewShotDataset(classes_dataset.val, ways, shots, iterations)

    model.to(device)
    loss = nn.BCELoss()

    accs = []
    accs_adv = []
    for i in range(iterations):
        prepare_model_for_adaptation(model, meta_embedding_file, ways, device)

        if is_noise_public:
            # using the same noise for all forward passes
            model.fix_noise()

        # adaptation
        model.train()
        X_support,y_support,class_ids = support_ds[i]
        X_support,y_support = torch.flatten(X_support, end_dim=1), torch.flatten(y_support, end_dim=1) # first dim is shots
        X_support,y_support = X_support.to(device), y_support.to(device)

        adapt(model, X_support, y_support, ways, loss, 
              adaptation_steps=adaptation_steps, lr=fast_lr, half_batch_size=shots*ways)

        del X_support, y_support

        # evaluation
        model.eval()
        X_query,y_query,_ = query_ds.sample(class_ids) # need to sample from the same classes as we did in the support set
        X_query,y_query = torch.flatten(X_query, end_dim=1), torch.flatten(y_query, end_dim=1) # first dim is shots
        X_query,y_query = X_query.to(device), y_query.to(device)

        out = model.forward(X_query, range(ways))
        accs.append((torch.argmax(out, dim=-1) == y_query).sum().item() / y_query.shape[0])

        # TODO: test what happens if noise is kept constant during PGD attack instead of varying each step
        attack = PGD(model, pgd_epsilon, pgd_stepsize, pgd_steps)
        X_query_adv = attack(X_query, y_query)

        out_adv = model.forward(X_query_adv, range(ways))
        accs_adv.append((torch.argmax(out_adv, dim=-1) == y_query).sum().item() / y_query.shape[0])

        del X_query, y_query, X_query_adv
        
    print(f'{ways}-way accuracy of the adapted model:')
    print(f'Mean over classes = {np.array(accs).mean()}')
    print(f'Standard deviation over classes = {np.array(accs).std()}')

    print(f'{ways}-way adversarial accuracy of the adapted model:')
    print(f'Mean over classes = {np.array(accs_adv).mean()}')
    print(f'Standard deviation over classes = {np.array(accs_adv).std()}')


def prepare_model_for_adaptation(model: NOTEModel, meta_embedding_file: str, ways: int, device: str = 'cpu'):
    """ Prepares the model for adaptation, using the best meta-task-embeddings that were computed beforehand """
    meta_embedding: torch.Tensor = torch.load(meta_embedding_file, map_location=device)
    model.task_representations = nn.Parameter(meta_embedding.repeat(ways, 1))

def adapt_binary(model: NOTEModel, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module,
          adaptation_steps: float = 5, lr: float = 0.05, device: str = 'cpu'):
    """ 
        Adapts a classifier to the given X,y by doing a few gradient steps on the embedding.
        Assumes the model already is correctly setup with one meta-task-embedding.

        X,y are expected to represent binary classification task
    """

    # train all binary classifiers for nr of steps
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.adapt_parameters(), lr=lr)
        
    X,y = X.to(device), y.to(device)
    for i in range(adaptation_steps):
        optimizer.zero_grad()

        out = F.sigmoid(model.forward_binary(X, class_id=0))
        loss = loss_fn(out, y.float())
        loss.backward()

        optimizer.step()

    del X,y

def adapt(model: NOTEModel, X_support: torch.Tensor, y_support: torch.Tensor, ways: int, loss_fn: nn.Module,
          adaptation_steps: float = 5, lr: float = 0.05, half_batch_size: int = 1, device: str = 'cpu'):
    """ 
        Adapts the trained model & meta-embedding to a support set.
        Assumes the model already is correctly setup with the correct amount of meta-task-embeddings for n-way classification.

        X_support, y_support are assumed to be few-shot tasksets
    """
    assert model.task_representations.shape[0] == ways

    optimizer = torch.optim.SGD(model.adapt_parameters(), lr=lr)

    # train all binary classifiers for nr of steps
    model.to(device)
    model.train()
    for class_id in range(ways):
        for i in range(adaptation_steps):
            optimizer.zero_grad()

            X,y = binary_tasks_from_fewshot(X_support, y_support, class_id, half_batch_size)
            X,y = X.to(device), y.to(device)

            out = F.sigmoid(model.forward_binary(X, class_id))
            loss = loss_fn(out, y.float())
            loss.backward()

            optimizer.step()

            del X,y