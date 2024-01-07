import os
import click
import random
import numpy as np
import torch

from uncertainty.commands import *

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--scenario',
              help='The scenario you want to train (either omniglot or miniimagenet)')
@click.option('--epochs', default=100,
              help='How many epochs should be trained')
@click.option('--taskembsize', default=256,
              help='How large should the task-embedding space be?')
@click.option('--outdir', default='models',
              help='Where do you want the models to be saved')
def train_binary_classifiers(scenario, epochs, taskembsize, outdir):
    fix_seed()

    best_model_file = f'best-binary-classifiers-{taskembsize}.pt'
    last_model_file = f'last-binary-classifiers-{taskembsize}.pt'
    if scenario == 'omniglot':
        out_dir = os.path.join(outdir, 'omniglot')
        os.makedirs(out_dir, exist_ok=True)

        train_omniglot_binary(taskembsize, epochs, os.path.join(out_dir, best_model_file), os.path.join(out_dir, last_model_file))
        
    elif scenario == 'miniimagenet':
        out_dir = os.path.join(outdir, 'mini-imagenet')
        os.makedirs(out_dir, exist_ok=True)

        train_miniimagenet_binary(taskembsize, epochs, os.path.join(out_dir, best_model_file), os.path.join(out_dir, last_model_file))
    
@cli.command()
@click.option('--scenario',
              help='The scenario you want to train (either omniglot or miniimagenet)')
@click.option('--method', default='maml',
              help='Do you want to use maml or mean to compute the meta-embedding?')
@click.option('--taskembsize', default=256,
              help='How large should the task-embedding space be?')
@click.option('--modelfile', default='models/omniglot/best-binary-classifiers-256.pt',
              help='Where you want to load the model from?')
@click.option('--outdir', default='models',
              help='Where do you want the models to be saved')
def train_meta_embedding(scenario, method, taskembsize, modelfile, outdir):
    fix_seed()

    best_embedding_file = f'best-meta-embedding-{method}-{taskembsize}.pt'
    last_embedding_file = f'last-meta-embedding-{method}-{taskembsize}.pt'
    if scenario == 'omniglot':
        out_dir = os.path.join(outdir, 'omniglot')
        os.makedirs(out_dir, exist_ok=True)

        train_omniglot_meta_embedding(taskembsize, modelfile, method, os.path.join(out_dir, best_embedding_file), os.path.join(out_dir, last_embedding_file))
        
    elif scenario == 'miniimagenet':
        out_dir = os.path.join(outdir, 'mini-imagenet')
        os.makedirs(out_dir, exist_ok=True)

        train_miniimagenet_meta_embedding(taskembsize, modelfile, method, os.path.join(out_dir, best_embedding_file), os.path.join(out_dir, last_embedding_file))
    
    print('Created best and last meta-embedding file. Please note, that best and last do not refer to the model that was used but to the meta embedding that worked best and the last one (only relevant for maml)')
    
@cli.command()
@click.option('--scenario',
              help='The scenario you want to train (either omniglot or miniimagenet)')
@click.option('--taskembsize', default=256,
              help='How large was the task embedding during training?')
@click.option('--modelfile', default='models/omniglot/best-binary-classifiers-256.pt',
              help='Where you want to load the model from?')
def val_original(scenario, taskembsize, modelfile):
    fix_seed()

    # only supports omniglot
    if scenario == 'omniglot':
        val_omniglot_original(task_emb_size=taskembsize, model_file=modelfile, nr_of_samples_per_class=16)

    elif scenario == 'miniimagenet':
        raise NotImplementedError()

@cli.command()
@click.option('--scenario',
              help='The scenario you want to train (either omniglot or miniimagenet)')
@click.option('--taskembsize', default=256,
              help='How large was the task embedding during training?')
@click.option('--modelfile', default='models/omniglot/best-binary-classifiers-256.pt',
              help='Where you want to load the model from?')
@click.option('--metaembfile', default='models/omniglot/best-meta-embedding-mean-256.pt',
              help='Where you want to load the meta embedding from?')
@click.option('--shots', default=5,
              help='How many positive and negative examples should the model see for adaptation?')
@click.option('--adaptationsteps', default=5,
              help='How many gradient steps should be performed')
@click.option('--lr', default=0.1,
              help='The learning rate that should be used')
def val_binary_adaptability(scenario, taskembsize, modelfile, metaembfile, shots, adaptationsteps, lr):
    fix_seed()
    
    # only supports omniglot
    if scenario == 'omniglot':
        val_omniglot_binary_adaptability(task_emb_size=taskembsize, model_file=modelfile, meta_embedding_file=metaembfile, 
                                        nr_of_samples_per_class=16, nr_of_samples_per_adaptation=2*shots,
                                        adaptation_steps=adaptationsteps, fast_lr=lr)

    elif scenario == 'miniimagenet':
        raise NotImplementedError()

@cli.command()
@click.option('--scenario',
              help='The scenario you want to train (either omniglot or miniimagenet)')
@click.option('--taskembsize', default=256,
              help='How large was the task embedding during training?')
@click.option('--modelfile', default='models/omniglot/best-binary-classifiers-256.pt',
              help='Where you want to load the model from?')
@click.option('--metaembfile', default='models/omniglot/best-meta-embedding-mean-256.pt',
              help='Where you want to load the meta embedding from?')
@click.option('--ways', default=5,
              help='How many ways do you want the model to distinguish?')
@click.option('--shots', default=5,
              help='How many shots should the model be able to use to adapt each binary classifier?')
@click.option('--adaptationsteps', default=5,
              help='How many gradient steps should be performed')
@click.option('--lr', default=0.1,
              help='The learning rate that should be used')
@click.option('--iterations', default=100,
              help='How many different tasksets should be sampled and contribute towards the final score')
def val_fewshot(scenario, taskembsize, modelfile, metaembfile, ways, shots, adaptationsteps, lr, iterations):
    fix_seed()
    # only supports omniglot
    if scenario == 'omniglot':
        val_omniglot_fewshot(task_emb_size=taskembsize, model_file=modelfile, meta_embedding_file=metaembfile, 
                             iterations=iterations, ways=ways, shots=shots, 
                             adaptation_steps=adaptationsteps, fast_lr=lr)
    elif scenario == 'miniimagenet':
        raise NotImplementedError()
    
@cli.command()
@click.option('--scenario',
              help='The scenario you want to train (either omniglot or miniimagenet)')
@click.option('--taskembsize', default=256,
              help='How large was the task embedding during training?')
@click.option('--modelfile', default='models/omniglot/best-binary-classifiers-256.pt',
              help='Where you want to load the model from?')
@click.option('--metaembfile', default='models/omniglot/best-meta-embedding-mean-256.pt',
              help='Where you want to load the meta embedding from?')
@click.option('--ways', default=5,
              help='How many ways do you want the model to distinguish?')
@click.option('--shots', default=5,
              help='How many shots should the model be able to use to adapt each binary classifier?')
@click.option('--adaptationsteps', default=5,
              help='How many gradient steps should be performed')
@click.option('--lr', default=0.1,
              help='The learning rate that should be used')
@click.option('--noise', default=0.05,
              help='How much noise to apply to the adapted task embedding')
@click.option('--samples', default=5,
              help='How many samples to take from the task-embedding + noise distribution')
@click.option('--noisepublic', is_flag=True, default=False,
              help='Does the attacker know the noise that is going to be used during querying?')
@click.option('--pgdepsilon', default=2/255,
              help='Maximum perturbation of PGD attack')
@click.option('--pgdstepsize', default=2/255,
              help='Step size for PGD attack')
@click.option('--pgdsteps', default=20,
              help='How many samples PGD attack can take')
@click.option('--iterations', default=100,
              help='How many different tasksets should be sampled and contribute towards the final score')
def val_defense(scenario, taskembsize, modelfile, metaembfile, ways, shots, adaptationsteps, lr, noise, samples, noisepublic, pgdepsilon, pgdstepsize, pgdsteps, iterations):
    fix_seed()
    # only supports omniglot
    if scenario == 'omniglot':
        val_omniglot_defense(task_emb_size=taskembsize, model_file=modelfile, meta_embedding_file=metaembfile, 
                             iterations=iterations, ways=ways, shots=shots, 
                             adaptation_steps=adaptationsteps, fast_lr=lr,
                             noise=noise, samples=samples, is_noise_public=noisepublic,
                             pgd_epsilon=pgdepsilon, pgd_stepsize=pgdstepsize, pgd_steps=pgdsteps)
    elif scenario == 'miniimagenet':
        raise NotImplementedError()

if __name__ == '__main__':
    cli()