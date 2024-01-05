import torch
import torch.nn as nn
import learn2learn as l2l
from torchattacks.attacks.pgd import PGD

from robustness.maml import MAMLRobustnessExperiment

# attacks maml with pgd
# TODO: interestingly, during standard training, adversarial accuracy is higher than natural accuracy which seems weird... even when mixing the gradients to add up to one in TRADES... look at this again
# TODO: add some other experiments
ways = 5
shots = 1
fast_lr = 0.5
meta_lr = 0.003
nr_of_meta_epochs = 50

def get_omniglot_tasksets():
    return l2l.vision.benchmarks.get_tasksets('omniglot', 
        train_ways=ways, train_samples=2*shots, # 2x since half of the data will be used to validate on the inner loop
        test_ways=ways, test_samples=2*shots,
        num_tasks=20000, root='~/data')
    
experiment = MAMLRobustnessExperiment(
    model_fn=lambda: l2l.vision.models.OmniglotFC(28 ** 2, ways),
    maml_fn=lambda model: l2l.algorithms.MAML(model, lr=fast_lr, first_order=False),
    optim_fn=lambda maml: torch.optim.Adam(maml.parameters(), meta_lr),
    loss_fn=lambda: nn.CrossEntropyLoss(reduction='mean'),
    tasksets_fn=get_omniglot_tasksets,
    nr_of_meta_epochs=nr_of_meta_epochs,
    meta_batchsize=32,
    nr_of_adaptation_steps=1,
    ways=ways,
    shots=shots,
    attack_fn=lambda model: PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True),
    trades_alpha=0.5,
    seed=14
)

experiment.run()