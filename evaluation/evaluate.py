from typing import Dict, List

import torch.nn.functional as F
import torchmetrics

from evaluation.attacker import Attacker, PGDAttacker, Natural
from datasets import get_benchmark_tasksets, TasksetWrapper
from evaluation.model_wrapper import ModelWrapper
from evaluation.metrics import ConfidenceIntervalOnAccuracy

def run_all_scenarios_for_method(model: ModelWrapper, num_tasks: int = 1000, seed: int = 42):
    """ 
        Runs all the attack scenarios from the proposal on the specified model
    """

    meta_scenarios = [
        { 'ways': 5, 'shots': 1 },
        { 'ways': 5, 'shots': 5 },
    ]

    pgd_attackers = [
        { 'steps': 20, 'epsilon': 2/255, 'alpha': 2/255 },
        { 'steps': 20, 'epsilon': 4/255, 'alpha': 2/255 },
        { 'steps': 20, 'epsilon': 8/255, 'alpha': 2/255 },
    ]

    def _report_metrics(ways, shots, dataset_name, attacker_name, metrics: Dict[str, torchmetrics.Metric]):
        print(f'{ways}-ways {shots}-shots on {dataset_name} with {attacker_name} attacker:')
        for metric_name, metric in metrics.items():
            print(f'\t {metric_name}: {metric.compute()}')

    for s in meta_scenarios:
        ways, shots = s['ways'], s['shots']

        # TODO: do we need 95% confidence intervals, if so => implement
        metrics = {
            'Mean (%) and 95% Confidence Interval': ConfidenceIntervalOnAccuracy(interval=0.95),
            #'Accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=ways),
        } # metrics can be used to compute the final metrics but only until they are reset
        

        # construct omniglot and mini-imagenet datasets for n-ways,k-shots scenario
        test_datasets = {
            'Omniglot': get_benchmark_tasksets('omniglot', ways, shots, num_tasks, seed=seed).test,
            'MiniImageNet': get_benchmark_tasksets('mini-imagenet', ways, shots, num_tasks, seed=seed).test,
        } 

        for dataset_name, test_dataset in test_datasets.items():
            model.init_model(dataset_name, ways, shots)

            # no attack / natural
            run_experiment(test_dataset, Natural(), model, metrics.values())
            _report_metrics(ways, shots, dataset_name, 'No', metrics)

            # pgd attacks
            for p in pgd_attackers:
                attacker = PGDAttacker(model, p['steps'], p['epsilon'], p['alpha'])
                run_experiment(test_dataset, attacker, model, metrics.values())
                _report_metrics(ways, shots, dataset_name, f'PGD(steps={attacker._steps},epsilon={attacker._epsilon},alpha={attacker._alpha})', metrics)
        


def run_experiment(test_dataset: TasksetWrapper, attacker: Attacker, model: ModelWrapper, metrics: List[torchmetrics.Metric]):
    """
        Evaluates the model on in a given attack scenario on some dataset
    """
    # Here we trust the impl of the ModelWrapper not to make use of the following:
    # 1. The gradient information still present in the model, that could be left over from the attack
    # 2. The fact that we always attack the support data
    # If any of these points is violated, this method needs to be rewritten!

    # TODO: check if 1) leads to an unfair advantage for any of our methods

    # reset all metrics
    for metric in metrics:
        metric.reset()

    for i in range(test_dataset.num_tasks):
        (support_x, support_lbls), (query_x, query_lbls) = test_dataset.sample()
        support_y = F.one_hot(support_lbls, test_dataset.ways).float()

        # make sure the model's state is the way it was directly after training
        model.reset_model()

        # adapt the model to the support set
        model.adapt(support_x, support_y)

        # predict on the query set
        query_x_attacked = attacker.attack(query_x, query_lbls)
        query_y_attacked = model.forward(query_x_attacked)

        for metric in metrics:
            metric.update(query_y_attacked, query_lbls)


    


