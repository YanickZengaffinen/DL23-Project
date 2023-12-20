from typing import Dict, List
import torchmetrics


from evaluation.attacker import Attacker, PGDAttacker, Natural
from evaluation.datasets import TestDataset
from evaluation.model_wrapper import ModelWrapper

models: Dict[ModelWrapper] = {
    # Your model/method extends from ModelWrapper and goes here
    'Test': ModelWrapper(),
}

def run_all_scenarios_for_method(model_name: str, num_tasks: int = 1000):
    # Runs all the attack scenarios from the proposal on the specified model
    model = models[model_name]

    meta_scenarios = [
        { 'ways': 5, 'shots': 1 },
        { 'ways': 5, 'shots': 5 },
    ]

    pgd_attackers = [
        { 'steps': 20, 'epsilon': 2, 'alpha': 2 },
        { 'steps': 20, 'epsilon': 4, 'alpha': 2 },
        { 'steps': 20, 'epsilon': 8, 'alpha': 2 },
    ]

    def _report_metrics(ways, shots, dataset_name, attacker_name, metrics: Dict[str, torchmetrics.Metric]):
        print(f'{ways}-ways {shots}-shots on {dataset_name} with {attacker_name} attacker:')
        for metric_name, metric in metrics.items():
            print(f'\t {metric_name}: {metric.compute()}')

    for s in meta_scenarios:
        ways, shots = s['ways'], s['shots']

        # TODO: do we need 95% confidence intervals, if so => implement
        metrics = {
            'Accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=ways),
            '95%Confidence': None
        } # metrics can be used to compute the final metrics but only until they are reset
        

        # construct omniglot and mini-imagenet datasets for n-ways,k-shots scenario
        test_datasets = {
            'Omniglot': TestDataset('omniglot', ways, shots, num_tasks),
            'MiniImageNet': TestDataset('mini-imagenet', ways, shots, num_tasks)
        } 

        for dataset_name, test_dataset in test_datasets.items():
            # no attack / natural
            run_experiment(test_dataset, Natural(), model, metrics.values())
            _report_metrics(ways, shots, dataset_name, 'No', metrics)

            # pgd attacks
            for p in pgd_attackers:
                attacker = PGDAttacker(model, p['steps'], p['epsilon'], p['alpha'])
                run_experiment(test_dataset, attacker, model, metrics.values())
                _report_metrics(ways, shots, dataset_name, f'PGD(steps={attacker._steps},epsilon={attacker._epsilon},alpha={attacker._alpha})', metrics)
        


def run_experiment(test_dataset: TestDataset, attacker: Attacker, model: ModelWrapper, metrics: List[torchmetrics.Metric]):
    # Here we trust the impl of the ModelWrapper not to make use of the following:
    # 1. The gradient information still present in the model, that could be left over from the attack
    # 2. The fact that we always attack the support data
    # If any of these points is violated, this method needs to be rewritten!

    # TODO: check if 1) leads to an unfair advantage for any of our methods

    # reset all metrics
    for metric in metrics:
        metric.reset()

    for i in range(test_dataset.num_tasks):
        (support_x, support_y), (query_x, query_y) = test_dataset.sample()
        # make sure the model's state is the way it was directly after training
        model.reset_model()

        # adapt the model to the support set
        support_x_attacked = attacker.attack(support_x, support_y)
        model.adapt(support_x_attacked, support_y)

        # predict on the query set
        query_x_attacked = attacker.attack(query_x, query_y)
        query_y_attacked = model.forward(query_x_attacked)

        metrics.update(query_y_attacked, query_y)


    


