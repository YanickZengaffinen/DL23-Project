import click
from evaluation.evaluate import run_all_scenarios_for_method

from uncertainty.benchmark import Uncertainty
from adversarial_querying.aq_baseline import AQBaseline

@click.command()
@click.option('--name', help='The name of the model/method you want to test.')
@click.option('--add_noise', default=False, help='Whether to add noise to the AQ model or not.')
def run(name, add_noise):
    """ Runs all tests for the specified model/method and reports the results """
    if name == "AQ":
        # for practical reasons you can choose this smaller than 1000 because it takes ages to run it even on a GPU
        # will reduce the statistical significance though
        run_all_scenarios_for_method(AQBaseline(add_noise=add_noise), num_tasks=1000)
    elif name == "uncertainty":
        # for practical reasons you can choose this smaller than 1000 because it takes ages to run it even on a GPU
        # will reduce the statistical significance though
        run_all_scenarios_for_method(Uncertainty(), num_tasks=1000)


if __name__ == '__main__':
    run()