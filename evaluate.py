import click
from evaluation.evaluate import run_all_scenarios_for_method

from uncertainty.benchmark import Uncertainty

@click.command()
@click.option('--name', help='The name of the model/method you want to test.')
def run(name):
    """ Runs all tests for the specified model/method and reports the results """
    if name == "AQ":
        pass
    elif name == "uncertainty":
        # for practical reasons you can choose this smaller than 1000 because it takes ages to run it even on a GPU
        # will reduce the statistical significance though
        run_all_scenarios_for_method(Uncertainty(), num_tasks=1000)


if __name__ == '__main__':
    run()