import click
from evaluation.evaluate import run_all_scenarios_for_method

from baselines.sample_baseline import SampleBaseline

@click.command()
@click.option('--name', help='The name of the model/method you want to test.')
def run(name):
    """ Runs all tests for the specified model/method and reports the results """
    if name == "sample":
        run_all_scenarios_for_method(SampleBaseline(), num_tasks=1000)
    elif name == "AQ":
        pass
    # extend here


if __name__ == '__main__':
    run()