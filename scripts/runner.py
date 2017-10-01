import click
import numpy as np
import os
import pickle

from joblib import Parallel, delayed
from multiprocessing import cpu_count

from experiments.within_box import BoxedCavesExperiment


def _setup_box_experiment(n_per_cave, n_caves,
                          n_iterations=2000, percolation_limit=True,
                          output_dir='output'):

    def _run_box_experiment(param_pair):

        e = BoxedCavesExperiment(n_per_cave, n_caves, param_pair[0])
        e.setup(percolation_limit=percolation_limit)
        e.iterate(n_iterations, noise_level=param_pair[1])

        pp = param_pair
        save_path = os.path.join(
            output_dir, 'box-experiment_bw={}_nl={}'.format(pp[0], pp[1])
        )
        pickle.dumps(e, open(save_path, 'wb'))

    return _run_box_experiment


def _box_experiment_5_20(param_pair, output_path='output'):

    e = BoxedCavesExperiment(5, 20, param_pair[0])
    e.setup(percolation_limit=True)
    e.iterate(1000, noise_level=param_pair[1])

    pp = param_pair
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    save_path = os.path.join(
        output_path, 'box-experiment_bw={:.2f}_nl={:.2f}'.format(pp[0], pp[1])
    )
    pickle.dump(e, open(save_path, 'wb'))


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = dict()


@cli.command()
@click.option('--n-trials', default=10,
              help='Number of trials per parameter pair')
@click.option('--output-dir', default='output',
              help='Save location for pickles')
@click.option('--boxwidth-min', default=0.05, help='Minimum box width to test')
@click.option('--boxwidth-max', default=1.0, help='Maximum box width to test')
@click.option('--boxwidth-step', default=0.45, help='Box width step')
@click.option('--noise-level-min', default=0.0, help='Minimum noise level')
@click.option('--noise-level-max', default=1.0, help='Maximum noise level')
@click.option('--noise-level-step', default=0.5, help='Noise level step')
@click.option('--n-caves', default=20, help='Number of caves')
@click.option('--n-per-cave', default=5, help='Agents per cave')
@click.pass_context
def box_experiment(
            ctx, n_trials, output_dir,
            boxwidth_min, boxwidth_max, boxwidth_step,
            noise_level_min, noise_level_max, noise_level_step,
            n_caves, n_per_cave
        ):
    "Run the box experiment"

    boxwidths = np.arange(boxwidth_min, boxwidth_max, boxwidth_step)
    noise_levels = np.arange(
        noise_level_min, noise_level_max, noise_level_step
    )

    param_pairs = [(bw, nl) for bw in boxwidths for nl in noise_levels]

    # run_fun = _setup_box_experiment(n_per_cave, n_caves)

    Parallel(n_jobs=cpu_count())(
        delayed(_box_experiment_5_20)(p, output_dir)
        for p in param_pairs
    )
