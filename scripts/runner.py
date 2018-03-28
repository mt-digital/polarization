import click
import multiprocessing as mp
import numpy as np
import os
import pickle

from joblib import Parallel, delayed
from functools import partial

from experiments.within_box import BoxedCavesExperiment




def _run_exp(_, experiment='connected caveman', n_caves=20, n_per_cave=5,
             K=2, n_iter=200, verbose=True):

    # Add the same number random short-range or long-range ties.
    n_edges = 20

    if experiment == 'connected caveman':
        cc = BoxedCavesExperiment(5, 20, 1.0, K=K)
        cc.iterate(n_iter, verbose=verbose)
        return cc

    elif experiment == 'random short-range':
        # Connected caveman with short-range ties added randomly.
        ccsrt = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        # ccsrt.iterate(2000, verbose=False)
        # ccsrt.add_shortrange_random_edges(n_edges)
        # ccsrt.iterate(n_iter - 2000, verbose=False)
        ccsrt.iterate(n_iter - 100, verbose=verbose)
        ccsrt.add_shortrange_random_edges(n_edges)
        ccsrt.iterate(100, verbose=verbose)
        return ccsrt

    elif experiment == 'random any-range':
        # Connected caveman with any-range ties added randomly.
        ccrt = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        # ccrt.iterate(2000, verbose=False)
        # ccrt.add_random_edges(n_edges)
        # ccrt.iterate(n_iter - 2000, verbose=False)
        ccrt.iterate(n_iter - 100, verbose=verbose)
        ccrt.add_random_edges(n_edges)
        ccrt.iterate(100, verbose=verbose)
        return ccrt

    else:
        raise RuntimeError(
            'experiment type ' + experiment + ' not recognized.'
        )


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = dict()


@cli.command()
@click.pass_context
def reproduce_fig11(ctx, n_per_cave, save_dir, n_trials=5):
    '''
    Create a set of HDF files corresponding to datasets for reproducing Figure 11b in FM2011
    '''
    K = 2

    pool = mp.Pool(mp.cpu_count())

    func = partial(_run_exp,
                   experiment='connected caveman', n_per_cave=n_per_cave)
    cc_experiments = pool.imap(func, range(n_trials))
    print('completed connected caveman')

    func = partial(_run_exp,
                   experiment='connected caveman', n_per_cave=n_per_cave)
    srt_experiments = pool.imap(func, range(n_trials))
    print('completed short-range')

    func = partial(_run_exp,
                   experiment='connected caveman', n_per_cave=n_per_cave)
    any_experiments = pool.imap(func, range(n_trials))
    print('completed any-range')

    experiments = {
        'connected caveman': list(cc_experiments),
        'random short-range': list(srt_experiments),
        'random any-range': list(any_experiments)
    }

    persist_experiments(
        experiments,
        directory=save_dir,
        hdf_filename='multiprocessing_persist.hdf5',
        metadata={'K': 2, 'n_per_cave': n_per_cave}
    )


@cli.command()
@click.pass_context
def reproduce_fig12(ctx, K, n_trials=5):
    '''
    Create a set of HDF files corresponding to datasets for reproducing Figure 12b in FM2011
    '''
    n_per_cave = 5

# @cli.command()
# @click.option('--n-trials', default=10,
#               help='Number of trials per parameter pair')
# @click.option('--output-dir', default='output',
#               help='Save location for pickles')
# @click.option('--boxwidth-min', default=0.05, help='Minimum box width to test')
# @click.option('--boxwidth-max', default=1.0, help='Maximum box width to test')
# @click.option('--boxwidth-step', default=0.45, help='Box width step')
# @click.option('--noise-level-min', default=0.0, help='Minimum noise level')
# @click.option('--noise-level-max', default=1.0, help='Maximum noise level')
# @click.option('--noise-level-step', default=0.5, help='Noise level step')
# @click.option('--n-caves', default=20, help='Number of caves')
# @click.option('--n-per-cave', default=5, help='Agents per cave')
# @click.pass_context
# def box_experiment(
#             ctx, n_trials, output_dir,
#             boxwidth_min, boxwidth_max, boxwidth_step,
#             noise_level_min, noise_level_max, noise_level_step,
#             n_caves, n_per_cave
#         ):
#     "Run the box experiment"

#     boxwidths = np.arange(boxwidth_min, boxwidth_max, boxwidth_step)
#     noise_levels = np.arange(
#         noise_level_min, noise_level_max, noise_level_step
#     )

#     param_pairs = [(bw, nl) for bw in boxwidths for nl in noise_levels]

#     # run_fun = _setup_box_experiment(n_per_cave, n_caves)

#     Parallel(n_jobs=mp.cpu_count())(
#         delayed(_box_experiment_5_20)(p, output_dir)
#         for p in param_pairs
#     )


# def _setup_box_experiment(n_per_cave, n_caves,
#                           n_iterations=2000, percolation_limit=True,
#                           output_dir='output'):

#     def _run_box_experiment(param_pair):

#         e = BoxedCavesExperiment(n_per_cave, n_caves, param_pair[0])
#         e.setup(percolation_limit=percolation_limit)
#         e.iterate(n_iterations, noise_level=param_pair[1])

#         pp = param_pair
#         save_path = os.path.join(
#             output_dir, 'box-experiment_bw={}_nl={}'.format(pp[0], pp[1])
#         )
#         pickle.dumps(e, open(save_path, 'wb'))

#     return _run_box_experiment


# def _box_experiment_5_20(param_pair, output_path='output'):

#     e = BoxedCavesExperiment(5, 20, param_pair[0])
#     e.setup(percolation_limit=True)
#     e.iterate(1000, noise_level=param_pair[1])

#     pp = param_pair
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)

#     save_path = os.path.join(
#         output_path, 'box-experiment_bw={:.2f}_nl={:.2f}'.format(pp[0], pp[1])
#     )
#     pickle.dump(e, open(save_path, 'wb'))
