import click
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
import pickle

from joblib import Parallel, delayed
from functools import partial
from uuid import uuid4

from experiments.within_box import BoxedCavesExperiment
from reproduce_fm2011 import persist_experiments


def _run_exp(_, experiment='connected caveman', noise_level=0.0,
             n_caves=20, n_per_cave=5, S=1.0, K=2, n_iterations=4000,
             distance_metric='fm2011', verbose=False):

    # Add the same number random short-range or long-range ties.
    n_edges = 20

    if experiment == 'connected caveman':
        cc = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K,
                                  distance_metric=distance_metric)
        cc.iterate(n_iterations, verbose=verbose, noise_level=noise_level)
        ret = cc

    elif experiment == 'random short-range':
        # Connected caveman with short-range ties added randomly.
        ccsrt = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K,
                                     distance_metric=distance_metric)
        ccsrt.iterate(2000, verbose=False)
        ccsrt.add_shortrange_random_edges(n_edges)
        ccsrt.iterate(n_iterations - 2000,
                      verbose=False, noise_level=noise_level)
        ret = ccsrt

    elif experiment == 'random any-range':
        # Connected caveman with any-range ties added randomly.
        ccrt = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K,
                                    distance_metric=distance_metric)
        ccrt.iterate(2000, verbose=False)
        ccrt.add_random_edges(n_edges)
        ccrt.iterate(n_iterations - 2000,
                     verbose=False, noise_level=noise_level)
        ret = ccrt

    else:
        raise RuntimeError(
            'experiment type ' + experiment + ' not recognized.'
        )

    return {
        'polarization': ret.history['polarization'],
        'final coords': ret.history['final coords'],
        'graph': nx.to_numpy_matrix(ret.network.graph)
    }


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = dict()

@cli.command()
@click.argument('k', type=int)
@click.argument('output_dir')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=10000)
@click.option('--distance_metric', default='fm2011')
@click.pass_context
def reproduce_fig12(ctx, k, output_dir, n_trials, n_iterations, distance_metric):
    '''
    Create a set of HDF files corresponding to datasets for reproducing Figure 12b in FM2011
    '''
    # Set some processors aside for Numpy computations.
    pool = _get_default_pool()

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='connected caveman', K=k, distance_metric=distance_metric)
    cc_experiments = pool.imap(func, range(n_trials))
    print('completed connected caveman')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random short-range', K=k, distance_metric=distance_metric)
    srt_experiments = pool.imap(func, range(n_trials))
    print('completed short-range')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random any-range', K=k, distance_metric=distance_metric)
    any_experiments = pool.imap(func, range(n_trials))
    print('completed any-range')

    experiments = {
        'connected caveman': cc_experiments,
        'random short-range': srt_experiments,
        'random any-range': any_experiments
    }

    output_path = os.path.join(output_dir, str(uuid4()) + '.hdf5')

    persist_experiments(
        experiments,
        hdf_filename=output_path,
        metadata={'K': k}
    )


@cli.command()
@click.argument('n_per_cave', type=int)
@click.argument('output_dir')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=10000)
@click.pass_context
def reproduce_fig11(ctx, n_per_cave, output_dir, n_trials, n_iterations):
    '''
    Create a set of HDF files corresponding to datasets for reproducing Figure 11b in FM2011
    '''
    K = 2

    pool = mp.Pool(mp.cpu_count())

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='connected caveman', n_per_cave=n_per_cave)
    cc_experiments = pool.imap(func, range(n_trials))
    print('completed connected caveman')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random short-range', n_per_cave=n_per_cave)
    srt_experiments = pool.imap(func, range(n_trials))
    print('completed short-range')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random any-range', n_per_cave=n_per_cave)
    any_experiments = pool.imap(func, range(n_trials))
    print('completed any-range')

    experiments = {
        'connected caveman': cc_experiments,
        'random short-range': srt_experiments,
        'random any-range': any_experiments
    }

    output_path = os.path.join(output_dir, str(uuid4()) + '.hdf5')

    persist_experiments(
        experiments,
        hdf_filename=output_path,
        metadata={'K': 2, 'n_per_cave': n_per_cave}
    )


def _get_default_pool():
    return mp.Pool(max(2, mp.cpu_count() - 4))


@cli.command()
@click.argument('s', type=float)
@click.argument('k', type=int)
@click.argument('noise_level', type=float)
@click.argument('output_dir', type=str)
@click.option('--distance_metric', default='fm2011')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=4000)
@click.pass_context
def complexity_experiment(ctx, s, k, noise_level, output_dir, distance_metric,
                          n_trials, n_iterations):
    '''
    Run n_trials for a given maximum initial opinion feature S and cultural complexity K.
    '''

    pool = _get_default_pool()

    func = partial(_run_exp, n_iterations=n_iterations,
                   noise_level=noise_level, experiment='random any-range',
                   K=k, S=s, distance_metric=distance_metric)

    experiments = {
        'random any-range': pool.imap(func, range(n_trials))
    }

    output_path = os.path.join(output_dir, str(uuid4()) + '.hdf5')

    persist_experiments(
        experiments, hdf_filename=output_path,
        metadata={'K': k, 'S': s, 'noise_level': noise_level,
                  'distance_metric': distance_metric}
    )


def _uuid_hdfname():
    return str(uuid4()) + '.hdf5'


def _run_k3_2d_exp(_, experiment='connected caveman', noise_level=0.0,
                   n_caves=20, n_per_cave=5,
                   S=1.0, K=2, n_iterations=4000, verbose=False):

    # Add the same number random short-range or long-range ties.
    n_edges = 20

    if experiment == 'connected caveman':
        cc = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K)
        for agent in cc.network.graph.nodes():
            agent.opinions[-1] = 0.0
        cc.iterate(n_iterations, verbose=verbose, noise_level=noise_level)
        ret = cc

    elif experiment == 'random short-range':
        # Connected caveman with short-range ties added randomly.
        ccsrt = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K)
        for agent in ccsrt.network.graph.nodes():
            agent.opinions[-1] = 0.0
        ccsrt.iterate(2000, verbose=False)
        ccsrt.add_shortrange_random_edges(n_edges)
        ccsrt.iterate(n_iterations - 2000,
                      verbose=False, noise_level=noise_level)
        ret = ccsrt

    elif experiment == 'random any-range':
        # Connected caveman with any-range ties added randomly.
        ccrt = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K)
        for agent in ccrt.network.graph.nodes():
            agent.opinions[-1] = 0.0
        ccrt.iterate(2000, verbose=False)
        ccrt.add_random_edges(n_edges)
        ccrt.iterate(n_iterations - 2000,
                     verbose=False, noise_level=noise_level)
        ret = ccrt

    else:
        raise RuntimeError(
            'experiment type ' + experiment + ' not recognized.'
        )

    return {
        'polarization': ret.history['polarization'],
        'final coords': ret.history['final coords'],
        'graph': nx.to_numpy_matrix(ret.network.graph)
    }


@cli.command()
@click.argument('output_dir')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=4000)
@click.option('--distance_metric', default='fm2011')
@click.pass_context
def k3_2d_ops(ctx, output_dir, n_trials, n_iterations, distance_metric):
    'Test of 2D opinions in 3D ambient space'

    pool = _get_default_pool()
    experiment_names = [
        'connected caveman', 'random short-range', 'random any-range'
    ]
    experiments = {
        exp: pool.imap(
            partial(_run_k3_2d_exp, n_iterations=n_iterations,
                    K=3, experiment=exp),
            range(n_trials)
        )
        for exp in experiment_names
    }

    output_path = os.path.join(output_dir,  _uuid_hdfname())

    persist_experiments(
        experiments,
        hdf_filename=output_path,
        metadata={
                'K': 3,
                'note':
                'This tests the effect of 2D opinion '
                'distribution in 3D opinion space.'
            }
        )
>>>>>>> d51db016111513964fa6ed88a4994867b92cf0b1
