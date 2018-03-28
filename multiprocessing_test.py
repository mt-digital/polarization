import multiprocessing as mp

from functools import partial

from experiments.within_box import BoxedCavesExperiment
from reproduce_fm2011 import persist_experiments


# K = 2
# N_ITER = 2000
N_TRIALS = 4


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


pool = mp.Pool(mp.cpu_count())


func = partial(_run_exp, experiment='connected caveman')
cc_experiments = pool.imap(func, range(N_TRIALS))
print('completed connected caveman')

func = partial(_run_exp, experiment='random short-range')
srt_experiments = pool.imap(func, range(N_TRIALS))
print('completed short-range')

func = partial(_run_exp, experiment='random any-range')
any_experiments = pool.imap(func, range(N_TRIALS))
print('completed any-range')

experiments = {
    'connected caveman': list(cc_experiments),
    'random short-range': list(srt_experiments),
    'random any-range': list(any_experiments)
}

persist_experiments(
    experiments,
    hdf_filename='multiprocessing_persist.hdf5',
    metadata={'K': 2}
)
