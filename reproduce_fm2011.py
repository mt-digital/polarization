'''
We will focus on the second experiment in the Flache and Macy (2011) paper.
We will only reproduce the model where agents are allowed to have negative
valence for now.

All figures contain three conditions: In one,
simulations run for the unmodified connected caveman graph. In the other two
conditions, ties are added randomly at iteration 2000. In the first of these
randomized conditions, 20 ``short-range'' ties are added at random at
iteration 2000. In the second,

What counts as an iteration? As you can see in the iterate method of the
Experiment class in macy/macy.py (currently l:165, which calls to the Network
method of the same name l:111), each iteration in my implementation corresponds
to calculating the update of all agents exactly once. This almost corresponds
to FM2011. For them "In every time step, one agent is selected randomly with
equal probability...either a randomly selected state or the weights of the
focal agent are selected for updating, but not both at the same time. Agents
are updated with replacement," so that "the same agent can be selected in two
consecutive time steps." Like I have done, for FM2011, "An iteration
corresponds to $N$ time steps, where $N$ is the number of individuals in the
population. Throughout this article we assume N=100."
'''
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from datetime import datetime
from experiments.within_box import BoxedCavesExperiment
from macy import get_opinions_xy


def figure_10(n_trials=3, n_iter=4000, verbose=True, hdf5_filename=None):
    '''
    p. 168
    '''
    # Set up
    n_caves = 20
    n_per_cave = 5
    K = 2

    experiments = {
        'connected caveman': [],
        'random short-range': [],
        'random any-range': []
    }

    for i in range(n_trials):
        # Connected caveman with no randomization.
        cc = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        cc.iterate(n_iter, verbose=verbose)
        experiments['connected caveman'].append(cc)

        # Add the same number random short-range or long-range ties.
        n_edges = 20

        # Connected caveman with short-range ties added randomly.
        ccsrt = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        ccsrt.iterate(2000, verbose=False)
        ccsrt.add_shortrange_random_edges(n_edges)
        ccsrt.iterate(n_iter - 2000, verbose=False)
        experiments['random short-range'].append(ccsrt)

        # Connected caveman with any-range ties added randomly.
        ccrt = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        ccrt.iterate(2000, verbose=False)
        ccrt.add_random_edges(n_edges)
        ccrt.iterate(n_iter - 2000, verbose=False)
        experiments['random any-range'].append(ccrt)

        print('finished {} of {}'.format(i+1, n_trials))

    persist_experiments(experiments, hdf5_filename=hdf5_filename)

    return experiments


def figure_11(n_trials=3, n_iter=4000, cave_sizes=[3, 5, 10, 20, 30, 40, 50],
              verbose=True, hdf5_filename=None):
    plt.gca().xaxis.grid(True)
    pass

def figure_12(n_trials=3, n_iter=4000, verbose=True, hdf5_filename=None):
    pass


def persist_experiments(experiments, hdf_filename=None, append_datetime=False,
                        metadata=None):
    '''
    Persist the three experiments to HDF5. Originally this accessed the classes
    themselves, but the classes are becoming too big to pass through
    multiprocessing interface on mapping. Eventually each experiment trial
    will have its own HDF reference, and these can be concatenated or
    something like that.
    '''

    nowstr = datetime.now().isoformat()

    if hdf_filename is None:
        hdf_filename = nowstr + '.hdf5'
    elif append_datetime:
        hdf_filename = \
            hdf_filename.replace('.hdf5', '') + '-' + nowstr + '.hdf5'

    experiment_names = [
        'connected caveman', 'random short-range', 'random any-range'
    ]

    with h5py.File(hdf_filename, 'w') as hf:

        if metadata is not None:
            assert type(metadata) is dict, 'metadata must be a dictionary'
            # Iterate over key/values and add metadata to root HDF attributes.
            for key in metadata:
                hf.attrs[key] = metadata[key]

        for experiment_name in experiment_names:

            # Each list in the experiments dictionary is considered a
            # single trial for the particular experimental condition.
            trials = experiments[experiment_name]

            # Take "y" vector from each polarization history, which is
            # polarization itself. XXX For some reason I am logging
            # which iteration, which is identical to the index, of course.
            # Should fix that sometime XXX.
            polarizations = np.array(
                [trial['polarization'] for trial in trials]
            )
            # Get timeseries of agent opinion coordinates for every trial.
            # coords = np.array([trial.history['coords'] for trial in trials])
            final_coords = np.array(
                [trial['final coords'] for trial in trials]
            )
            # Get adjacency matrix of each trial's graph.
            adjacencies = np.array(
                [trial['graph'] for trial in trials]
            )

            hf.create_dataset(
                experiment_name + '/polarization',
                data=polarizations,
                compression='gzip'
            )
            hf.create_dataset(
                experiment_name + '/final coords',
                data=final_coords,
                compression='gzip'
            )
            hf.create_dataset(
                experiment_name + '/graph',
                data=adjacencies,
                compression='gzip'
            )


def plot_figure10b(hdf_filename, save_name=None,
                   low_pct=25, high_pct=75):

    # fig, axes = plt.subplots(3, sharex=True)
    colors = ['r', 'b', 'g']
    # Keep track of maximum polarization to adjust axes.
    max_polarization = 0.0
    fig, ax = plt.subplots()
    with h5py.File(hdf_filename, 'r') as hf:

        for idx, key in enumerate(hf.keys()):

            polarizations = hf[key + '/polarization']

            plow = np.percentile(polarizations, low_pct, axis=0)
            phigh = np.percentile(polarizations, high_pct, axis=0)
            pmean = np.mean(polarizations, axis=0)

            max_polarization = max(np.max(phigh), max_polarization)

            plt.plot(plow, color=colors[idx], ls='--')
            plt.plot(phigh, color=colors[idx], ls='--')
            plt.plot(pmean, color=colors[idx], label=key)

    plt.ylim(0.0, max_polarization + 0.05)

    plt.legend(loc='best')

    plt.xlabel('Iteration')
    plt.ylabel('Polarization')
    ax.grid(axis='y', zorder=0)

    if save_name is not None:
        from matplotlib2tikz import save as tikz_save
        tikz_save(hdf_filename.replace('.hdf5', '.tex'))


def plot_figure11b():
    pass


def plot_figure12b():
    pass
