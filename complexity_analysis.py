import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

from glob import glob


def _all_final_polarizations(hdf, experiment='random any-range'):
    return hdf[experiment + '/polarization'][:, -1]


def _final_mean(hdf, experiment='random any-range'):

    # Extract the final polarization measurement from all n_trials trials.
    all_final_polarizations = _all_final_polarizations(hdf, experiment)

    return all_final_polarizations.mean()


def _hdf_list(data_dir):
    return [h5py.File(f, 'r') for f in glob(os.path.join(data_dir, '*'))]


def _lookup_hdf(data_dir, **criteria):
    '''
    Assumes there is only one with the key, returns the first HDF file found
    in data_dir that matches the criteria.

    Arguments:
        data_dir (str): directory containing HDF files from a full modeling run
    '''
    for f in glob(os.path.join(data_dir, '*')):

        hdf = h5py.File(f, 'r')
        match = True

        for k, v in criteria.items():
            try:
                match &= hdf.attrs[k] == v
            except KeyError:
                warnings.warn('key {} not found for file {} in {}'.format(
                    k, f, data_dir
                ))
                return None

        if match:
            return hdf
        else:
            hdf.close()


def final_polarization_histogram(data_dir, **criteria):

    polarizations = _all_final_polarizations(
        _lookup_hdf(data_dir, **criteria)
    )
    plt.hist(polarizations)


def plot_p_v_noise_and_k(data_dir, Ks=[2, 3, 4, 5], **kwargs):

    hdfs = _hdf_list(data_dir)

    hdf0 = hdfs[0]
    distance_metric = hdf0.attrs['distance_metric']

    for K in Ks:

        if 'figsize' in kwargs:
            plt.figure(figsize=kwargs['figsize'])
            del kwargs['figsize']
        else:
            plt.figure()

        # Limit hdfs of interest to those of the K of current interest.
        final_means = [
            _final_mean(hdf) for hdf in hdfs
            if hdf.attrs['K'] == K
        ]

        # Use noise_level and S as index to force uniqueness.
        index = [
            (hdf.attrs['noise_level'], hdf.attrs['S']) for hdf in hdfs
            if hdf.attrs['K'] == K
        ]
        index = pd.MultiIndex.from_tuples(index)
        index.set_names(['noise level', 'S'], inplace=True)

        df = pd.DataFrame(
            index=index, data=final_means, columns=['Average polarization']
        ).reset_index(
        ).pivot('noise level', 'S', 'Average polarization')

        ax = sns.heatmap(df, cmap='YlGnBu_r')

        # Make noise level run from small to large.
        ax.invert_yaxis()

        ax.set_title(r'$K={}$'.format(K))

        # Force the heatmap to be square.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0)/(y1 - y0))

        plt.savefig(
            'reports/Figures/p_v_noise_k={}_{}.pdf'.format(K, distance_metric)
        )
        plt.close()

    for hdf in hdfs:
        hdf.close()


def plot_S_K_experiment(data_dirs, plot_start=0, agg_fun=np.mean, **kwargs):
    '''
    Plot average, median, or other aggregate of final polarizaiton over
    trials for various values of initial opinion magnitude S and cultural
    complexity K. Pass optional kwargs of figsize or other matplotlib
    plot options.

    Arguments:
        data_dirs (str or list): Location of HDF files to be used
        agg_fun (function): Method for aggregating final polarizations
        plot_start (int): Index to start plotting at. Used for excluding
        uninteresting parts of the plot.
    Returns:
        None
    '''
    if 'figsize' in kwargs:
        plt.figure(figsize=kwargs['figsize'])
        del kwargs['figsize']
    else:
        plt.figure()

    hdf_lookup = 'random any-range/polarization'

    if type(data_dirs) is str:
        data_dirs = [data_dirs]

    # Build list of HDF files from data directories.
    hdfs = []
    for d in data_dirs:
        hdfs += [h5py.File(f) for f in glob(os.path.join(d, '*.hdf5'))]

    Ks = list(set(hdf.attrs['K'] for hdf in hdfs))
    Ks.sort()

    for K in Ks:

        if 'noise_level' in hdfs[0].attrs:
            hdfs_K = [hdf for hdf in hdfs
                      if hdf.attrs['K'] == K
                      and hdf.attrs['noise_level'] == 0.0]
        else:
            hdfs_K = [hdf for hdf in hdfs if hdf.attrs['K'] == K]

        hdfs_K.sort(key=lambda x: x.attrs['S'])

        n_hdfs_K = len(hdfs_K)
        y_vals = np.zeros(n_hdfs_K)
        y_std = np.zeros(n_hdfs_K)

        for idx in range(n_hdfs_K):
            # Get final polarization value for all trials and average.
            final_polarizations = hdfs_K[idx][hdf_lookup][:, -1]
            y_vals[idx] = agg_fun(final_polarizations)
            y_std[idx] = final_polarizations.std()

        x = [str(hdf.attrs['S']) for hdf in hdfs_K]

        # These [3:] are ugly, but just working for the data.
        # plt.plot(x[3:], y_vals[3:], 'o-', label=r'$K={}$'.format(K),
        #          lw=2, ms=8, alpha=0.65)
        plt.plot(x[plot_start:], y_vals[plot_start:], 'o-', label=r'$K={}$'.format(K),
                 lw=2, ms=8, alpha=0.65)

    plt.legend(loc='upper left')
    if agg_fun == np.mean:
        plt.ylabel('Average polarization')
    elif agg_fun == np.median:
        plt.ylabel('Median polarization')
    plt.xlabel('S')

    if plot_start > 0:
        plt.xticks(range(n_hdfs_K)[:-plot_start], x[plot_start:])


def _nonzero_final_polarization_selector(hdf,
                                         experiment='random any-range',
                                         tol=1e-6):
    final_polarizations = _all_final_polarizations(hdf, experiment=experiment)
    return final_polarizations > tol



