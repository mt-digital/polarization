'''
Visualization routines for polarization project

Author: Matthew A. Turner
Date: May 7, 2017
'''
import matplotlib.pyplot as plt
import os
import seaborn as sns

from numpy import arange, flipud, poly1d, polyfit, roots, sort, zeros
from operator import sub

from experiments.within_box import BoxedCavesExperiment


def phase_diagram(df, noise_level_lim=0.4, n_iter=1000,
                  seaborn_context='paper', font_scale=1.2,
                  xlabel='Maximum initial opinion magnitude, $S$',
                  ylabel='Cultural effects magnitude, $\sigma^c$',
                  annot=True, output_path=None,
                  **seaborn_heatmap_kwargs):
    '''
    2D phase diagram with maximum initial opinion magnitude on x-axis and
    random cultural effects magnitude on the y-axis. The variables are
    currently named for their original, more generic names, box_width and
    noise_level, respectively.

    Arguments:
        df (pandas.DataFrame): data frame built from raw .csv data exported
            from cluster
        noise_level_lim (float): maximum cultural effects magnitude to include
            in the diagram
        seaborn_context (str): seaborn context ('talk', 'paper', etc)
            see http://seaborn.pydata.org/generated/seaborn.set_context.html
        font_scale (float): scaling for all fonts
            see http://seaborn.pydata.org/generated/seaborn.set_context.html

    Returns:
        (matplotlib.pyplot.Axes)
    '''
    sns.set_context(context=seaborn_context, font_scale=font_scale)

    # limit to final iteration's data and drop iteration column
    # df_finpol = df[df.iteration == n_iter - 1][
    #     ['trial', 'box_width', 'noise_level', 'polarization']
    # ]
    df_finpol = df[df.iteration == n_iter - 1][
        ['box_width', 'noise_level', 'polarization']
    ]

    groupby_bw_nl = df_finpol.groupby(['box_width', 'noise_level'])

    means_reset = groupby_bw_nl.mean().reset_index()

    means_reset_part = means_reset[means_reset.noise_level <= noise_level_lim]

    pivot = means_reset_part.pivot('noise_level', 'box_width', 'polarization')

    fig = plt.figure(figsize=(9, 6.5))

    ax = sns.heatmap(pivot, annot=annot, fmt='.2f', cmap='YlGnBu_r',
                     **seaborn_heatmap_kwargs)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, size=20)
    ax.set_ylabel(ylabel, size=20)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    cur_xticklabels = ax.get_xticklabels()
    ax.set_xticks(ax.get_xticks()[1::2])
    ax.set_xticklabels(cur_xticklabels[1::2])

    if output_path is None:
        return ax
    else:
        fig.savefig(output_path, dpi=300)



def polarization_by_iteration(df, box_width, noise_level,
                              seaborn_context='paper', font_scale=1.5,
                              save_dir='/Users/mt/workspace/papers/269final/figures'):
    '''
    Going to have to do some weird stuff because I was accidentally writing
    multiple copies of each row in data_export.py
    '''
    sns.set_context(context=seaborn_context, font_scale=font_scale)
    sns.set_palette(sns.color_palette("husl", 10))

    df_part = df[
        (df.box_width == box_width) &
        (df.noise_level == noise_level)
    ]

    trial_idxs = sort(df_part.trial.unique())

    fig = plt.figure(figsize=(6.5, 3.5))
    for trial_idx in trial_idxs:
        trial = df_part[df_part.trial == trial_idx]
        x = trial.iteration.unique()
        y = trial.polarization.unique()

        plt.plot(x, y, lw=1.25, label='Trial {}'.format(trial_idx))

    # plt.legend(loc='best', prop=dict(size=8))
    plt.xlabel('Iteration', size=14)
    plt.ylabel('Polarization', size=14)
    plt.title(
        '$S={:.2f} \quad \sigma_c={:.2f}$'.format(box_width, noise_level),
        size=16)

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            save_dir, 's-{:.2f}_sc-{:.2f}_series'.format(
                box_width, noise_level
            ).replace('.', 'd') + '.pdf'  # yeah..just go with it
        )
    )

    plt.close()


def _get_groupby_noise_level(
            df, box_width, min_noise_level, max_noise_level, n_iter
        ):

    df_part = df[
        (df.box_width == box_width) &
        (df.noise_level >= min_noise_level) &
        (df.noise_level <= max_noise_level)
    ]

    df_part_finpol = df_part[df_part.iteration == n_iter - 1][
        ['noise_level', 'polarization']
    ]

    df_part_finpol = df_part_finpol.drop_duplicates()

    gb_nl = df_part_finpol.groupby('noise_level')

    return gb_nl


def polarization_by_cultural_effect(df, save_path,
                                    box_widths=[0.25, 0.5, 0.75],
                                    n_iter=1000,
                                    min_noise_level=0.0,
                                    max_noise_level=0.2,
                                    seaborn_context='paper',
                                    font_scale=1.2):
    '''
    Plot the polarization over the cultural effect (noise_levels) within
    the range [min_noise_level, max_noise_level]
    '''
    sns.set_context(context=seaborn_context, font_scale=font_scale)
    sns.set_palette(sns.color_palette("husl", len(box_widths)))

    fig = plt.figure(figsize=(6.5, 4))

    for box_width in box_widths:

        df_part = df[
            (df.box_width == box_width) &
            (df.noise_level >= min_noise_level) &
            (df.noise_level <= max_noise_level)
        ]

        df_part_finpol = df_part[df_part.iteration == n_iter - 1][
            ['noise_level', 'polarization']
        ]

        df_part_finpol = df_part_finpol.drop_duplicates()

        gb_nl = df_part_finpol.groupby('noise_level')

        means = gb_nl.mean()
        var = gb_nl.var()

        plt.errorbar(
            means.index.values, means.values, var.values,
            ls='-', marker='o', label='$S = {}$'.format(box_width)
        )

    plt.legend(loc='best')
    plt.xlim(-.01, .21)
    plt.ylim(-.025, 1.025)

    plt.xlabel('Cultural effect magnitude, $\sigma_c$')
    plt.ylabel('Polarization')

    plt.title('Average polarization over 10 trials')

    plt.tight_layout()

    fig.savefig(save_path)

    plt.close()


def transition_widths(box_widths, transition_widths, save_path,
                      seaborn_context='paper', font_scale=1.2):
    '''
    Plot the transition widths as found by calculating the widths of fitted
    quadratic functions
    '''
    sns.set_context(context=seaborn_context, font_scale=font_scale)
    sns.set_palette(sns.color_palette("husl", len(box_widths)))

    fig = plt.figure(figsize=(6.5, 4))

    plt.plot(box_widths, transition_widths, 'o-', color='black', lw=2, ms=6)

    plt.xticks(box_widths)
    plt.xlim(0, 1.05)

    plt.xlabel('Maximum initial opinion magnitude, $S$')
    plt.ylabel('Transition width along $\sigma_c$')
    plt.title('Phase transition in variance over trials')

    plt.tight_layout()

    fig.savefig(save_path)

    plt.close()


def make_transition_widths_array(
            df,
            box_widths=[0.05, .1, .15, .2, .25, .3, .35, .4, .45,
                        .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.0],
            n_iter=1000,
            min_noise_level=0.0,
            max_noise_level=0.2,
            seaborn_context='paper',
            font_scale=1.2
        ):
    transition_widths = zeros((len(box_widths),))

    for bw_idx, box_width in enumerate(box_widths):
        df_part = df[
            (df.box_width == box_width) &
            (df.noise_level >= min_noise_level) &
            (df.noise_level <= max_noise_level)
        ]

        df_part_finpol = df_part[df_part.iteration == n_iter - 1][
            ['noise_level', 'polarization']
        ]

        df_part_finpol = df_part_finpol.drop_duplicates()

        var = df_part_finpol.groupby('noise_level').var()
        y = var.polarization.values

        # find the first index of variance that is non-zero
        try:
            first_nonzero_idx = arange(len(y))[y > 1e-2][0]

            if first_nonzero_idx == 0:
                first_idx = 0
            else:
                first_idx = first_nonzero_idx - 1

            x = var.index.values[first_idx:]
            y = y[first_idx:]

            fit = polyfit(
                x, y, 2,
                # weight the first zero more heavily than the rest
                w=[1] + [.1 for _ in range(len(x) - 1)]
            )

            transition_width = sub(*flipud(sort(roots(fit))))

            transition_widths[bw_idx] = transition_width

        except IndexError:
            print('\nerror on {}!\n\nx: {}\ny: {}\nvar vals:{}'.format(
                box_width, x, y, var.polarization.values))
            transition_widths[bw_idx] = 0.0

    return box_widths, transition_widths


def quadratic_fit_variance(df, min_noise_level, max_noise_level,
                           save_path, box_widths=[.25, .5, .75, .9],
                           seaborn_context='paper', font_scale=1.2):

    sns.set_context(context=seaborn_context, font_scale=font_scale)
    color = sns.color_palette("husl", 1)[0]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 8))
    axes = axes.flatten()

    for i, box_width in enumerate(box_widths):

        ax = axes[i]

        var, x, y, fit = \
            _do_quadratic_fit_variance(
                df, box_width, min_noise_level, max_noise_level
            )

        var.columns = ['variance in final polarization']
        var.plot(color=color, marker='o', lw=0, ms=8, ax=ax, legend=False)

        poly = poly1d(fit)

        ax.plot(x, poly(x), '--', color=color)

        # if box_width == .9:
        #     ax.legend(loc='upper right')
        # else:
        #     ax.legend(loc='upper left')

        if i in [2, 3]:
            ax.set_xlabel('Cultural effects magnitude, $\sigma^c$')
        else:
            ax.set_xlabel('')
        if i % 2 == 0:
            ax.set_ylabel('Variance of final polarization')
        ax.set_title('$S = {}$'.format(box_width))
        ax.set_xlim(-.005, .205)

        # magic number from maximum variance of default box_widths
        ax.set_ylim(-.005, .22)

        ax.set_aspect('equal')

    plt.tight_layout()

    fig.savefig(save_path)

    plt.close()


def _do_quadratic_fit_variance(df, box_width,
                               min_noise_level, max_noise_level, n_iter=1000):
    '''
    Fit a quadratic to the variance for a particular box_width

    Returns:
        (numpy.ndarray): length-three array of quadratic fit values
    '''
    df_part = df[
        (df.box_width == box_width) &
        (df.noise_level >= min_noise_level) &
        (df.noise_level <= max_noise_level)
    ]

    df_part_finpol = df_part[df_part.iteration == n_iter - 1][
        ['noise_level', 'polarization']
    ]

    df_part_finpol = df_part_finpol.drop_duplicates()

    var = df_part_finpol.groupby('noise_level').var()
    y = var.polarization.values

    # find the first index of variance that is non-zero
    first_nonzero_idx = arange(len(y))[y > 1e-2][0]

    if first_nonzero_idx == 0:
        first_idx = 0
    else:
        first_idx = first_nonzero_idx - 1

    x = var.index.values[first_idx:]
    y = y[first_idx:]

    fit = polyfit(
        x, y, 2,
        # weight the first zero more heavily than the rest
        w=[1] + [.1 for _ in range(len(x) - 1)]
    )

    return var, x, y, fit


def sample_initial_opinion_limits(save_path=None, box_widths=[.2, .5, .8]):

    fig, axs = plt.subplots(1, 3)

    for i, bw in enumerate(box_widths):

        ax = axs[i]

        bce = BoxedCavesExperiment(5, 5, bw)
        bce_nodes = bce.network.graph.nodes()

        x = [el.opinions[0] for el in bce_nodes]
        y = [el.opinions[1] for el in bce_nodes]

        ax.plot(x, y, 'o', mfc='#4285F4', lw=0)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title('$S = {}$'.format(bw))
        ax.set_xlabel('Opinion 1')

        if i == 0:
            ax.set_ylabel('Opinion 2')

        ax.set_aspect('equal')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
