'''
Visualization routines for polarization project

Author: Matthew A. Turner
Date: May 7, 2017
'''
import matplotlib.pyplot as plt
import os
import seaborn as sns

from numpy import sort

def phase_diagram(df, noise_level_lim=0.4, n_iter=1000,
                  seaborn_context='paper', font_scale=1.2,
                  xlabel='Maximum initial opinion magnitude, $S$',
                  ylabel='Cultural effects magnitude, $\sigma^c$',
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

    plt.figure()

    ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu_r',
                     **seaborn_heatmap_kwargs)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, size=20)
    ax.set_ylabel(ylabel, size=20)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return ax


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
