import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import seaborn as sns

from experiments.within_box import BoxedCavesExperiment


def vis_graph(n_caves, n_per_cave, l=2, lp=0.2, randomize=False):

    pos = [(
            (l * np.cos(i * 2 * np.pi / n_caves)) +
            (lp * np.cos(j * 2 * np.pi / n_per_cave)),
            (l * np.sin(i * 2 * np.pi / n_caves)) +
            (lp * np.sin(j * 2 * np.pi / n_per_cave))
        )
        for i in range(n_caves)
        for j in range(n_per_cave)
    ]

    cc = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, 2)
    if randomize:
        cc.add_random_edges(randomize)

    gr = cc.network.graph

    pos_dict = {
        node: pos[idx] for idx, node in enumerate(gr.nodes())
    }

    # colors = sns.color_palette(sns.diverging_palette(255, 133, l=60,
    #                      n=n_caves, center="dark"))
    # colors = sns.color_palette('husl', n_caves)
    colors = sns.color_palette('hls', n_caves)

    node_colors = [colors[gr.node[node]['cave_idx']] for node in pos_dict.keys()]

    plt.figure()

    nx.draw(gr, pos=pos_dict, node_color=node_colors)
