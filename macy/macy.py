'''
Python implementation of Flache & Macy's "caveman" model of polarization.

'''
import numpy as np
import networkx as nx
import os

from collections import OrderedDict

from copy import deepcopy
from datetime import datetime
from random import choice as random_choice
from random import shuffle


class Agent:

    def __init__(self, n_opinions=2, cave=None, opinion_fill='random'):

        self.opinions = np.random.uniform(low=-1.0, high=1.0, size=2)
        self.cave = cave


class Network(nx.Graph):

    def __init__(self, initial_graph=None, close_connections=False):
        '''
        Create a network of any initial configuration. Provides methods
        for iterating (updating opinions and weights) and for randomizing
        connections. We can provide other helper functions or class methods
        for building specific initial configurations.

        '''
        self.graph = initial_graph

        # all pairs of vertices
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        self.possible_edges = [
            (nodes[i], nodes[j])
            for i in range(n_nodes)
            for j in range(i, n_nodes)
            if i != j
        ]

        # all pairs of vertices with current neighbors removed
        current_edges = self.graph.edges()

        self.non_neighbors = deepcopy(self.possible_edges)
        # print(len(current_edges))
        # for i, el in enumerate(current_edges):
        #     if (el[0], el[1]) in self.non_neighbors:
        #         self.non_neighbors.remove((el[0], el[1]))
        #     elif (el[1], el[0]) in self.non_neighbors:
        #         self.non_neighbors.remove((el[1], el[0]))

        #     print(i)

        self.non_neighbors = [
            el for el in self.possible_edges
            if (
                (el[0], el[1]) not in current_edges and
                (el[1], el[0]) not in current_edges
            )
        ]

    def add_random_connections(self, add_cxn_prob, percolation_limit=False):

        # you're not removing any existing!

        for pair in self.non_neighbors:
            if np.random.uniform() < add_cxn_prob:
                self.graph.add_edge(*pair)
                self.non_neighbors.remove(pair)

        # add edges until percolation limit is reached
        if percolation_limit:
            while not nx.is_connected(self.graph):
                new_edge = random_choice(self.non_neighbors)
                self.graph.add_edge(*new_edge)
                self.non_neighbors.remove(new_edge)

    def add_random_connection(self):

        if len(self.non_neighbors) > 0:
            new_edge = random_choice(self.non_neighbors)
            self.graph.add_edge(*new_edge)

            # new_edge now defines neighbors
            self.non_neighbors.remove(new_edge)
        else:
            raise RuntimeError('No non-neighbors left to connect')

    def iterate(self, noise_level=0.0):

        # asynchronous updating
        nodes = list(self.graph.nodes())
        shuffle(nodes)

        for agent in nodes:
            neighbors = self.graph.neighbors(agent)
            agent.opinions = opinion_update_vec(agent, neighbors,
                                                noise_level=noise_level)

    def draw(self):
        nx.draw_circular(self.graph)


class Experiment:

    def __init__(self, n_per_cave, n_caves):
        self.network = Network(caves(n_caves=n_caves, n_agents=n_per_cave))
        self.history = OrderedDict()
        self.iterations = 0
        self.n_caves = n_caves

    def setup(self, add_cxn_prob=.006, percolation_limit=False):

        self.network.add_random_connections(
            add_cxn_prob, percolation_limit=percolation_limit)

    def iterate(self, n_steps=1, noise_level=0.0):

        for i in range(n_steps):

            self.history.update(
                {
                    self.iterations: {
                        'polarization': polarization(self.network.graph),
                        'opinions': [agent.opinions for agent in
                                     deepcopy(sorted(self.network.graph.nodes()))]
                    }
                }
            )

            self.network.iterate(noise_level=noise_level)
            self.iterations += 1

    def make_opinion_movie(self, movie_name=None, fps=15, dpi=150):
        '''
        Arguments:
            movie_name (str): becomes directory for video and components
            resolution (int):
        '''
        import matplotlib.pyplot as plt
        import matplotlib.animation as anim
        import seaborn as sns

        if movie_name is None:
            movie_name = datetime.now().isoformat() + '.mp4'

        ffmpeg_writer = anim.writers['ffmpeg']
        metadata = dict(
            title='Flache & Macy Animation ' + movie_name,
            artist='Matthew A. Turner',
            comment='See https://github.com/mtpain/polarization for more info'
        )

        writer = ffmpeg_writer(fps=fps, metadata=metadata)

        hist = self.history
        fig = plt.figure()
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        ax = fig.add_subplot(111)

        cave_plots = []
        colors = sns.color_palette('husl', self.n_caves)

        for i in range(self.n_caves):
            l, = plt.plot([], [], 'o', lw=0,
                          color=colors[i], ms=10, alpha=0.85)
            cave_plots.append(l)

        ax.set_aspect('equal')

        with writer.saving(fig, movie_name, dpi):

            for i in range(max(hist.keys())):

                # plt.title('Polarization: {:.2f}\nIteration: {}'.format(
                #         hist[i]['polarization'], i
                #     )
                # )

                cave_opinions = get_cave_opinions_xy(hist[i]['opinions'])

                for idx, l in enumerate(cave_plots):
                    x, y = cave_opinions[idx]
                    l.set_data(x, y)

                writer.grab_frame()


def get_opinions_xy(opinions):
    return np.array([o[0] for o in opinions]), np.array([o[1] for o in opinions])


def get_cave_opinions_xy(agents, n_caves=20):

    return {

        i: get_opinions_xy(
                [
                    agent.opinions for agent in agents
                    if agent.cave == i
                ]
            )
        for i in range(n_caves)
    }


def experiment(n_caves, n_agents, add_cxn_prob=.003, iterations=1000):

    network = Network(caves(n_caves, n_agents))
    network.add_random_connections(add_cxn_prob=add_cxn_prob)

    for _ in range(iterations):
        network.iterate()

    return network


def caves(n_caves=20, n_agents=5):
    '''
    Make a totally connected cave with n_agents
    '''
    relabelled_graph = nx.relabel_nodes(
        nx.caveman_graph(n_caves, n_agents),
        {i: Agent() for i in range(n_caves * n_agents)}
    )

    for idx, subg in enumerate(nx.connected_components(relabelled_graph)):
        for agent in subg:
            agent.cave = idx + 1

    return relabelled_graph


def weight(a1, a2, nonnegative=False):
    '''
    Calculate connection weight between two agents (Equation [1])
    '''
    o1 = a1.opinions
    o2 = a2.opinions

    if o1.shape != o2.shape:
        raise RuntimeError("Agent's opinion vectors have different shapes")
    K = len(o1)

    diff = abs(o2 - o1)
    numerator = np.sum(diff)

    if nonnegative:
        nonneg_fac = 2.0
    else:
        nonneg_fac = 1.0

    return 1 - (numerator / (nonneg_fac * K))


def raw_opinion_update_vec(agent, neighbors):

    neighbors = list(neighbors)
    factor = (1.0 / (2.0 * len(neighbors)))

    return factor * np.sum(
        weight(agent, neighbor) * (neighbor.opinions - agent.opinions)
        for neighbor in neighbors
    )


def opinion_update_vec(agent, neighbors, noise_level=0.0):

    raw_update_vec = raw_opinion_update_vec(agent, neighbors)

    ret = np.zeros(raw_update_vec.shape)

    noise_term = noise_level * np.random.normal()
    for i, op in enumerate(agent.opinions):
        if op > 0:
            ret[i] = op + ((noise_term + raw_update_vec[i])*(1 - op))
        else:
            ret[i] = op + ((noise_term + raw_update_vec[i])*(1 + op))

    return ret


def polarization(graph):
    '''
    Implementing Equation 3
    '''

    nodes = list(graph.nodes())

    L = len(nodes)
    distances = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            distances[i, j] = distance(nodes[i], nodes[j])

    d_expected = distances.sum() / (L*(L-1))

    d_sub_mean = (distances - d_expected)
    for i in range(L):
        d_sub_mean[i, i] = 0.0

    d_sub_mean_sq = np.sum(np.power(d_sub_mean, 2))

    return (1/(L*(L-1))) * d_sub_mean_sq


def distance(agent_1, agent_2):

    n_ops = len(agent_1.opinions)
    return (1.0 / n_ops) * np.sum(np.abs(agent_1.opinions - agent_2.opinions))
