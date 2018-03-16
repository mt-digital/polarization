'''
Python implementation of Flache & Macy's "caveman" model of polarization.

'''
import numpy as np
import networkx as nx
import os
import random
import secrets

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from random import choice as random_choice
from random import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


class Agent:

    def __init__(self, n_opinions=2, cave=None, opinion_fill='random'):

        self.opinions = np.random.uniform(low=-1.0, high=1.0, size=2)
        self.weights = None
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

        self.n_nodes = n_nodes

    def add_random_edges(self, n_edges=20):
        '''
        FM2011 add 20 edges randomly and to immediate cave "to the right".
        This will add n_edges randomly.
        '''
        pass

    def randomize_edges(self, rewire_prob, percolation_limit=False):
        '''
        Arguments:
            rewire_fraction (float): Fraction of edges to rewire
        '''
        # Sample with replacement from this to keep self's edges pristine.
        edges_copy = list(self.graph.edges())

        # Helper to get two edges to swap as explained in reference at top.
        def get_swap_edges(edges, target_edges):
            e1, e2 = random.sample(edges, 2)
            A, B = e1
            C, D = e2

            def retry_condition(A, B, C, D):
                return (
                    A in e2 or
                    B in e2 or
                    (A, C) in edges or (C, A) in edges or
                    (B, D) in edges or (D, B) in edges or
                    (A, C) in target_edges or (C, A) in target_edges or
                    (B, D) in target_edges or (D, B) in target_edges
                )
            while retry_condition(A, B, C, D):
                e1, e2 = random.sample(edges, 2)
                A, B = e1
                C, D = e2
            return e1, e2

        # Helper to swap edges e1 and e2 from graph in-place.
        def swap_edges(graph, e1, e2):
            graph.remove_edge(*e1)
            graph.remove_edge(*e2)
            graph.add_edges_from([
                (e1[0], e2[0]), (e1[1], e2[1])
            ])

        # Must halve the given rewire_fraction to know how many swaps to do,
        # since each swap operation swaps two edges.
        rewire_number = int(round(
            (rewire_prob * 0.5) * self.graph.number_of_nodes()
        ))

        for _ in range(rewire_number):
            # Get the edges to be swapped and swap them.
            e1, e2 = get_swap_edges(edges_copy, self.graph.edges())
            swap_edges(self.graph, e1, e2)

            # Remove edges from potential ones to be swapped.
            edges_copy.remove(e1)
            edges_copy.remove(e2)

    # Actually this is doing 100 FM2011 iterations every time, one for each
    # agent.
    def iterate(self, noise_level=0.0):
        '''
        See bottom of p. 155 to p. 156. Each iteration in for-loop below
        is one time step. N time steps are one "iteration" in the model.
        N is the number of agents, but each agent is not necessarily updated.
        When an agent is selected for updating, its opinions or the weights
        associated with each of its neighbors are updated but not both.
        '''
        # Select n_nodes with replacement to update.
        node_list = list(self.graph.nodes())
        update_nodes = np.random.choice(node_list, size=self.n_nodes)

        for agent in update_nodes:
            # Update either agent opinions or weights depending on flip.
            flip = secrets.choice([False, True])
            # TODO make neighbors an attribute of agent and make functions
            # below into Agent methods.
            neighbors = self.graph.neighbors(agent)

            if flip:
                agent.opinions = opinion_update_vec(agent, neighbors,
                                                    noise_level=noise_level)
            else:
                update_weights(agent, neighbors)

    def draw(self):
        nx.draw_circular(self.graph)


class Experiment:
    '''
    Wrap the basic experiment structure. This will provide a randomly rewired
    connected caveman graph, where each edge is rewired with probability
    rewire_prob. Perhaps later I will make it more general to test the
    disconnected caveman graph, or have the graph type be more general. For now
    we are just investigating randomized connected caveman graphs; no need
    to add unnecessary complexity.
    '''
    def __init__(self, n_caves, n_per_cave, outcome_metric='fm2011',
                 rewire_prob=0.1, percolation_limit=False):
        # Initialize graph labelled by integers and randomize.
        network = Network(nx.connected_caveman_graph(n_caves, n_per_cave))
        network.randomize_edges(
            rewire_prob, percolation_limit=percolation_limit
        )
        # Initialize an agent at each node.
        relabelled_graph = nx.relabel_nodes(
            network.graph,
            {n: Agent() for n in network.graph.nodes()}
        )
        self.network = Network(relabelled_graph)
        del network

        i = 0
        for agent in self.network.graph.nodes():
            neighbors = self.network.graph.neighbors(agent)
            update_weights(agent, neighbors)
            i += 1
        print(i)

        # History will store each timestep's polarization measure.
        self.history = {'polarization': [], 'coords': []}
        self.iterations = 0
        self.n_caves = n_caves
        self.outcome_metric = outcome_metric

    # def setup(self, add_cxn_prob=.006, percolation_limit=False):

    #     self.network.add_random_connections(
    #         add_cxn_prob, percolation_limit=percolation_limit)

    def iterate(self, n_steps=1, noise_level=0.0):

        from progressbar import ProgressBar
        bar = ProgressBar()

        for i in bar(range(n_steps)):

            self.history['polarization'].append(
                (i,
                 polarization(self.network.graph, metric=self.outcome_metric))
            )
            self.history['coords'].append(
                [n.opinions for n in self.network.graph.nodes()]
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
            comment='See https://github.com/mt-digital/polarization for more info'
        )

        writer = ffmpeg_writer(fps=fps, metadata=metadata)

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
            pol_hist = self.history['polarization']
            coords_hist = self.history['coords']
            for i, pol in pol_hist:

                coords = coords_hist[i]
                plt.title('Polarization: {:.2f}\nIteration: {}'.format(
                        pol, i
                    )
                )
                x = [el[0] for el in coords]
                y = [el[1] for el in coords]
                l.set_data(x, y)
                writer.grab_frame()

                # cave_opinions = get_cave_opinions_xy(hist[i]['opinions'])

                # for idx, l in enumerate(cave_plots):
                #     x, y = cave_opinions[idx]


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


def calculate_weight(a1, a2, nonnegative=False, distance_metric='fm2011'):
    '''
    Calculate connection weight between two agents (Equation [1])
    '''
    o1 = a1.opinions
    o2 = a2.opinions

    if distance_metric == 'fm2011':
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

    elif distance_metric == 'cosine_similarity':
        return cosine_similarity(o1, o2)


def update_weights(agent, neighbors):
    '''
    Update agent weights in-place. TODO make this an Agent method.
    '''
    agent.weights = {
        neighbor: calculate_weight(agent, neighbor)
        for neighbor in neighbors
    }
    # print('updated {} with {}'.format(agent, agent.weights))


def raw_opinion_update_vec(agent, neighbors, distance_metric='fm2011'):

    # import ipdb
    # ipdb.set_trace()
    neighbors = list(neighbors)
    factor = (1.0 / (2.0 * len(neighbors)))

    return factor * np.sum(
        agent.weights[neighbor] *
        (neighbor.opinions - agent.opinions)

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


def polarization(graph, metric='fm2011'):
    '''
    Implementing Equation 3. Metrics used: fm2011, cityblock, or euclidian.
    fm2011 uses cityblock scaled by 1/K.

    Returns:
        (float) : variance of all pairwise distances.
    '''
    # List of opinion coordinates for all agents.
    X = [n.opinions for n in graph.nodes()]
    # To be used in slicing out the upper triangle of the distance matrix.
    N = len(X)

    if metric == 'fm2011':
        # FM2011 distance metric contains an averaging factor over features.
        K = len(X[0])
        # The FM2011 distance is just cityblock/manhattan/L1 distance
        # scaled by 1/K.
        distances = (1.0 / K) * cdist(X, X, metric='cityblock')
    else:
        distances = cdist(X, X, metric=metric)

    # FM2011 use the variance over non-repeating d_ij with i≠j, as
    # best I can tell. Their explanation/notation is confusing, see p. 156.
    # I believe by taking either the upper or lower triangle of the distance
    # matrix implements the summation; the triangles are equivalent. k=1
    # drops the diagonal.
    return distances[np.triu_indices(N, k=1)].var()
