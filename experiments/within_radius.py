import networkx as nx
import numpy as np

from macy import Experiment, Network, Agent


class InRadiusExperiment(Experiment):
    '''
    Start with a completely connected graph with all opinion vectors within a
    given radius.
    '''
    def __init__(self, radius, n_agents=100):

        self.network = Network(_radius_constrained_network(radius, n_agents))
        self.history = {}
        self.iterations = 0
        self.n_caves = 1


def _radius_constrained_network(radius, n_agents, K=2):

    agents = [Agent() for _ in range(n_agents)]

    _assign_random_constrained_opinions(agents, radius, K)

    ret = nx.relabel_nodes(
        nx.complete_graph(n_agents),
        {i: agent for i, agent in enumerate(agents)}
    )

    # all agents go in single cave in this experiment
    for agent in ret.nodes():
        agent.cave = 0

    return ret


def _radius_constrain_caves(graph, radius):
    '''
    Constrain initial values of caveman graph so all opinion vectors are within
    given radius
    '''
    pass


def _assign_random_constrained_opinions(agents, radius, K):

    count = 0
    # generate n_agents random opinion vectors inside the (K-1)-ball
    n_agents = len(agents)
    while count < n_agents:

        candidate = np.random.uniform(-1.0, 1.0, size=(K,))

        if np.linalg.norm(candidate, 2) < radius:

            agents[count].opinions = candidate

            count += 1


class InRadiusCavesExperiment(Experiment):

    def __init__(self, n_per_cave, n_caves, radius, K=2):
        Experiment.__init__(self, n_per_cave, n_caves)

        _assign_random_constrained_opinions(
            self.network.graph.nodes(), radius, K
        )
