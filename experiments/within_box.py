import numpy as np

from macy import Experiment


class BoxedCavesExperiment(Experiment):
    def __init__(self, n_per_cave, n_caves, box_length, K=2):
        Experiment.__init__(self, n_per_cave, n_caves)

        _assign_boxed_constrained_opinions(
            self.network.graph.nodes(), box_length, K
        )


def _assign_boxed_constrained_opinions(agents, box_length, K):
    '''
    Arguments:
        box_length (float): value between 0 and 1 that limits absolute value of
            any given opinion in each agent

    Returns:
        None
    '''

    for agent in agents:
        agent.opinions = np.random.uniform(-box_length, box_length, size=(K,))
