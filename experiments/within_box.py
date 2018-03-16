import numpy as np

from macy import Experiment


class BoxedCavesExperiment(Experiment):

    def __init__(self, n_caves, n_per_cave, box_length, K=2,
                 outcome_metric='fm2011'):

        Experiment.__init__(self, n_caves, n_per_cave,
                            outcome_metric=outcome_metric)

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
