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

So apparently I need to update how I update. But that's annoying! Oh well.
'''

from experiments.within_box import BoxedCavesExperiment


def figure_10():
    '''

    p. 168
    '''

    # Set up

    # Save figure.
    pass


def figure_11b():
    '''
    p. 170
    '''
    # Set up

    # Save figure.
    pass


def figure_12b():
    '''
    p. 171
    '''
    # Set up

    # Save figure.
    pass


if __name__ == '__main__':
    figure_10()
    figure_11b()
    figure_12b()
