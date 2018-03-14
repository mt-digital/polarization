import warnings
warnings.filterwarnings("ignore")  # to silence tight_layout warnings

import matplotlib.pyplot as plt

from experiments.within_box import BoxedCavesExperiment

S = 1.0
noise_level = 0.0
K = 2
rewire_probability = 0.006  # need to double check this

def k_experiment(n_caves=20, n_per_cave=5, outcome_metric='fm2011', cxn_prob=0.01):
    for K in [1, 2, 3, 5]:

        experiment = BoxedCavesExperiment(
            n_per_cave, n_caves, S, K, outcome_metric=outcome_metric
        )
        experiment.setup(add_cxn_prob=cxn_prob)

        experiment.iterate(100, noise_level=noise_level)

        h = experiment.history['polarization']
        t = [el[0] for el in h]

        polarization = [el[1] for el in h]
        plt.plot(t, polarization, label='K={}'.format(K))

    plt.legend()
