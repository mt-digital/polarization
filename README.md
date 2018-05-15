# polarization

Implementation and extension of Flache and Macy (2011) work on cultural polarization.

## Introduction

The work by Flache and Macy (2011) showed that more complex cultures less often
found themselves in states of high polarization. They also found that high
levels of cultural polarization were more likely if agents were
connected on a small-world network. At first, I wondered what was the effect of
initial conditions on the final polarization an artificial society achieved.
Secondly, I wondered what was the effect of communication noise on these
artificial societies. So, I implemented the Flache and Macy (2011) model (FM 
model) in Python and began experimenting. A year later we are submitting 
this work for review.

Our focus here is on the software that powers our results, and not the social
theory. Please see Flache and Macy's 2011 paper in the Journal of Mathematical
Sociology for more information (https://www.tandfonline.com/doi/abs/10.1080/0022250X.2010.532261). 

## Experiments

The experiment class we mainly used was the `BoxedCaveExperiment`, which got
its name originally because agent opinions were boxed in between a value of
-S and S, where |S| ≤ 1. This experiment is initialized with a conected caveman
network structure with `n_caves` and `n_per_cave` agents in each cave. 
Each node in the network is an `Agent`. The Agent class is
simple with just two attributes, `opinions` and `weights`. 

The minimal example below
shows how to initialize a new `BoxedCaveExperiment` for K=2, run a few iterations of
agent updates, and plot the agent coordinates. See `scripts/runner.py` for
how this is used in our large-scale experiments. The `sge_scripts/` directory
contains job submission scripts for use on the Sun Grid Engine cluster we have
on campus. It may just work with other queue management systems.

```python
import matplotlib.pyplot as plt
import numpy as np

from experiments.within_box import BoxedCavesExperiment

n_caves = 20
n_per_cave = 5
S = 0.5
bce = BoxedCavesExperiment(n_caves, n_per_cave, S, K=2)

bce.iterate(5)

init_coords = np.array(bce.history['coords'][0])
final_coords = np.array(bce.history['final_coords'])

xi = init_coords[:, 0]; yi = init_coords[:, 1]
xf = final_coords[:, 0]; yf = final_coords[:, 1]

plt.plot(xi, yi, 'o', label='Initial opinions')
plt.plot(xf, yf, 'o', label='Opinions after five iterations')
lim = [-.55, .55]; plt.xlim(lim); plt.ylim(lim)
plt.axis('equal')
plt.legend()
```

This script can be run via `python simple_readme.py`. 
You should get an image like the one below

<img src="https://github.com/mt-digital/polarization/raw/master/simple_experiment.png" width="450">

## Advanced usage/future work

Here we demonstrate some advanced usage. Some advanced usage is straight
forward, but some of the advanced usage reveals some
of the technical debt amassed in the process of getting interesting
scientific results. Nonetheless, it is possible to use the codebase to do more than
model opinion dynamics on randomized connected caveman networks. 

First we add some time-varying communication noise to 

To run custom computational experiments of opinion dynamics, 
begin with the `Experiment` class:

```python
from macy import Experiment

# Default experiment gives a (n, k) connected caveman graph
# with n caves and k agents/nodes per cave. Agents begin with
# opinions drawn randomly, each opinion feature drawn from U(-1, 1).
ex = Experiment()
```

Experiments are not as self-contained as they might be.


## Data model

To sync experiment data we used the hierarchical data format, HDF5. HDF gives
a number of advantages. Data is read from memory on-demand, so loading the
file does not load the data, just pointers to the data. HDF is self-describing,
meaning that it includes its own metadata. The fact that it is hierarchical
is also powerful. So far, we have used this to store agent opinions at
various points in time, the network adjacency matrix, and a timeseries of 
polarization for all 100 trials in a single file, for each network
configuration, connected caveman, random short-range, and random long-range
conditions.

## Reference

Flache, A., & Macy, M. W. (2011). Small Worlds and Cultural Polarization. The Journal of Mathematical Sociology, 35(1-3), 146–176. doi: 10.1080/0022250X.2010.532261
