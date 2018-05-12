#! /bin/bash
#
# Runs 10 trials of the polarization experiment for the configuration below.
# Eventually could add arguments to this for the number of trials,
# box width, noise level, etc.

N_ITERATIONS = 10000

for N_PER_CAVE in 3 5 10 20 30 40 50; do
     qsub -v N_PER_CAVE=$N_PER_CAVE -v N_ITERATIONS=$N_ITERATIONS -v OUTPUT_DIR="data/figure11b_test_3-28" run-fig11.sub
done
