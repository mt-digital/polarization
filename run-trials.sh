#! /bin/bash
#
# Runs 10 trials of the polarization experiment for the configuration below.
# Eventually could add arguments to this for the number of trials,
# box width, noise level, etc.

for K in $(seq $1 $2); do
     qsub -v N_ITER=$N_ITER -v NETWORK="connected caveman" -v OUTPUT_DIR="reproduce_fm2011_3-28" -v K=$K run-experiment.sub
     qsub -v N_ITER=$N_ITER -v NETWORK="random short-range" -v OUTPUT_DIR="reproduce_fm2011_3-28" -v K=$K run-experiment.sub
     qsub -v N_ITER=$N_ITER -v NETWORK="random any-range" -v OUTPUT_DIR="reproduce_fm2011_3-28" -v K=$K run-experiment.sub
done
