#! /bin/bash
#
# Runs 10 trials of the polarization experiment for the configuration below.
# Eventually could add arguments to this for the number of trials,
# box width, noise level, etc.

for i in $(seq 1 10); do
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.1 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.2 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.3 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.4 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.05 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.15 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.25 run-experiment.sub
     qsub -v OUTPUT_DIR="data/outputs_$i" -v BW_MIN=0.35 run-experiment.sub
done
