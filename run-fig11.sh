#! /bin/bash
#

N_ITERATIONS=10000
for N_PER_CAVE in 3 5 10 20 30 40 50; do
    NAME="reproduce11b-$N_PER_CAVE"
    qsub -N $NAME -o "$NAME.log" -v N_PER_CAVE=$N_PER_CAVE -v N_ITERATIONS=$N_ITERATIONS -v OUTPUT_DIR="data/figure11b_test_3-28" run-fig11.sub
done
