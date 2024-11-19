#!/bin/sh
#
#SBATCH --exclusive     # exclusive node for the job
#SBATCH --time=05:00    # allocation for 2 minutes

for threads in {1..48}
do
    export OMP_NUM_THREADS=$threads
    echo "Running with OMP_NUM_THREADS=$threads"
    time ./fluid_sim
done