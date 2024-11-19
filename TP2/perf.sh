#!/bin/bash
#
#SBATCH --exclusive     # exclusive node for the job
#SBATCH --time=02:00    # allocation for 2 minutes

if [ "$1" == "seqStat" ]; then
    echo "Running perf stat..."
    export OMP_NUM_THREADS=1
    perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
    echo "Running with OMP_NUM_THREADS=1"
    time ./fluid_sim
elif [ "$1" == "parStat" ]; then
    echo "Running perf stat..."
    export OMP_NUM_THREADS=20
    perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
    echo "Running with OMP_NUM_THREADS=20"
    time ./fluid_sim
elif [ "$1" == "report" ]; then
    echo "Running perf record and generating report..."
    perf record ./fluid_sim
    perf report -n --stdio > perfreport.txt
    echo "Perf report saved to 'perfreport.txt'"
else
    echo "Usage: $0 {stat|report}"
    exit 1
fi