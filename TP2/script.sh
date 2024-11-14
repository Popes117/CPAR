#!/bin/bash

module load gcc/11.2.0

perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time ./fluid_sim