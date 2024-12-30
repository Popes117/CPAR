#!/bin/bash
#SBATCH --time=2:00
#SBATCH --partition=cpar
#SBATCH --constraint=k20

time nvprof ./fluid_sim 
