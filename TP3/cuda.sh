#!/bin/bash
#SBATCH --time=2:00
#SBATCH --partition=cpar
#SBATCH --constraint=k20

nvprof ./fluid_sim 
