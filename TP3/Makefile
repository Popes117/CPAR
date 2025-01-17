CPP = nvcc
CXXFLAGS = --std=c++17 -O3 -arch=sm_35
SRCS = TP3/src/main.cu TP3/src/fluid_solver.cu TP3/src/EventManager.cpp

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run: all
	sbatch TP3/cuda.sh

mem_check: all
	sbatch TP3/cuda_mem_check.sh

time_check: all
	sbatch TP3/cuda_time_check.sh
