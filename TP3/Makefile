CPP = nvcc
CXXFLAGS = --std=c++17 -O3 -arch=sm_35
COFASTFLAGS = --std=c++17 -Xcompiler -Ofast -arch=sm_35
SRCS = src/main.cu src/fluid_solver.cu src/EventManager.cpp 

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run: all
	sbatch ./cuda.sh

mem_check:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda_mem_check.sh

time_check:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda_time_check.sh