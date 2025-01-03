CPP = nvcc
CXXFLAGS = --std=c++17 -O3 -arch=sm_35
COFASTFLAGS = --std=c++17 -Xcompiler -Ofast -arch=sm_35
SRCS = src/main.cu src/fluid_solver.cu src/EventManager.cpp 
SRCS2 = src/main84.cu src/fluid_solver.cu src/EventManager.cpp

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch ./cuda.sh

84:
	$(CPP) $(CXXFLAGS) $(SRCS2) -o fluid_sim
	sbatch ./cuda.sh

168:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda.sh

mem_check:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda_mem_check.sh

time_check:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda_time_check.sh

Ofast:
	$(CPP) $(COFASTFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda.sh