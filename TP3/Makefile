CPP = nvcc
CXXFLAGS = --std=c++17 -O3 
SRCS = src/main.cu src/fluid_solver.cu src/EventManager.cpp 

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch ./cuda.sh