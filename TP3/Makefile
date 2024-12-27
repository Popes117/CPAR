CPP = nvcc
CXXFLAGS = --std=c++17 -O3 
SRCS = src/main.cu src/fluid_solver.cu src/EventManager.cpp 
SRCS2 = src/main.cu src/fluid_solver1.cu src/EventManager.cpp

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch ./cuda.sh

comp:
	$(CPP) $(CXXFLAGS) $(SRCS2) -o fluid_sim
