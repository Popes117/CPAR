CPP = nvcc
CXXFLAGS = --std=c++17 -O3 -arch=sm_35
COFASTFLAGS = --std=c++17 -Xcompiler -Ofast -arch=sm_35
SRCS = src/main.cu src/fluid_solver1.cu src/EventManager.cpp 
SRCS2 = src/main84.cu src/fluid_solver2.cu src/EventManager.cpp
SRCS3 = src/main.cu src/fluid_solver2.cu src/EventManager.cpp

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch ./cuda.sh

84:
	$(CPP) $(CXXFLAGS) $(SRCS2) -o fluid_sim
	sbatch ./cuda.sh

168fluid2:
	$(CPP) $(CXXFLAGS) $(SRCS3) -o fluid_sim
	sbatch ./cuda.sh

168fluid1:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim
	sbatch ./cuda.sh

fast:
	$(CPP) $(COFASTFLAGS) $(SRCS3) -o fluid_sim
	sbatch ./cuda.sh