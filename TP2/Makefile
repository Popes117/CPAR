CPP = g++ -Wall -Ofast -funroll-loops -ftree-vectorize -ffast-math -march=native -fopenmp -mavx -std=c++11 
PROF_FLAGS = -pg
SRCS = src/main.cpp src/fluid_solver1.cpp src/EventManager.cpp
BIN = bin/

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

prof:
	$(CPP) $(CFLAGS) $(PROF_FLAGS) $(SRCS) -o fluid_sim

srun-prof: prof
	srun --partition=cpar --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim

run-prof: prof
	./fluid_sim 

graph-prof: run-prof
	gprof fluid_sim > main.gprof
	./gprof2dot.py -o output.dot main.gprof
	rm gmon.out
	dot -Tpng -o output.png output.dot

clean-prof:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out main.gprof output.dot output.png
	@echo Done.
