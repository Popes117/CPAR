CPP = g++ -Wall -Ofast -funroll-loops -ftree-vectorize -ffast-math -march=native -mavx -std=c++11 
PROF_FLAGS = -pg
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
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
	./prof_md 

graph-prof: run-prof
	gprof prof_md > main.gprof
	./gprof2dot.py -o output.dot main.gprof
	rm gmon.out
	dot -Tpng -o output.png output.dot

clean-prof:
	@echo Cleaning up...
	@rm -f prof_md gmon.out main.gprof output.dot output.png
	@echo Done.
