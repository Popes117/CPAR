CPP = g++ -Wall -Ofast -funroll-loops -ftree-vectorize -ffast-math -march=native -mavx -std=c++11 
PROF_FLAGS = -pg
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp
BIN = bin/

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm $(BIN)fluid_sim
	@echo Done.

prof:
	$(CPP) $(CFLAGS) $(PROF_FLAGS) $(SRCS) -o $(BIN)fluid_sim

srun-prof: all
	sbatch --partition=day --cpus-per-task=1 --exclusive --wrap="perf stat -r 3 -e instructions,cycles,branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time ./fluid_sim"

run-prof: prof
	./$(BIN)fluid_sim 

graph-prof: run-prof
	gprof $(BIN)fluid_sim > main.gprof
	./gprof2dot.py -o output.dot main.gprof
	rm gmon.out
	dot -Tpng -o output.png output.dot

clean-prof:
	@echo Cleaning up...
	@rm -f $(BIN)fluid_sim gmon.out main.gprof output.dot output.png
	@echo Done.
