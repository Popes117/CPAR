CPP = g++ -Wall
CFLAGS = -Ofast -funroll-loops -ftree-vectorize -ffast-math -march=native -fopenmp -mavx -std=c++11
SRCS = src/main.cpp src/fluid_solver1.cpp src/EventManager.cpp
#THREADS = 2
THREADS ?= 2

all:
	$(CPP) $(CFLAGS) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

runseq:
	./fluid_sim

runpar:
#	./fluid_sim $(THREADS) -> exige mudar o código da main e não sei quão boa ideia é isso
	OMP_NUM_THREADS=$(THREADS) ./fluid_sim	


PROF_FLAGS = -pg

prof:
	$(CPP) $(CFLAGS) $(PROF_FLAGS) $(SRCS) -o fluid_sim

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
