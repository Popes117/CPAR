CPP = g++ -Wall -O3 -funroll-loops -ftree-vectorize -ffast-math -march=native -fopenmp -mavx -std=c++11 
PROF_FLAGS = -pg
SRCS = src/main.cpp src/fluid_solver1.cpp src/EventManager.cpp
#THREADS = 2
THREADS ?= 2

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

runseq:
	./fluid_sim

runpar:
#	./fluid_sim $(THREADS) -> exige mudar o código da main e não sei quão boa ideia é isso
	OMP_NUM_THREADS=$(THREADS) 
	./fluid_sim	

runclus: all
	export OMP_NUM_THREADS=2
	srun --partition=day --nodelist=compute-134-116 --cpus-per-task=2 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim

runclus2 : all 
	sbatch --partition=day --nodelist=compute-134-116 --cpus-per-task=24 run.sh

runclus3: all
	export OMP_NUM_THREADS=2
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=2 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=4
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=4 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=8
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=8 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=16
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=16 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=19
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=19 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=20
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=20 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=21
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=21 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=22
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=22 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=32
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=32 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=48
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=48 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim

runclus4: all 
	sbatch --partition=day --constraint=c24 --ntasks=1 --cpus-per-task=48 --exclusive run2.sh

run: all
	OMP_NUM_THREADS=2
	perf stat -d ./fluid_sim
	OMP_NUM_THREADS=4
	perf stat -d ./fluid_sim
	OMP_NUM_THREADS=8
	perf stat -d ./fluid_sim
	OMP_NUM_THREADS=16
	perf stat -d ./fluid_sim
	OMP_NUM_THREADS=32
	perf stat -d ./fluid_sim

run2: all
	export OMP_NUM_THREADS=16
	perf stat -d ./fluid_sim

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
