CPP = g++ -Wall -O3 -funroll-loops -ftree-vectorize -ffast-math -march=native -fopenmp -mavx -std=c++11 
PROF_FLAGS = -pg
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

runseq: all
	./fluid_sim

runpar: all 
	export OMP_NUM_THREADS=22
	./fluid_sim

runseq2: all
	sbatch --partition cpar --cpus-per-task=40 ./perf.sh seqStat

runpar2: all
	sbatch --partition cpar --cpus-per-task=40 ./perf.sh parStat

clusreport : all 
	sbatch --partition cpar --cpus-per-task=40 ./perf.sh report 

runclus: all
	export OMP_NUM_THREADS=1
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=1 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=2
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=2 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=4
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=4 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=8
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=8 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=12
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=12 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
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
	export OMP_NUM_THREADS=24
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=24 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=28
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=28 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=32
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=32 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=36
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=36 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=40
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=40 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=44
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=44 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
	export OMP_NUM_THREADS=48
	srun --partition day --constraint=c24 --ntasks=1 --cpus-per-task=48 --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim

runclus2: all 
	sbatch --partition=day --constraint=c24 --ntasks=1 --cpus-per-task=48 --exclusive run.sh

runlocal: all
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

prof:
	$(CPP) $(CFLAGS) $(PROF_FLAGS) $(SRCS) -o fluid_sim

perf-report: prof
	perf record ./fluid_sim
	perf report -n --stdio > perfreport.txt

run-prof: prof
	./fluid_sim 

graph-prof: run-prof
	gprof fluid_sim > gprof.txt
	gprof fluid_sim | gprof2dot | dot -Tpng -o graph.png

graph-prof2: run-prof
	gprof fluid_sim > main.gprof
	./gprof2dot.py -o output.dot main.gprof
	rm gmon.out
	dot -Tpng -o output.png output.dot

clean-prof:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out main.gprof output.dot output.png
	@echo Done.
