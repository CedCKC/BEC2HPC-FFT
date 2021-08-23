#!/bin/bash

# Print performance table for threaded FFTW C scripts returning a single time per execution

# ./test_fftw_threads.sh <filepath> <arguments>

# Examples:
# ./test_fftw_threads.sh threads-1D.c
# ./test_fftw_threads.sh threads-1D.c 500000
# ./test_fftw_threads.sh threads-2D_with2D.c 4096 4096

# Installation directory for FFTW with ./configure:
pkgdir=$HOME/opt



filepath=$1
shift

gcc -O3 $filepath -o test_o -I$pkgdir/include -L$pkgdir/lib -fopenmp -lfftw3_omp -lfftw3 -lm

# Compute the number of CPU's (= max number of threads before overlap)
id_last_cpu=$(lscpu -p --parse=CPU | tail -1)
nb_cpus=$(expr $id_last_cpu + 1)

nb_procs=1
n=0
while [ $nb_procs -le $nb_cpus ]
do
	n=$(expr $n + 1)
	export OMP_NUM_THREADS=$nb_procs
	test_time[$n-1]=$(./test_o $nb_procs $*)
	printf -v test_time[$n-1] "%.8f" "${test_time[$n-1]}"		# Convert to decimal notation
	test_speedup[$n-1]=$(echo ${test_time[0]}/${test_time[$n-1]} | bc -l)
	test_efficiency[$n-1]=$(echo ${test_speedup[$n-1]}/$nb_procs | bc -l)
	nb_procs=$((1<<$n))
done

rm test_o

printf "| Nb threads | Time (s) | Speedup  | Efficiency |\n|------------|----------|----------|------------|\n"
for ((i = 0 ; i < $n ; i++))
do
	printf "|         %02d | %.2e | %.2f     | %.2f       |\n" $((1<<$i)) ${test_time[$i]} ${test_speedup[$i]} ${test_efficiency[$i]}
done