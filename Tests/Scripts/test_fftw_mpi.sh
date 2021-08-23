#!/bin/bash

# Print performance table for FFTW MPI C scripts returning a single time per execution

# ./test_fftw_mpi.sh <filepath> <arguments>

# Examples:
# ./test_fftw_mpi.sh MPI-1D.c
# ./test_fftw_mpi.sh MPI-1D.c 500000
# ./test_fftw_mpi.sh MPI-2D_with1D.c 4096 4096

# Installation directory for FFTW with ./configure:
pkgdir=$HOME/opt



filepath=$1
shift

mpicc -O3 $filepath -o test_o -I$pkgdir/include -L$pkgdir/lib -lfftw3_mpi -lfftw3 -lm

# Compute the number of physical cores (= max number of MPI processes)
id_last_core=$(lscpu -p --parse=Core | tail -1)
nb_cores=$(expr $id_last_core + 1)

nb_procs=1
n=0
while [ $nb_procs -le $nb_cores ]
do
	n=$(expr $n + 1)
	test_time[$n-1]=$(mpirun --mca btl_base_warn_component_unused 0 --mca mpi_warn_on_fork 0 -np $nb_procs ./test_o $*)
	printf -v test_time[$n-1] "%.8f" "${test_time[$n-1]}"		# Convert to decimal notation
	test_speedup[$n-1]=$(echo ${test_time[0]}/${test_time[$n-1]} | bc -l)
	test_efficiency[$n-1]=$(echo ${test_speedup[$n-1]}/$nb_procs | bc -l)
	nb_procs=$((1<<$n))
done

rm test_o

printf "| Nb procs | Time (s) | Speedup  | Efficiency |\n|----------|----------|----------|------------|\n"
for ((i = 0 ; i < $n ; i++))
do
	printf "|       %02d | %.2e | %.2f     | %.2f       |\n" $((1<<$i)) ${test_time[$i]} ${test_speedup[$i]} ${test_efficiency[$i]}
done