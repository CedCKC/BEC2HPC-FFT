#!/bin/bash

# Print performance table for FluidFFT MPI Python scripts returning a single time per execution

# ./test_fluid_mpi.sh <filepath> <arguments>

# Examples:
# ./test_fluid_mpi.sh MPI-2D_with1D.py
# ./test_fluid_mpi.sh MPI-2D_with1D.py 4096 4096
# ./test_fluid_mpi.sh MPI-2D_with2D.py 50000 40000

# Local installation directory for FFTW:
pkgdir=$HOME/.local



filepath=$1
shift

export LD_LIBRARY_PATH=$pkgdir/lib:$LD_LIBRARY_PATH

# Compute the number of physical cores (= max number of MPI processes)
id_last_core=$(lscpu -p --parse=Core | tail -1)
nb_cores=$(expr $id_last_core + 1)

nb_procs=1
n=0
while [ $nb_procs -le $nb_cores ]
do
	n=$(expr $n + 1)
	test_time[$n-1]=$(mpirun --mca btl_base_warn_component_unused 0 --mca mpi_warn_on_fork 0 -np $nb_procs python3 $filepath $*)
	printf -v test_time[$n-1] "%.8f" "${test_time[$n-1]: -12}"		# Convert to decimal notation
	test_speedup[$n-1]=$(echo ${test_time[0]}/${test_time[$n-1]} | bc -l)
	test_efficiency[$n-1]=$(echo ${test_speedup[$n-1]}/$nb_procs | bc -l)
	nb_procs=$((1<<$n))
done

printf "| Nb procs | Time (s) | Speedup  | Efficiency |\n|----------|----------|----------|------------|\n"
for ((i = 0 ; i < $n ; i++))
do
	printf "|       %02d | %.2e | %.2f     | %.2f       |\n" $((1<<$i)) ${test_time[$i]} ${test_speedup[$i]} ${test_efficiency[$i]}
done