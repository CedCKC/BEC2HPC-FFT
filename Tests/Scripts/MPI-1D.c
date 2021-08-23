

/* 
   mpicc -O3 example1D-MPI.c -o example1D-MPI.ex -I/home/jeremie/opt/stow/fftw/3.3.5/include -L/home/jeremie/opt/stow/fftw/3.3.5/lib -lfftw3_mpi -lfftw3 -lm
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <fftw3-mpi.h>
// FFTW MPI 1D
// Return execution time of one MPI process in .6e format


//#define DEBUG

int main(int argc, char **argv) {
  int i;

  ptrdiff_t N = 4096;
  if (argc == 2) {
    N = atol(argv[1]);
  }
  
  double startwtime, endwtime;

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  fftw_mpi_init();

  /* distribution */
  ptrdiff_t local_ni, local_i_start, local_no, local_o_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_1d(N, MPI_COMM_WORLD,FFTW_FORWARD, FFTW_ESTIMATE,
						 &local_ni, &local_i_start, &local_no, &local_o_start);
  
  /* prepare a cosine wave */
  fftw_complex* in  = fftw_alloc_complex(alloc_local);
  for (i = 0; i < local_ni; ++i) {
    in[i][0] = cos(3 * 2*M_PI*i/N);
  }
  
  /* forward Fourier transform, save the result in 'out' */
  fftw_complex* out = fftw_alloc_complex(alloc_local);
  fftw_plan p = fftw_mpi_plan_dft_1d(N, in, out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

  struct timeval t0, t1; // timer
  gettimeofday(&t0, NULL);

  fftw_execute(p);

  gettimeofday(&t1, NULL);
  
  fftw_destroy_plan(p);
  fftw_cleanup();
  MPI_Finalize();
  
  if (rank == 0)
	fprintf(stdout, "%.6e", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000);

  return 0;
}


