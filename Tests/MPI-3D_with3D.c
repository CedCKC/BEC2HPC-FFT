

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
  int i, j, k;

  ptrdiff_t N0 = 512, N1 = 512, N2 = 512;
  if (argc == 4) {
    N0 = atol(argv[1]);
	N1 = atol(argv[2]);
	N2 = atol(argv[3]);
  }
  
  double startwtime, endwtime;

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  fftw_mpi_init();

  /* distribution */
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_3d(N0, N1, N2, MPI_COMM_WORLD, &local_n0, &local_0_start);
  
  /* prepare a cosine wave */
  fftw_complex* in  = fftw_alloc_complex(alloc_local);
  fftw_complex* out = fftw_alloc_complex(alloc_local);
  
  fftw_plan p = fftw_mpi_plan_dft_3d(N0, N1, N2, in, out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

  for (i = 0; i < local_n0; ++i) for (j=0; j<N1; ++j) for (k=0; k<N2; ++k) {
	in[(i*N1+j)*N2+k][0] = cos(3 * 2*M_PI*i/N0);
  }
  
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


