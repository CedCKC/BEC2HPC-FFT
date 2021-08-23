/* 
   mpicc -O3 example1D-MPI.c -o example1D-MPI.ex -I/home/jeremie/opt/stow/fftw/3.3.5/include -L/home/jeremie/opt/stow/fftw/3.3.5/lib -lfftw3_mpi -lfftw3 -lm
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <fftw3-mpi.h>

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

  printf("rank=%d local_ni=%ld local_i_start=%ld local_n0=%ld local_o_start=%ld\n", rank, local_ni, local_i_start, local_no, local_o_start);
  
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
  fprintf(stdout, "rank=%d Time elapsed: %10.3E sec.\n", rank, (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000);
  
  fftw_destroy_plan(p);

#ifdef DEBUG
  for (i = 0; i < local_no; i++)
    printf("rank=%d freq: %3ld %+9.5f %+9.5f I\n", rank, i+local_o_start, out[i][0], out[i][1]);
#endif
  
  /* backward Fourier transform, save the result in 'in2' */
  fftw_complex *in2 = fftw_alloc_complex(alloc_local);
  fftw_plan q = fftw_mpi_plan_dft_1d(N, out, in2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(q);

  /* normalize */
  for (i = 0; i < local_ni; i++) {
    in2[i][0] *= 1./N;
  }
  
#ifdef DEBUG
  for (i = 0; i < local_ni; i++)
    printf("rank=%d recover: %3ld %+9.5f %+9.5f I vs. %+9.5f %+9.5f I\n",
	   rank, i+local_i_start, in[i][0], in[i][1], in2[i][0], in2[i][1]);
#endif

  fftw_cleanup();
  MPI_Finalize();

  return 0;
}


