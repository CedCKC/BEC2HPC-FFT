// FFTW threads 2D with FFTW1D (stride)
// Return execution time of one MPI process in .6e format
// Arguments: nb_threads N0 N1

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include <fftw3.h>

fftw_plan plan_many_1d_dft(fftw_complex* in, fftw_complex* out, int n0, int n1, int dist, int stride) {
  /* Transform each row of a local 2d array with n0_local rows and n1 columns */
  
  int rank = 1;        /* we are computing 1d transforms */
  int n[] = {n1};  /* 1d transforms of length in.n1 */
  int howmany = n0;
  int idist = dist, odist = dist; /* the distance in memory between the first element
					 of the first array and the first element of the second array */
  int istride = stride, ostride = stride; /* distance between two elements in 
				   the same array */
  int *inembed = n, *onembed = n;
  
  fftw_plan plan_dft = fftw_plan_many_dft(rank, n, howmany,
					  in, inembed, istride, idist,
					  out, onembed, ostride, odist,
					  FFTW_FORWARD, FFTW_ESTIMATE);
  
  return plan_dft;
}

int main(int argc, char **argv)
{
	int i, j;

	int N = 8, M = 16;
	int nb_threads = 1;
	if (argc > 1){
		nb_threads = atol(argv[1]);
		if (argc == 4){
			N = atol(argv[2]);
			M = atol(argv[3]);
		}
	}

	omp_set_num_threads(nb_threads);
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());

	fftw_complex *in, *dft1, *dft2;
	fftw_plan p1, p2;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	dft1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	dft2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);

	p1 = plan_many_1d_dft(in, dft1, N, M, M, 1);
	p2 = plan_many_1d_dft(dft1, dft2, M, N, 1, M);
	
	for (i=0; i<N; i++) for (j=0; j<M; j++)
	{
		in[i*M+j][0] = cos(3 * 2*M_PI*i/N);
		in[i*M+j][1] = 0.0;
	}
	
	struct timeval t0, t1; // timer
	gettimeofday(&t0, NULL);
	
	fftw_execute(p1);
	fftw_execute(p2);
	
	gettimeofday(&t1, NULL);
	
	fftw_destroy_plan(p1);
	fftw_destroy_plan(p2);
	fftw_cleanup_threads();

	fprintf(stdout, "%.6e", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000);

	return 0;
}