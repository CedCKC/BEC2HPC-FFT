// FFTW threads 2D with FFTW1D (transpose)
// Return execution time of one MPI process in .6e format
// Arguments: nb_threads N0 N1

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <complex.h>
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

// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
inline void transpose_scalar_block(fftw_complex *A, fftw_complex *B, const int lda, const int ldb, const int block_size) {
  //#pragma omp parallel for
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
			B[j*ldb + i] = A[i*lda +j];
        }
    }
}

//inline
void transpose_block(fftw_complex *A, fftw_complex *B, const int n, const int m, const int lda, const int ldb, const int block_size) {
#pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*lda +j], &B[j*ldb + i], lda, ldb, block_size);
        }
    }
}

int main(int argc, char **argv)
{
	int i, j;

	int N = 16, M = 16;
	int nb_threads = 1;
	if (argc > 1){
		nb_threads = atol(argv[1]);
		if (argc == 4){
			N = atol(argv[2]);
			M = atol(argv[3]);
		}
	}
	assert(N == M);

	int nthreads, tid;
	/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel private(nthreads, tid)
	{
	  /* Obtain thread number */
	  tid = omp_get_thread_num();
	  //printf("Hello World from thread = %d\n", tid);
	  
	  /* Only master thread does this */
	  if (tid == 0) 
	    {
	      nthreads = omp_get_num_threads();
	      assert(nthreads == nb_threads);
	      //printf("Number of OpenMP threads = %d\n", nthreads);
	    }
	  
	} /* All threads join master thread and disband */
	
	
	fftw_init_threads();
	fftw_plan_with_nthreads(nb_threads);

	fftw_complex *in, *dft1, *dft1_transpose, *dft2, *dft2_transpose;
	fftw_plan p1, p2;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	dft1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	dft1_transpose = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	dft2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	dft2_transpose = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
	
	// Check alignment
#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
	int lda = ROUND_UP(M, 16);
	int ldb = ROUND_UP(N, 16);
	//printf("M, lda, N, ldb: %d %d %d %d\n", M, lda, N, ldb);
	assert(lda == M);
	assert(ldb == N);
	
	p1 = plan_many_1d_dft(in, dft1, N, M, M, 1);
	p2 = plan_many_1d_dft(dft1_transpose, dft2, M, N, N, 1); //plan_many_1d_dft(dft1, dft2, M, N, 1, M);
	
	for (i=0; i<N; i++) for (j=0; j<M; j++)
	{
		in[i*M+j] = cos(3 * 2*M_PI*i/N);
	}
	
	struct timeval t0, t1; // timer
	gettimeofday(&t0, NULL);
	
	fftw_execute(p1);
	transpose_block(dft1, dft1_transpose, N, M, M, N, 16);
	fftw_execute(p2);
	transpose_block(dft2, dft2_transpose, M, N, N, M, 16);
	
	gettimeofday(&t1, NULL);
	
	fftw_destroy_plan(p1);
	fftw_destroy_plan(p2);
	fftw_cleanup_threads();

	fprintf(stdout, "%.6e", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000);

	return 0;
}