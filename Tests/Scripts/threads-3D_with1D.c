// FFTW threads 2D with FFTW1D (transpose)
// Return execution time of one MPI process in .6e format
// Arguments: nb_threads N0 N1 N2
// N0, N1, N2 must be equal and multiples of 16

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

void print_array(fftw_complex* a, int n0, int n1, int n2) {
	int i, j, k, index;
	for (i = 0; i < n0; i++) {
		printf("--------------------------\n");
		for (j = 0; j < n1; j++) {
			printf("|");
			for (k = 0; k < n2; k++) {
				index = (i*n1 + j)*n2 + k;
				printf(" %.1f+%.1fi ;", crealf(a[index]), cimagf(a[index]));
			}
			printf("|\n");
		}
	}
	printf("--------------------------\n");
}

fftw_plan plan_many_1d_dft(fftw_complex* in, fftw_complex* out, int n0, int n1, int n2) {
  /* Transform each row of a local 2d array with n0_local rows and n1 columns */
  
  int rank = 1;        /* we are computing 1d transforms */
  int n[] = {n2};  /* 1d transforms of length in.n1 */
  int howmany = n0*n1;
  int idist = n2, odist = n2; /* the distance in memory between the first element
					 of the first array and the first element of the second array */
  int istride = 1, ostride = 1; /* distance between two elements in 
				   the same array */
  int *inembed = n, *onembed = n;
  
  fftw_plan plan_dft = fftw_plan_many_dft(rank, n, howmany,
					  in, inembed, istride, idist,
					  out, onembed, ostride, odist,
					  FFTW_FORWARD, FFTW_ESTIMATE);
  
  return plan_dft;
}

// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
inline void transpose_scalar_block(fftw_complex *A, fftw_complex *B, const int lda, const int ldb, const int block_size, const int howmany_elements) {
  //#pragma omp parallel for
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
			for (int k=0; k<howmany_elements; k++) {
				B[(j*ldb + i)*howmany_elements + k] = A[(i*lda +j)*howmany_elements + k];
			}
        }
    }
}

//inline
void transpose_block(fftw_complex *A, fftw_complex *B, const int n, const int m, const int lda, const int ldb, const int block_size, const int howmany_elements) {
#pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[(i*lda +j)*howmany_elements], &B[(j*ldb + i)*howmany_elements], lda, ldb, block_size, howmany_elements);
        }
    }
}

int main(int argc, char **argv)
{
	int i, j, k;

	int N0 = 16, N1 = 16, N2 = 16;
	int nb_threads = 1;
	if (argc > 1){
		nb_threads = atol(argv[1]);
		if (argc == 5){
			N0 = atol(argv[2]);
			N1 = atol(argv[3]);
			N2 = atol(argv[4]);
		}
	}
	assert(N0 == N1);
	assert(N0 == N2);

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

	fftw_complex *data1, *data2;
	fftw_plan p1, p2, p3;

	data1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N0 * N1 * N2);
	data2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N0 * N1 * N2);
	
	// Check alignment
#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
	int lda = ROUND_UP(N0, 16);
	int ldb = ROUND_UP(N1, 16);
	int ldc = ROUND_UP(N2, 16);
	//printf("M, lda, N, ldb: %d %d %d %d\n", M, lda, N, ldb);
	assert(lda == N0);
	assert(ldb == N1);
	assert(ldc == N2);
	
	p1 = plan_many_1d_dft(data1, data2, N0, N1, N2);
	p2 = plan_many_1d_dft(data2, data1, N1, N2, N0);
	p3 = plan_many_1d_dft(data1, data2, N2, N0, N1);
	
	int index;
	for (i=0; i<N0; i++) for (j=0; j<N1; j++) for (k=0; k<N2; k++)
	{
		index = (i*N1+j)*N2+k;
		data1[index] = cos((1+k) * (3+j) * 2*M_PI*i/N0);
	}
	
	
	
	struct timeval t0, t1; // timer
	gettimeofday(&t0, NULL);
	
	// in -> dft1
	fftw_execute(p1);
	
	// dft1 (N0, N1, N2) -> dft1_trans1 (N1, N0, N2)
	transpose_block(data2, data1, N0, N1, ldb, lda, 16, N2);
	
	// dft1_trans1 (N1, N0, N2) -> dft1_trans2 (N1, N2, N0)
	for (j=0; j<N1; j++) {
		transpose_block(data1 + j*N0*N2, data2 + j*N0*N2, N0, N2, ldc, lda, 16, 1);
	}
	
	// dft1_trans2 -> dft2
	fftw_execute(p2);
	
	// dft2 (N1, N2, N0) -> dft2_trans1 (N2, N1, N0)
	transpose_block(data1, data2, N1, N2, lda, ldb, 16, N0);
	
	// dft2_trans1 (N2, N1, N0) -> dft2_trans2 (N2, N0, N1)
	for (k=0; k<N2; k++) {
		transpose_block(data2 + k*N0*N1, data1 + k*N0*N1, N1, N0, lda, ldb, 16, 1);
	}
	
	// dft2_trans1 -> dft3
	fftw_execute(p3);
	
	// dft3 (N2, N0, N1) -> dft3_trans1 (N0, N2, N1)
	transpose_block(data2, data1, N2, N0, lda, ldc, 16, N1);
	
	// dft3_trans1 (N0, N2, N1) -> dft3_trans2 (N0, N1, N2)
	for (i=0; i<N0; i++) {
		transpose_block(data1 + i*N1*N2, data2 + i*N1*N2, N2, N1, ldb, ldc, 16, 1);
	}
	
	gettimeofday(&t1, NULL);
	
	fftw_destroy_plan(p1);
	fftw_destroy_plan(p2);
	fftw_destroy_plan(p3);
	fftw_cleanup_threads();
	
	// printf("out\n");
	// print_array(data2, N0, N1, N2);
	// printf("\n");
	
	free(data1);
	free(data2);

	fprintf(stdout, "%.6e", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000);

	return 0;
}