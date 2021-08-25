// FFTW MPI 2D with FFTW1D
// Return execution time of one MPI process in .6e format

/* 
   mpicc -O3 example2D-MPI.c -o example2D-MPI.ex -I/home/jeremie/opt/stow/fftw/3.3.5/include -L/home/jeremie/opt/stow/fftw/3.3.5/lib -lfftw3_mpi -lfftw3 -lm
*/
#include <unistd.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include <complex.h>
#include <fftw3-mpi.h>

struct array {
  // 1d block distribution of the data, distributed along the first dimension (called n0).
  // the current process stores the rows local_0_start to local_0_start+local_n0-1
  char* name;
  fftw_complex* data;
  ptrdiff_t     n0;
  ptrdiff_t     n1;
  ptrdiff_t		n2;
  int           block;
  ptrdiff_t     local_n0;
  ptrdiff_t     local_n0_start;
  ptrdiff_t     alloc_local;
};

void print_array(struct array* a) {
	int i, j, k, index;
	for (i = 0; i < a->local_n0; i++) {
		printf("--------------------------\n");
		for (j = 0; j < a->n1; j++) {
			printf("|");
			for (k = 0; k < a->n2; k++) {
				index = (i*a->n1 + j)*a->n2 + k;
				printf(" %.1f+%.1fi ;", crealf(a->data[index]), cimagf(a->data[index]));
			}
			printf("|\n");
		}
	}
	printf("--------------------------\n");
}

void init_array(struct array* a, char* name, ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2) {
  a->name = name;
  a->n0 = n0;
  a->n1 = n1;
  a->n2 = n2;
  a->block = FFTW_MPI_DEFAULT_BLOCK;
  a->data = NULL;
}

void init_array_transpose(struct array* in, char* in_name, struct array *out, char* out_name, ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2) {
  init_array(in,  in_name,  n0, n1, n2);
  init_array(out, out_name, n1, n0, n2);
  
  in->alloc_local = out->alloc_local = fftw_mpi_local_size_3d_transposed(n0, n1, n2, MPI_COMM_WORLD,
									 &in->local_n0, &in->local_n0_start,
									 &out->local_n0, &out->local_n0_start);
  
}

void alloc_array(struct array* a) {
  a->data  = fftw_alloc_complex(a->alloc_local);
}

void free_array(struct array* a) {
  free(a->data);
}
  
fftw_plan plan_many_1d_dft(struct array* in, struct array* out) {
  assert(in->n0 == out->n0);
  assert(in->n1 == out->n1);
  assert(in->n2 == out->n2);
  assert(in->local_n0 == out->local_n0);

  /* Transform each row of a local 2d array with n0_local rows and n1 columns */
  
  int rank = 1;        /* we are computing 1d transforms */
  int n[] = {in->n2};  /* 1d transforms of length in.n2 */
  int howmany = in->local_n0*in->n1;
  int idist = in->n2, odist = in->n2; /* the distance in memory between the first element
					 of the first array and the first element of the second array */
  int istride = 1, ostride = 1; /* distance between two elements in 
				   the same array */
  int *inembed = n, *onembed = n;
  
  fftw_plan plan_dft = fftw_plan_many_dft(rank, n, howmany,
					  in->data, inembed, istride, idist,
					  out->data, onembed, ostride, odist,
					  FFTW_FORWARD, FFTW_ESTIMATE);
  
  if (plan_dft == NULL) { printf("plan_many_1d_dft ERROR: plan_dft=%p\n", plan_dft); }
  assert(plan_dft != NULL);
  
  return plan_dft;
}

fftw_plan plan_many_transpose(struct array* in, struct array* out, int howmany_elements) {
  // howmany_elements: nb of elements between first element along n1 and second element along n1
  int howmany = 2*howmany_elements; // transpose complex number (ie. double*2)

  // use advanced interface because we are transposing complex vectors (howmany=2)
  fftw_plan p_transpose = fftw_mpi_plan_many_transpose(in->n0, in->n1, howmany, in->block, out->block, (double *)in->data, (double *)out->data, MPI_COMM_WORLD, FFTW_ESTIMATE);
  assert(p_transpose != NULL);

  return p_transpose;
}

void n0_transpose(struct array* in, struct array* out) {
	int i, j, k;
	for (i = 0; i < in->local_n0; i++) for (j = 0; j < in->n1; j++) for (k = 0; k < in->n2; k++) {
		out->data[(i*in->n2+k)*in->n1+j] = in->data[(i*in->n1+j)*in->n2+k];
	}
}



int main(int argc, char **argv)
{
  ptrdiff_t N0 = 2, N1 = 4, N2 = 6;

  if (argc == 4) {
    N0 = atol(argv[1]);
    N1 = atol(argv[2]);
	N2 = atol(argv[3]);
  }

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
  
  fftw_mpi_init();

  // FFTW uses a 1d block distribution of the data, distributed along the first two dimensions. 
  
  // in -> dft1 -> dft1_trans1 -> dft1_trans2
  //	-> dft2 -> dft2_trans1 -> dft2_trans2
  //	-> dft3 -> dft3_trans1 -> dft3_trans2
  struct array dft1, dft1_trans1;
  init_array_transpose(&dft1, "dft1", &dft1_trans1, "dft1_trans1", N0, N1, N2);

  struct array in; 			// => same distribution as dft1
  init_array(&in, "in", N0, N1, N2);
  in.alloc_local = dft1.alloc_local; // not mandatory
  in.local_n0 = dft1.local_n0;
  in.local_n0_start = dft1.local_n0_start;
  
  
  struct array dft2, dft2_trans1;
  init_array_transpose(&dft2, "dft2", &dft2_trans1, "dft2_trans1", N1, N2, N0);
  
  struct array dft1_trans2; // => same distribution as dft2
  init_array(&dft1_trans2, "dft1_trans2", N1, N2, N0);
  dft1_trans2.alloc_local = dft2.alloc_local; // not mandatory
  dft1_trans2.local_n0 = dft2.local_n0;
  dft1_trans2.local_n0_start = dft2.local_n0_start;
  
  
  struct array dft3, dft3_trans1;
  init_array_transpose(&dft3, "dft3", &dft3_trans1, "dft3_trans1", N2, N0, N1);
  
  struct array dft2_trans2; // => same distribution as dft3
  init_array(&dft2_trans2, "dft2_trans2", N2, N0, N1);
  dft2_trans2.alloc_local = dft3.alloc_local; // not mandatory
  dft2_trans2.local_n0 = dft3.local_n0;
  dft2_trans2.local_n0_start = dft3.local_n0_start;
  
  
  struct array dft3_trans2; // => same distribution as in
  init_array(&dft3_trans2, "dft3_trans2", N0, N1, N2);
  dft3_trans2.alloc_local = in.alloc_local; // not mandatory
  dft3_trans2.local_n0 = in.local_n0;
  dft3_trans2.local_n0_start = in.local_n0_start;


  fftw_plan p_row, p_transpose;


  alloc_array(&in);

  /* Initialize 'in' */
  int i, j, k, i_global;
  for (i = 0; i < in.local_n0; i++) for (j = 0; j < N1; j++) for (k = 0; k < N2; k++) {
	i_global = in.local_n0_start + i;
	in.data[(i*N1+j)*N2+k] = cos((3+j) * 2*M_PI*i_global/N0);
   }
   
   
   

  //
  // dft1
  //
  
  alloc_array(&dft1);

  double t = 0, t_ = 0;
  struct timeval t0, t1; // timer
    
  p_row = plan_many_1d_dft(&in, &dft1);
  gettimeofday(&t0, NULL);
  
  fftw_execute(p_row);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  fftw_destroy_plan(p_row);
  free_array(&in);
  
  
  
  
  //
  // dtf1_trans1
  //
  
  alloc_array(&dft1_trans1);
  
  /* Transpose (N0, N1, N2) -> (N1, N0, N2) */

  p_transpose = plan_many_transpose(&dft1, &dft1_trans1, N2);
  
  gettimeofday(&t0, NULL);

  fftw_execute(p_transpose);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;

  fftw_destroy_plan(p_transpose);
  free_array(&dft1);
  
	
  //
  // dtf1_trans2
  //
  
  alloc_array(&dft1_trans2);
  
  /* Transpose (N1, N0, N2) -> (N1, N2, N0) */
  
  gettimeofday(&t0, NULL);
  
  n0_transpose(&dft1_trans1, &dft1_trans2);
  
  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  free_array(&dft1_trans1);
  
  
  //
  // dft2
  //
  
  alloc_array(&dft2);

  p_row = plan_many_1d_dft(&dft1_trans2, &dft2);
  gettimeofday(&t0, NULL);
  
  fftw_execute(p_row);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  fftw_destroy_plan(p_row);
  free_array(&dft1_trans2);
  
  
  //
  // dft2_trans1
  //
  
  alloc_array(&dft2_trans1);

  /* Transpose (N1, N2, N0) -> (N2, N1, N0) */
  
  p_transpose = plan_many_transpose(&dft2, &dft2_trans1, N0);

  gettimeofday(&t0, NULL);

  fftw_execute(p_transpose);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;

  fftw_destroy_plan(p_transpose);
  free_array(&dft2);
  
  
  //
  // dtf2_trans2
  //
  
  alloc_array(&dft2_trans2);
  
  /* Transpose (N2, N1, N0) -> (N2, N0, N1) */
  
  gettimeofday(&t0, NULL);
  
  n0_transpose(&dft2_trans1, &dft2_trans2);
  
  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  free_array(&dft2_trans1);
  
  
  //
  // dft3
  //

  alloc_array(&dft3);
    
  p_row = plan_many_1d_dft(&dft2_trans2, &dft3);
  
  gettimeofday(&t0, NULL);
  
  fftw_execute(p_row);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  fftw_destroy_plan(p_row);
  free_array(&dft2_trans2);
  
  
  //
  // dft3_trans1
  //
  
  alloc_array(&dft3_trans1);

  /* Transpose (N2, N0, N1) -> (N0, N2, N1) */
  
  p_transpose = plan_many_transpose(&dft3, &dft3_trans1, N1);

  gettimeofday(&t0, NULL);

  fftw_execute(p_transpose);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;

  fftw_destroy_plan(p_transpose);
  free_array(&dft3);
  
  
  //
  // dtf3_trans2
  //
  
  alloc_array(&dft3_trans2);
  
  /* Transpose (N0, N2, N1) -> (N0, N1, N2) */
  
  gettimeofday(&t0, NULL);
  
  n0_transpose(&dft3_trans1, &dft3_trans2);
  
  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  free_array(&dft3_trans1);
  
  
  //
  // Finalize
  //
  if (rank == 0)
	fprintf(stdout, "%.6e", t);
	
  free_array(&dft3_trans2);

  fftw_cleanup();
  
  MPI_Finalize();

  return 0;
}
