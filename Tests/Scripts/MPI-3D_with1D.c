// FFTW MPI 2D with FFTW1D
// Return execution time of one MPI process in .6e format

/* 
   mpicc -O3 example2D-MPI.c -o example2D-MPI.ex -I/home/jeremie/opt/stow/fftw/3.3.5/include -L/home/jeremie/opt/stow/fftw/3.3.5/lib -lfftw3_mpi -lfftw3 -lm
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include <fftw3-mpi.h>

#define IDK(i,j) ((i)*M + (j))
#define IDKTRANS(i,j) ((i)*N + (j))
#define IDK_(DIM, i,j) ((i)*(DIM) + (j))

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

#ifdef DEBUG
#define debug_print_array print_array
#else
void debug_print_array(struct array* a) { }
#endif

void print_array(struct array* a) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
  
  printf("rank=%d/%d - array=%s\t n0=%ld n1=%ld local_n0=%ld, local_n0_start=%ld, alloc_local=%ld\n",
	 rank, size, a->name, a->n0, a->n1, a->local_n0, a->local_n0_start, a->alloc_local);

  if (a->data != NULL) {
    char filename[200];
    sprintf(filename, "out/%s-%d.txt", a->name, rank);
    printf("%s\n", filename);
    
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
      printf("Error opening file %s!\n", filename);
      exit(1);
    }
  
    // print per col
    int i, j;
    for(j = 0; j < a->n1; j++) {
      for(i = 0; i < a->local_n0; i++) {
	if (abs(a->data[IDK_(a->n1, i,j)][0]) > 0.01 || abs(a->data[IDK_(a->n1, i,j)][1]) > 0.01) {
	  fprintf(f, "col=%d - freq: %3ld %+9.5f %+9.5f I\n", j, i+a->local_n0_start, a->data[IDK_(a->n1, i,j)][0], a->data[IDK_(a->n1, i,j)][1]);
	}
      }
    }

    fclose(f);
	
  }
}

void init_array(struct array* a, char* name, ptrdiff_t n0, ptrdiff_t n1) {
  a->name = name;
  a->n0 = n0;
  a->n1 = n1;
  a->block = FFTW_MPI_DEFAULT_BLOCK;
  a->data = NULL;
}

void init_array_transpose(struct array* in, char* in_name, struct array *out, char* out_name, ptrdiff_t n0, ptrdiff_t n1) {
  init_array(in,  in_name,  n0, n1);
  init_array(out, out_name, n1, n0);
  
  in->alloc_local = out->alloc_local = fftw_mpi_local_size_2d_transposed(n0, n1, MPI_COMM_WORLD,
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
  assert(in->local_n0 == out->local_n0);

#ifdef DEBUG
  printf("n0 %ld n1 %ld block %d block %d\n", in->n0, in->n1, in->n2, in->block, out->block);
#endif

  /* Transform each row of a local 2d array with n0_local rows and n1 columns */
  
  int rank = 1;        /* we are computing 1d transforms */
  int n[] = {in->n1*in->n2};  /* 1d transforms of length in.n1 */
  int howmany = in->local_n0;
  int idist = in->n1*in->n2, odist = in->n1*in->n2; /* the distance in memory between the first element
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

fftw_plan plan_many_transpose(struct array* in, struct array* out) {
  int howmany = 2; // transpose complex number (ie. double*2)

  // use advanced interface because we are transposing complex vectors (howmany=2)
  fftw_plan p_transpose = fftw_mpi_plan_many_transpose(in->n0, in->n1, howmany, in->block, out->block, (double *)in->data, (double *)out->data, MPI_COMM_WORLD, FFTW_ESTIMATE);
  assert(p_transpose != NULL);

  return p_transpose;
}


int main(int argc, char **argv)
{
  ptrdiff_t N0 = 16, N1 = 16, N2 = 16;

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

  // FFTW uses a 1d block distribution of the data, distributed along the first dimension. 
  
  // in -> dft1 -> dft1_trans -> dft2 -> dft2_trans
  struct array dft1, dft1_trans;
  init_array_transpose(&dft1, "dft1", &dft1_trans, "dft1_trans", N, M);

  struct array in; // => same distribution as dft1
  init_array(&in, "in", N, M);
  in.alloc_local = dft1.alloc_local; // not mandatory
  in.local_n0 = dft1.local_n0;
  in.local_n0_start = dft1.local_n0_start;
  
  struct array dft2, dft2_trans;
  init_array_transpose(&dft2, "dft2", &dft2_trans, "dft2_trans", M, N);
    
  debug_print_array(&in);
  debug_print_array(&dft1);
  debug_print_array(&dft1_trans);
  debug_print_array(&dft2);
  debug_print_array(&dft2_trans);

  alloc_array(&in);

  /* Initialize 'in' */
  int i, j;
  for (i = 0; i < in.local_n0; i++) {
    for (j = 0; j < M; j++) {
      int i_global = in.local_n0_start + i;
      in.data[IDK(i,j)][0] = cos((3+j) * 2*M_PI*i_global/N);
      in.data[IDK(i,j)][1] = 0;
    }
  }

  //
  // dft1
  //

  alloc_array(&dft1);


  double t = 0, t_ = 0;
  struct timeval t0, t1; // timer
    
  fftw_plan p_row = plan_many_1d_dft(&in, &dft1);
  gettimeofday(&t0, NULL);
  
  fftw_execute(p_row);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  fftw_destroy_plan(p_row);
  
  debug_print_array(&dft1);

  //
  // dtf1_trans
  //

  alloc_array(&dft1_trans);
  
  /* Transpose */

  fftw_plan p_transpose = plan_many_transpose(&dft1, &dft1_trans);
  
  gettimeofday(&t0, NULL);

  fftw_execute(p_transpose);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;

  fftw_destroy_plan(p_transpose);

  debug_print_array(&dft1_trans); 

  if (size == 1) {
    int i, j;
    for(j = 0; j < dft1.n1; j++) {
      for(i = 0; i < dft1.n0; i++) {
	assert(dft1.data[IDK(i,j)][0] == dft1_trans.data[IDKTRANS(j,i)][0]);
	assert(dft1.data[IDK(i,j)][1] == dft1_trans.data[IDKTRANS(j,i)][1]);
      }
    }
  }
  
  //
  // dft2
  //

  alloc_array(&dft2);

    
  p_row = plan_many_1d_dft(&dft1_trans, &dft2);
  gettimeofday(&t0, NULL);
  
  fftw_execute(p_row);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;
  
  fftw_destroy_plan(p_row);
  
  debug_print_array(&dft2);
  
  //
  // transpose dft2
  //

  /* Transpose */

  alloc_array(&dft2_trans);
  p_transpose = plan_many_transpose(&dft2, &dft2_trans);

  gettimeofday(&t0, NULL);

  fftw_execute(p_transpose);

  gettimeofday(&t1, NULL);
  t_ = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000;
  t = t + t_;

  fftw_destroy_plan(p_transpose);

  debug_print_array(&dft2_trans); 

  if (size == 1) {
    int i, j;
    for(j = 0; j < dft2.n1; j++) {
      for(i = 0; i < dft2.n0; i++) {
	assert(dft2.data[IDKTRANS(i,j)][0] == dft2_trans.data[IDK(j,i)][0]);
	assert(dft2.data[IDKTRANS(i,j)][1] == dft2_trans.data[IDK(j,i)][1]);
      }
    }
  }
  
  //
  // Finalize
  //
  if (rank == 0)
	fprintf(stdout, "%.6e", t);

  free_array(&dft2_trans);
  free_array(&dft2);
  free_array(&dft1_trans);
  free_array(&dft1);
  free_array(&in);

  fftw_cleanup();
  
  MPI_Finalize();

  return 0;
}
