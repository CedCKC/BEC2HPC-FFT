// FFTW threads 2D with FFTW2D
// Return execution time of one MPI process in .6e format
// Arguments: nb_threads N0 N1

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <complex.h>
#include <fftw3.h>

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

fftw_init_threads();
fftw_plan_with_nthreads(nb_threads);

fftw_complex *in, *out;
fftw_plan p;

in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
p = fftw_plan_dft_2d(N, M, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

for (i=0; i<N; i++) for (j=0; j<M; j++)
{
	in[i*M+j] = cos(3 * 2*M_PI*i/N);
}

struct timeval t0, t1; // timer
gettimeofday(&t0, NULL);

fftw_execute(p);

gettimeofday(&t1, NULL);

fftw_destroy_plan(p);
fftw_cleanup_threads();

fprintf(stdout, "%.6e", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / (double)1000000);

return 0;
}