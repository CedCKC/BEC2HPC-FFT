// FFTW threads 1D
// Return execution time of one MPI process in .6e format
// Arguments: nb_threads N

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <complex.h>
#include <fftw3.h>

int main(int argc, char **argv)
{
int i;

int N = 4096;
int nb_threads = 1;
if (argc > 1){
	nb_threads = atol(argv[1]);
	if (argc == 3){
		N = atol(argv[2]);
	}
}

fftw_init_threads();
fftw_plan_with_nthreads(nb_threads);

fftw_complex *in, *out;
fftw_plan p;

in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

for (i=0; i<N; i++)
{
	in[i] = cos(3 * 2*M_PI*i/N);
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