# FluidFFT MPI 2D with FFTW1D
# Return execution time of one MPI process in .6e format
# Arguments: N0 N1

import sys
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
from fluidfft import import_fft_class

N = 512
M = 512
if len(sys.argv)==3:
    N = int(sys.argv[1])
    M = int(sys.argv[2])

cls = import_fft_class('fft2d.mpi_with_fftw1d')
o = cls(N, M)

a = np.ones(o.get_shapeX_loc())
a_fft = np.empty(o.get_shapeK_loc(), dtype=np.complex128)

t0 = time.time()
o.fft_as_arg(a, a_fft)
t1 = time.time()

if rank==0:
    print("{:.6e}".format(t1-t0))