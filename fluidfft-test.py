from fluiddyn.util.mpi import rank, nb_proc
print("Hello world! I'm rank {}/{}".format(rank, nb_proc))

import numpy as np
from fluidfft.fft2d import methods_mpi
from fluidfft import import_fft_class

if rank == 0:
    print(methods_mpi)
    
cls = import_fft_class('fft2d.mpi_with_fftw1d')
o = cls(48, 32)

_ = o.run_tests()
print(_)

times = o.run_benchs()
if rank == 0:
    print('t_fft = {} s; t_ifft = {} s'.format(*times))
    
print(o.get_is_transposed())

k0, k1 = o.get_k_adim_loc()
print('k0:', k0)
print('k1:', k1)

print(o.get_shapeX_loc())
print(o.get_shapeK_loc())

print(o.get_seq_indices_first_X())

print(o.get_seq_indices_first_K())

a = np.ones(o.get_shapeX_loc())
a_fft = o.fft(a)

a_fft = np.empty(o.get_shapeK_loc(), dtype=np.complex128)
o.fft_as_arg(a, a_fft)

o.ifft_as_arg(a_fft, a)

a = o.ifft(a_fft)