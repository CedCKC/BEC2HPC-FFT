# Plot execution time and speedup of methods and save to results-*D.png

import matplotlib.pyplot as plt

def plot_everything(X,T,S,L,title):
    #Return fig of execution time and speedup plot
    #X: x-axis values, nb of processes
    #T: list of list of execution times for each nb of procs, for each method
    #S: list of list of speedup for each nb of procs, for each method
    #L: list of methods
    #title: string, title of plot
    fig, ax = plt.subplots(1, 2, dpi=150, figsize=(9,4))
    fig.suptitle(title)
    
    ax[0].grid()
    ax[0].grid(which='minor', linewidth=0.25)
    ax[0].set_xscale('log', base=2)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xticks(X)
    ax[0].set_xlabel('number of processes')
    ax[0].set_ylabel('execution time (seconds)')
    for i in range(len(L)):
        ax[0].plot(X, T[i], '-x', linestyle='--')
    
    ax[1].loglog(base=2)
    ax[1].grid()
    ax[1].set_xlabel('number of processes')
    ax[1].set_ylabel('speedup')
    ax[1].plot([X[1],X[-1]], [X[1],X[-1]], color='k')
    for i in range(len(L)):
        ax[1].plot(X[1:], S[i][1:], '-x', linestyle='--', label=L[i])
        
    ax[1].legend(bbox_to_anchor=(1,1), loc="upper left")
    fig.tight_layout()
    return fig
    
def speedup(T):
    #Return speedup for list of list of execution times T 
    return [T[0]/t for t in T]


#Nb of processes tested for all methods
procs = [1, 2, 4, 8, 16]


#1D (33554432)
L_1D = ['FFTW MPI',
        'FFTW threads']

T_1D = [[2.62e+00, 1.58e+00, 8.16e-01, 4.94e-01, 3.28e-01],
        [2.64e+00, 1.73e+00, 8.75e-01, 4.49e-01, 3.07e-01]]

S_1D = [speedup(T) for T in T_1D]


#2D (16384, 16384)
L_2D = ['FFTW MPI (with FFTW1D)',
        'FFTW MPI (with FFTW2D)',
        'FFTW threads (stride)',
        'FFTW threads (transpose)',
        'FluidFFT MPI (with FFTW1D)',
        'FluidFFT MPI (with FFTW2D)']

T_2D = [[1.73e+01, 1.06e+01, 5.49e+00, 3.07e+00, 2.05e+00],
        [2.54e+01, 1.46e+01, 7.57e+00, 4.18e+00, 2.52e+00],
        [3.47e+01, 2.22e+01, 1.13e+01, 6.57e+00, 5.23e+00],
        [1.87e+01, 1.02e+01, 5.57e+00, 3.54e+00, 3.10e+00],
        [5.56e+00, 3.44e+00, 2.15e+00, 9.47e-01, 4.67e-01],
        [4.27e+00, 2.47e+00, 1.42e+00, 7.94e-01, 6.01e-01]]

S_2D = [speedup(T) for T in T_2D]


#3D (512, 512, 512)
L_3D = ['FFTW MPI (with FFTW3D)',
        'FFTW threads (with FFTW3D)',
        'FluidFFT MPI (with FFTW1D)',
        'FluidFFT MPI (with FFTW3D)']

T_3D = [[1.59e+01, 8.86e+00, 4.28e+00, 2.36e+00, 1.39e+00],
        [1.39e+01, 7.34e+00, 4.52e+00, 3.01e+00, 2.24e+00],
        [3.99e+00, 2.13e+00, 1.50e+00, 6.90e-01, 4.38e-01],
        [1.94e+00, 1.26e+00, 6.43e-01, 3.86e-01, 2.89e-01]]

S_3D = [speedup(T) for T in T_3D]


fig_1D = plot_everything(procs, T_1D, S_1D, L_1D, '1D, N = 33 554 432')
fig_2D = plot_everything(procs, T_2D, S_2D, L_2D, '2D, (N0, N1) = (16 384, 16 384)')
fig_3D = plot_everything(procs, T_3D, S_3D, L_3D, '3D, (N0, N1, N2) = (512, 512, 512)')

fig_1D.savefig('results-1D.png')
fig_2D.savefig('results-2D.png')
fig_3D.savefig('results-3D.png')
