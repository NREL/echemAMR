# Two dimensional Open-Circuit-Electrode (OCE) case

This 2D case simulates the potential distribution in a
hypothetical situation where an electrode is immersed in 
an electrolyte. The potential in the electrode should be higher than the electrolyte
so as to match up with the open circuit potential, resulting in 
zero current. Butler Volmer 
conditions are applied at each interface along with 
higher resolutions using adaptive meshing.
A comparison with analytic solution can also be done with 
the final plot file.

### Build instructions

To build a serial executable with gcc do
`$ make -j COMP=gnu USE_MPI=FALSE`

To build a serial executable with clang++ do
`$ make -j COMP=llvm USE_MPI=FALSE`

To build a parallel executable with gcc do
`$ make -j COMP=gnu USE_MPI=TRUE`

To build a parallel executable with gcc, mpi and cuda
`$ make -j COMP=gnu USE_CUDA=TRUE USE_MPI=TRUE`

### Run instructions

Run in serial with inputs
`$./*.ex inputs`


Run in parallel with inputs
`$ mpirun -n 4 ./*.ex inputs`

### Post process

You can open the `plt00001` plot file in 
paraview/visit. 

The exactcompare.py script compares
the calculated solution with the analytic solution.
`yt` is used to extract data from the plot file and 
`gnuplot` is used for one dimensional plots.

`$. ./verifycase.sh` generates an image
file called `oce_soln_2d.png` for the baseline case.

