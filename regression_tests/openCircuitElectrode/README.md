# Open circuit electrode (OCE) case

This case simulates the potential distribution in a
hypothetical situation where an electrode is immersed in 
an electrolyte.
The potential in the electrode should be higher than the electrolyte
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

Run in serial with inputs_x/y/z
`$./*.ex inputs_x`


Run in parallel with inputs_x/y/z
`$ mpirun -n 4 ./*.ex inputs_x`

### Post process

You can open the `plt00001` plot file in 
paraview/visit. The exactcompare.py script compares
the calculated solution with the analytic solution.
`yt` python package is used for extracting data from 
plot files and then plots are made using `matplotlib`.

`$python exactcompare.py "plt00001"` generates an image
file called `pot_x_OCE.png` for inputs_x case

The interface sharpness factor can be varied to see its 
impact on the accuracy of the solution. It is currently set 
at 500 which gives a very accurate solution. If it is reduced to 
100 the accuracy reduces.

