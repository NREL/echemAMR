# Cathode-electrolyte-Anode (CEA) case with diffusional conductivity

This case simulates the potential distribution in a
simplified battery with three 
domains of different conductivities. Butler Volmer 
conditions are applied at each interface along with 
higher resolutions using adaptive meshing. The difference 
between this case and the `CEA` case is the presence of 
a diffusional conductivity (`kd`) term.
A comparison with a 1d finite difference solution is also done with 
the final plot file to verify correctness.

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
file called `pot_x_CEAkd.png` for inputs_x case.

