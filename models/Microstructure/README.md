# Three dimensional microstructure case

This case simulates the time dependent potential and species 
distribution in 3D microstructures. These are 
production cases. The geometry of the microstructure is provided 
through pixelated files in the `Geometry` folder. There are
problem specific input files that can be used, e.g. `inputs_Onesphere`
for a case with anode and cathode as single spheres.
It should be noted that these are production cases and may take 
significantly long time to run and requires high performance computing resources.
This case
requires the use of HYPRE library to solve linear systems 
with higher stiffness.

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

You can open the various plotfiles in
paraview/visit. 
