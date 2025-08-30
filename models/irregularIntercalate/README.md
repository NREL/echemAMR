# Three dimensional half cell with adaptive meshing

This case simulates the time dependent potential and species 
distribution in a half cell during charging. The interface is 
also resolved using AMR.

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
