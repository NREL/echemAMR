# 2D Cathode-electrolyte-Anode (CEA) with charging case

This case simulates the time dependent 
potential and species distribution in a
simplified battery with three 
domains of different conductivities. The interfaces in this 
case are slanted and tests the robustness of the immersed interface
method. Butler Volmer 
conditions are applied at each interface for both species 
and potential equations. A comparison with analytic solution can also be done with 
the time history of plot files.
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

Run in serial with inputs_x
`$./*.ex inputs_x`


Run in parallel with inputs_x
`$ mpirun -n 4 ./*.ex inputs_x`

### Post process

Plotfiles for 1000 steps are generated with an interval of 
10. Each step is for a time step of 1 sec.
You can open the `plt00001` plot file in 
paraview/visit. 

To verify the solution, use the verifycase script:
`yt` python package is used for extracting data from 
plot files and then plots are made using `gnuplot`.
`$. ./verifycase.sh` generates image
files called `zerodsoln_cellvolt_2d.png` and 
`zerodsoln_conc_2d.png` showing cell voltage and species 
concentration variation with time. 
