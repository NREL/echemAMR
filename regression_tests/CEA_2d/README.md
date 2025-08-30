# Two dimensional Cathode-electrolyte-Anode (CEA) case

This case simulates the potential distribution in a
simplified battery with three 
domains of different conductivities in a 2D domain. The 
interface are set at an angle to include fraction cells. 
Butler Volmer conditions are applied at each interface. 
A comparison with analytic solution can also be done with 
the final plot file for a 1d case. 
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

Run in serial with inputs_lowcond/highcond
`$./*.ex inputs_lowcond`


Run in parallel with inputs_lowcond/highcond
`$ mpirun -n 4 ./*.ex inputs_lowcond`

### Post process

You can open the `plt00001` plot file in 
paraview/visit. The exactcompare.py script compares
the calculated solution with the analytic solution.
`yt` is used to extract data from the plot file and 
`gnuplot` is used for one dimensional plots.

`$. ./verify_lowcond.sh` generates an image
file called `cea_soln_2d.png` for inputs_lowcond case.
Similarly, use verify_highcond.sh for the higher conductivity case.

It can be seen that the 1D solution matches well with the 
higher conductivity case as opposed to the lower conductivity case.
This is mainly because the transverse gradients are more 
pronounced in the lower conductivity case.

