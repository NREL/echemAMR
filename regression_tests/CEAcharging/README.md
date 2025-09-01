# Cathode-electrolyte-Anode (CEA) with charging case

This case simulates the time dependent 
potential and species distribution in a
simplified battery with three 
domains of different conductivities. Butler Volmer 
conditions are applied at each interface for both species 
and potential equations. A comparison with analytic solution assuming uniform distribution of 
species and potential can be done with 
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

Plotfiles for 400 steps are generated with an interval of 
10. Each step is for a time step of 1 sec.
You can open the `plt00001` plot file in 
paraview/visit. 

To verify the solution, use the verifycase script:
`yt` python package is used for extracting data from
plot files and then plots are made using `gnuplot` here.
`$. ./verifycase.sh` generates image
files called `zerodsoln_cellvolt_1d.png` and 
`zerodsoln_conc_1d.png` showing cell voltage and species 
concentration variation with time. 

<img width="300" alt="zerodsoln_cellvolt_1d" src="https://github.com/user-attachments/assets/ac417081-b98f-430b-a273-2517c5aaa011" />
<img width="300" alt="zerodsoln_conc_1d" src="https://github.com/user-attachments/assets/3ee56079-f305-4a74-b064-cf4a0a031a32" />


