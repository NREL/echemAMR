# Anode-electrolyte (CEA) case

This case simulates a half battery. Butler Volmer 
conditions are applied at the active material electrolyte interface. 
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
file called `pot_x_AE.png` for inputs_x case

<img width="300" alt="pot_x_AE" src="https://github.com/user-attachments/assets/6efd295b-33fd-4034-bf4e-66cd621e5c86" />



