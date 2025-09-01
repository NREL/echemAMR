# Drift and diffusion physics case

This case solves a coupled convection-diffusion and Laplace problem 
with two species, for which an analytic solution can be obtained, at steady state:

$$\frac{ds_1}{dt}+\frac{d (v s_1)}{dx}=D_1\frac{d^2 s_1}{d x^2} \quad s_1(0)=0 \quad s_1(1)=1$$
$$\frac{ds_2}{dt}=\frac{d^2 s_2}{d x^2}+1.0 \quad s_2(0)=s_2(1)=0$$
$$\frac{d2\phi}{dx^2}=0 \quad \phi(0)=1 \quad \phi(1)=0$$
$$D_1=0.1 \quad v=E=-\frac{d\phi}{dx}$$

The exact solution at steady state is given by:

$$\phi(x)=1-x$$
$$v=E=1$$
$$s_1(x)=\frac{exp(vx/D_1)-1}{exp(v/D_1)-1}$$
$$s_2(x)=(x-x^2)/2$$

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

You can open the final plot file after 7000 steps (`plt07000`) in 
paraview/visit. The exactcompare.py script compares
the calculated solution with the analytic solution.
`yt` python package is used for extracting data from 
plot files and then plots are made using `matplotlib`.

`$python exactcompare.py "plt07000" 0` generates an image
file called `species_x.png` for inputs_x case. The second argument 
of `0` in the exactcompare script is for the axial direction (0 for x,1 for y,2 for z).

<img width="600" alt="species_x" src="https://github.com/user-attachments/assets/53b2b61c-2106-4cb2-96dd-427e64f00368" />



