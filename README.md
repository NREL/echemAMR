# echemAMR
## A 3D microstructure resolving electrochemical transport and interfacial chemistry solver
<img src="https://github.com/user-attachments/assets/54859857-0559-45e5-bb36-f8dfe798d549" width="500" />

EchemAMR is a coupled transport-reaction solver that simulates 
electrochemical transport and interfacial chemistry at the microstructure scale. EchenAMR 
can simulate lithium ion battery microstructures and has been applied to other electrochemical applications 
such as electrochemical CO2 reduciton. An immersed interface formulation in echemAMR 
enables rapid representation of complex electrode geometries. Buttler-Volmer flux conditions are 
imposed at electrode-electrolyte interfaces using an iterative non-linear solve within the immersed interface method. 
The solver is developed on top of open-source performance portable library, AMReX, providing mesh adaptivity and 
parallel execution capabilities on current and upcoming high-performance-computing (HPC) architectures. 

# Build instructions
* echemAMR uses submodules, please clone with git clone --recursive https://github.com/NREL/mesoflow.git (or update your existing clone with git submodule update --init --recursive)
* gcc and an MPI library (openMPI/MPICH) for CPU builds. cuda > 11.0 is also required for GPU builds
* This tool depends on the AMReX library - which is included as a submodule
* This tool also depends on the HYPRE library for some of the stiff electrochemical simulations, which can be obtained 
built following these instructions - https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#external-solvers
* go to any of the test cases in tests or model folder (e.g. cd regression_tests/CEAcharging)
* build executable using the GNUMakefile - do $make for CPU build or do $make USE_CUDA=TRUE for GPU build

# Run instructions

* By default MPI is enabled in all builds, you can turn it off by doing $make USE_MPI=FALSE
* For parallel execution do $mpirun -n nprocs echemAMR3d.gnu.MPI.ex inputs
* For serial builds do $./mesoflow3d.gnu.ex inputs
* For GPU execution make sure the number of ranks match the number of GPUs on the machine. 
  For example, if you have 2 GPUs on a node, do $mpirun -n 2 mesoflow3d.gnu.MPI.CUDA.ex inputs
  
  
# Visualization instructions
  
* The outputs for a case are in the form of AMReX plotfiles
* These plot files can be open usine AMReX grid reader in ParaView (see https://amrex-codes.github.io/amrex/docs_html/Visualization.html#paraview)
* Alternatively yt or visit can also be used. see https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html
