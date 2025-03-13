srun -n 1 ./gpu_ex inputs_GPU_tests amr.n_cell=32 16 16 > out32gpu
srun -n 1 ./gpu_ex inputs_GPU_tests amr.n_cell=64 32 32 > out64gpu
srun -n 1 ./gpu_ex inputs_GPU_tests amr.n_cell=128 64 64 > out128gpu
srun -n 1 ./gpu_ex inputs_GPU_tests amr.n_cell=192 96 96 > out192gpu
srun -n 1 ./gpu_ex inputs_GPU_tests amr.n_cell=256 128 128 > out256gpu

srun -n 1 ./cpu_ex inputs_GPU_tests amr.n_cell=32 16 16 > out32cpu
srun -n 1 ./cpu_ex inputs_GPU_tests amr.n_cell=64 32 32 > out64cpu
srun -n 1 ./cpu_ex inputs_GPU_tests amr.n_cell=128 64 64 > out128cpu
srun -n 1 ./cpu_ex inputs_GPU_tests amr.n_cell=192 96 96 > out192cpu
srun -n 1 ./cpu_ex inputs_GPU_tests amr.n_cell=256 128 128 > out256cpu
