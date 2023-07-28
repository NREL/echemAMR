#! /bin/bash
#SBATCH -A exabatt 	    # Project handle
#SBATCH --job-name=dbg # Name of job
#SBATCH --time=00:59:00  # --time=02-00:00:00 --time=04:00:00  # #SBATCH --mem=150000 
#SBATCH --nodes=2 # Says the job needs n nodes and should place X processes on each node
#SBATCH --ntasks-per-node=34
#SBATCH -p debug

##SBATCH --qos=high

cd $SLURM_SUBMIT_DIR	 # Set to the location the job was submitted from.
module purge
module load gcc/8.4.0
module load cuda/10.2.89
module load openmpi/4.0.4/gcc-8.4.0

# Note: if you edit the file in windows with Notepad++: select Edit/EOL conversion/Unix

echo '- Current folder is (where program will be run):'
pwd

# Execute the code
mpirun -n 68 ./echemAMR3d.gnu.DEBUG.MPI.ex inputs_Onesphere > Onesphere.diary

echo ">>> >>> >>> END OF ALL CALCULATIONS <<< <<< <<<"