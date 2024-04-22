#!/bin/bash
TOPDIR=${PWD}

#number of processors to use
NPROCS=${1:-16}

#need mpirun command (on eagle it is srun and 
#peregrine it is mpirun)
MPI_RUN_COMMAND=${2:-srun}

#script that is used to set environment variables for
#post processing or doing conda related commands
SCRIPTFILE=${3:-~/ytenv.sh}

RESULTS_DIR=${4:-~/echemamr_results_cpu}

MAKEFILE_OPTIONS=${5:-}
rm -rf ${RESULTS_DIR}

declare -a allcases=('DriftDiffuse1d' 'CEA' 'openCircuitElectrode' 'anodeElectrolyte' 'CEA_with_kdterm' 'CEA_2d' 'OCE_2d' 'CEAcharging' 'CEAcharging_2d')

#clean directories
for case in "${allcases[@]}";
do
	cd ${case}
        make realclean
        rm -rf finalpl* *.png
        . ../clean.sh
        cd ${TOPDIR}
done

#run cases
for case in "${allcases[@]}";
do
        echo ${case}
	cd ${case}
        make -j ${MAKEFILE_OPTIONS} 
        . ./runcase.sh "${MPI_RUN_COMMAND}" ${NPROCS}
        cd ${TOPDIR}
done

source ${SCRIPTFILE}
mkdir ${RESULTS_DIR}

#post process
for case in "${allcases[@]}";
do
	cd ${case}
        . ./verifycase.sh
        mv *.png ${RESULTS_DIR}
        cd ${TOPDIR}
done

cd ${RESULTS_DIR}
convert *.png regtest_results_$(date '+%d_%b_%Y_%H').pdf
