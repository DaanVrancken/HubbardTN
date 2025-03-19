#!/bin/bash
#
#PBS -N PT_params
#PBS -m a
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4
#PBS -l mem=10 GB
#PBS -A starting_2024_121
#

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4

module purge
module load Julia/1.10.5-linux-x86_64
export JULIA_DEPOT_PATH="/dodrio/scratch/projects/starting_2024_121/.julia"

cd $STARTDIR
echo "PBS: $PBS_ID"

ls

julia -tauto /dodrio/scratch/projects/starting_2024_121/HubbardMPS/data/params/PT/extr_params.jl | tee /dodrio/scratch/projects/starting_2024_121/HubbardMPS/data/params/PT/out.dat
