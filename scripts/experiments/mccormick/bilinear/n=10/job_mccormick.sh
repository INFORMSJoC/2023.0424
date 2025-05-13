#!/bin/bash
#SBATCH --partition=scaling
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=qcqp_v10_b45_s100_%a_mccormick.txt
#SBATCH --array=1-1000%8
#SBATCH --qos=normal --time=0-10:00:00
#SBATCH --exclusive
julia run_mccormick.jl "$SLURM_ARRAY_TASK_ID"
