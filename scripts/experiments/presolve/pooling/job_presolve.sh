#!/bin/bash
#SBATCH --partition=scaling
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=random_schweiger_c15_e150_q1_%a_presolve.txt
#SBATCH --array=1-1000%2
#SBATCH --qos=normal --time=0-10:00:00
#SBATCH --exclusive
julia run_presolve.jl "$SLURM_ARRAY_TASK_ID"
