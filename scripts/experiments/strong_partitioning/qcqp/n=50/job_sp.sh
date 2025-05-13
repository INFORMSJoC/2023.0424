#!/bin/bash
#SBATCH --partition=scaling
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=qcqp_v50_b250_q12_s100_%a.txt
#SBATCH --array=1-1000%8
#SBATCH --qos=normal --time=0-10:00:00
#SBATCH --exclusive
julia run_sp.jl "$SLURM_ARRAY_TASK_ID"
