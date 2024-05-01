#!/bin/bash
#SBATCH --job-name="Milestone2"
#SBATCH --output="Milestone2.out"
#SBATCH --error="Milestone2.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m2 1000 > m2.out
# { time srun ./m2 100 > m2_0.out; } 2> time2_0.out
# { time srun ./m2 1000 > m2_1.out; } 2> time2_1.out
# { time srun ./m2 10000 > m2_2.out; } 2> time2_2.out

# Generate nsys profile information, output will be in profile.out
# srun nsys profile --stats=true ./m2 > profile.out
# srun nv-nsight-cu-cli -f -o analysis_file ./m2 > profile.out
ncu --set full -o analysis_file ./m2 > m2.out