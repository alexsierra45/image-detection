#!/bin/bash -l
# Comprueba desde un nodo de cómputo que el .venv ve la GPU y que OpenCV lee los tif igual que rasterio (CHG-001).
# Uso: sbatch slurm/gpu_check.sbatch
#SBATCH --job-name=gpu_check
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/y404/y404034/image-detection
#SBATCH --output=experiments/logs/%x-%j.out

module purge
module load apps/2021 Python/3.13.5-GCCcore-14.3.0
source .venv/bin/activate
source slurm/env.sh

echo "host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi
srun python tools/gpu_check.py
