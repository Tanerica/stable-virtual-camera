#!/bin/sh
#SBATCH --job-name=VSD
#SBATCH --output=/lustre/scratch/client/movian/research/users/tannv34/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/tannv34/slurm_out/slurm_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-gpu=50G
#SBATCH --partition=movianr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=v.tannv34@vinai.io

# Set NCCL interface for multi-GPU communication
export NCCL_SOCKET_IFNAME=bond0

# Launch your container and run the script
srun \
  /bin/bash -c "
    conda activate seva
    bash train_vsd.sh
  "
