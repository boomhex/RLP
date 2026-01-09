#!/bin/bash
#SBATCH --partition gpu
#SBATCH --cpus-per-task 4
#SBATCH --mem 8G
#SBATCH --time 0-4:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100:1

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# source $HOME/venvs/rlp_venv/bin/activate

python3 ./02-network-sweep.py