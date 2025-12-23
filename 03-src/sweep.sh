#!/bin/bash
#SBATCH --partition gpu
#SBATCH --cpus-per-task 4
#SBATCH --mem 8G
#SBATCH --time 0-8:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:2

module purge
source $HOME/venvs/rlp_venv/bin/activate

python3 ./02-network-sweep.py