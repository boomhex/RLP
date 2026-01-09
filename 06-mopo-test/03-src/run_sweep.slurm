#!/bin/bash
#SBATCH --job-name=network_sweep
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=network_sweep.out
#SBATCH --error=network_sweep.err

module purge
source $HOME/venvs/rlp_venv/bin/activate

python3 ./02-network-sweep.py