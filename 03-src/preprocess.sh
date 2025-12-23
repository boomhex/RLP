#!/bin/bash
#SBATCH --partition regular
#SBATCH --cpus-per-task 24
#SBATCH --mem 3G
#SBATCH --time 0-0:15:00
#SBATCH --nodes 1

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/rlp_venv/bin/activate
python3 ./01-preprocess.py
