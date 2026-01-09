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

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python/3.10.8
module load CUDA/12.1
source ~/venvs/habrok/bin/activate

export PYTHONUNBUFFERED=1

# quick sanity check inside the job
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
    if torch.cuda.device_count() > 1:
        print("gpu1:", torch.cuda.get_device_name(1))
PY

# run your normal python script and capture stdout+stderr into a single file
python -u 02-network-sweep.py > output.txt 2>&1