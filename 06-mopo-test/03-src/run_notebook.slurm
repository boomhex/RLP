#!/bin/bash
#SBATCH --job-name=mopo_nb
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=mopo_nb.out
#SBATCH --error=mopo_nb.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python/3.10.8
module load CUDA/12.1   # if this module name exists on Hábrok
source ~/venvs/habrok/bin/activate

# quick sanity check inside the job
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version (torch):", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

jupyter nbconvert --to notebook --execute mopo.ipynb \
  --output mopo_executed.ipynb \
  --ExecutePreprocessor.allow_errors=True
