#!/bin/bash
# GPU job on UCF ARCC Newton: runs whatever command follows the script name.
#
#   sbatch --job-name=bert_pilot --time=00:20:00 slurm/newton_gpu.sbatch \
#       python train_bert.py --subset 2000 --eval-subset 1000 --epochs 1
#   sbatch --job-name=bert_final --time=02:00:00 slurm/newton_gpu.sbatch \
#       python train_bert.py --name bert_final --eval-test
#
# Any #SBATCH value can be overridden on the sbatch command line, e.g. a specific GPU:
#   --gres=gpu:nvidia_h100_pcie:1   or   --gres=gpu:tesla_v100-pcie-32gb:1
# Run slurm/setup_newton.sh once on the login node first (Python environment). Compute nodes
# have internet: the first job downloads the dataset and checkpoints, later jobs reuse the cache.
#SBATCH --account=qlou
#SBATCH --partition=normal
#SBATCH --job-name=rct-gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env_newton.sh
mkdir -p logs

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| transformers', transformers.__version__, '| gpu ok:', torch.cuda.is_available())"
echo "+ $*"
"$@"
echo "done at $(date)"
