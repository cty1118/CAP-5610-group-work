#!/bin/bash
# One-time setup on the Newton login node:
#   bash slurm/setup_newton.sh
# Creates the venv and installs CUDA PyTorch + Transformers. No Qwen/Alibaba models (ARCC policy).
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env_newton.sh

if [ ! -f "$VENV/bin/activate" ]; then
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
fi
python -m pip install -q --upgrade pip
python -m pip show torch >/dev/null 2>&1 || \
    python -m pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
python -m pip install -q "transformers>=4.44" datasets scikit-learn scipy numpy matplotlib

# The login node caps memory (4 GB virtual, few threads): too little to import CUDA PyTorch or
# even download with the Hugging Face tools. Compute nodes do have internet, so the first
# job downloads the dataset and checkpoints into $HF_HOME / data/hf_cache and later jobs reuse them.
echo "venv ready: $VENV"
