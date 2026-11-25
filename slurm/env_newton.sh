# Sourced by the Newton job scripts and by setup_newton.sh: Python module, venv, caches.
module load python/python-3.11.4-gcc-12.2.0
VENV="${RCT_VENV:-$HOME/cap5610/venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/cap5610/hf_home}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
