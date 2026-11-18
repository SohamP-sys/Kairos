#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
$PYTHON_BIN -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "$ROOT/requirements.txt"
# Apple Silicon: PyTorch installs its MPS backend in the standard macOS wheel.
python -m pip install torch torchvision
python -m pip install -r "$ROOT/requirements-models.txt"
python -m pip install huggingface_hub
python "$ROOT/tools/environment_check.py"
echo "Environment ready. Next run: python tools/download_models.py"
