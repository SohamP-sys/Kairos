#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -ne 2 ]]; then echo "Usage: ./run.sh <video.mp4> <out.json>" >&2; exit 2; fi
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
exec "$PYTHON_BIN" "$ROOT/code/run_video.py" "$1" "$2"
