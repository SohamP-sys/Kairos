#!/usr/bin/env bash
# Fetch the upstream AVION source tree and its checkpoint script.
# This prepares the upstream environment only; inference integration is not yet validated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/AVION"
mkdir -p "$ROOT/third_party"
if [[ -d "$DEST/.git" ]]; then git -C "$DEST" pull --ff-only
else git clone --depth 1 https://github.com/zhaoyue-zephyrus/AVION.git "$DEST"
fi
if [[ -f "$DEST/scripts/download_checkpoints.sh" ]]; then
  echo "Upstream download script found. Inspect it before running: $DEST/scripts/download_checkpoints.sh"
  echo "The script may download large files; checkpoint authenticity must be checked by MD5."
else
  echo "AVION source cloned, but no checkpoint script at expected location. Inspect upstream docs/model zoo."
fi
