#!/usr/bin/env python3
"""Download and verify the AVION ViT-B EK100 fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("KAIROS_WEIGHTS_DIR", ROOT / "weights")).expanduser()
CHECKPOINT_URL = "https://drive.google.com/uc?id=1YgxYjmpSxI26wdnKeeT0YakEIAFP5HjF"
EXPECTED_MD5_PREFIX = "6c3c5e"


def md5_prefix(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:len(EXPECTED_MD5_PREFIX)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WEIGHTS / "avion" / "avion_finetune_cls_lavila_vitb_best.pt",
    )
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.is_file() and md5_prefix(args.output) == EXPECTED_MD5_PREFIX:
        print(f"Verified AVION checkpoint: {args.output}")
        return 0

    try:
        import gdown
    except ImportError:
        raise SystemExit("Install gdown first: pip install gdown")

    gdown.download(CHECKPOINT_URL, str(args.output), quiet=False)
    if not args.output.is_file():
        raise SystemExit("Checkpoint download failed")
    if md5_prefix(args.output) != EXPECTED_MD5_PREFIX:
        args.output.unlink(missing_ok=True)
        raise SystemExit("Checkpoint MD5 verification failed")

    print(f"Verified AVION checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
