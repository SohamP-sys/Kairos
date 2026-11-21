#!/usr/bin/env python3
"""Download local Hugging Face model snapshots needed for offline inference."""
import os
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=Path(os.environ.get("KAIROS_WEIGHTS_DIR",ROOT/"weights")).expanduser()
MODELS={
 "siglip": "google/siglip-base-patch16-224",
 "videomae_kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--models",nargs="*",choices=list(MODELS),default=list(MODELS)); args=p.parse_args()
 try: from huggingface_hub import snapshot_download
 except ImportError: raise SystemExit("Install first: python -m pip install huggingface_hub")
 for name in args.models:
  target=WEIGHTS/name; target.mkdir(parents=True,exist_ok=True)
  print(f"Downloading {MODELS[name]} -> {target}")
  snapshot_download(repo_id=MODELS[name],local_dir=str(target))
 print("Downloads complete. run.sh remains offline-only.")
if __name__=="__main__":main()
