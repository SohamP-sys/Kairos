#!/usr/bin/env python3
"""Validate generated prediction JSONs with the bundled schema mirror."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from schema import validate_output

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("directory",type=Path); ap.add_argument("--classes",type=Path); args=ap.parse_args()
 files=sorted(args.directory.glob("*.json")); bad=0; warnings=0
 for f in files:
  try:
   data=json.loads(f.read_text(encoding="utf-8")); ok,errs,warns=validate_output(data,expected_video_id=f.stem,classes_path=args.classes)
   if warns:warnings+=len(warns)
   if not ok:
    bad+=1; print(f"FAIL {f.name}: "+" | ".join(errs))
   elif warns: print(f"WARN {f.name}: "+" | ".join(warns))
  except Exception as e:bad+=1; print(f"FAIL {f.name}: {e}")
 print(f"Validated {len(files)} files; invalid={bad}; warnings={warnings}")
 return 1 if bad else 0
if __name__=="__main__":raise SystemExit(main())
