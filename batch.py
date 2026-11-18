#!/usr/bin/env python3
"""Batch inference for videos, optionally creating manifest-duration fallbacks."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
from assemble import assemble_prediction
from baseline import evenly_spaced_windows
from schema import validate_output

def main():
    ap=argparse.ArgumentParser(description="Batch video predictions from folder/manifest")
    ap.add_argument("video_dir",type=Path); ap.add_argument("output_dir",type=Path)
    ap.add_argument("--manifest",type=Path); ap.add_argument("--fill-missing",action="store_true")
    ap.add_argument("--team-id",default=os.environ.get("TEAM_ID_STR","TEAM_ID")); ap.add_argument("--classes",type=Path)
    args=ap.parse_args(); rows=[]
    if args.manifest:
        with args.manifest.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    if rows:
        entries=[]
        for row in rows:
            vid=(row.get("video_id") or row.get("id") or "").strip()
            try: dur=float(row.get("duration_sec") or row.get("duration") or 0)
            except (TypeError,ValueError):dur=0.0
            dataset=(row.get("dataset") or "").lower()
            entries.append((vid,dur,dataset))
    else:
        entries=[(p.stem,0.0,"") for p in sorted(args.video_dir.glob("*.mp4"))]
    args.output_dir.mkdir(parents=True,exist_ok=True); failed=[]; processed=0
    for vid,dur,dataset in entries:
        if not vid:continue
        src=args.video_dir/(vid+".mp4"); out=args.output_dir/(vid+".json")
        if src.exists():
            result=subprocess.run([str(ROOT/"run.sh"),str(src),str(out)],check=False)
            if result.returncode:failed.append(vid)
        elif args.fill_missing and dur>0:
            label="cooking" if dataset in {"epic_kitchens_100","epic","epic-kitchens-100"} else "other"
            pred=assemble_prediction(args.team_id,vid,dur,label,0.3,[],evenly_spaced_windows(dur),classes_path=args.classes)
            ok,errs,_=validate_output(pred,expected_video_id=vid,manifest_duration=dur,classes_path=args.classes)
            if not ok:failed.append(vid); print(f"invalid fallback {vid}: {errs}");continue
            out.write_text(json.dumps(pred,indent=2)+"\n",encoding="utf-8")
            print(f"FALLBACK {vid}: no video file; emitted metadata-duration placeholder (not a prediction)")
        else:
            print(f"skip missing video: {src}");continue
        if out.exists():processed+=1
    print(f"Batch complete. manifest/inputs={len(entries)} written={processed} failures={len(failed)}")
    if failed:print("Failures:",", ".join(failed))
    return 1 if failed else 0
if __name__=="__main__":raise SystemExit(main())
