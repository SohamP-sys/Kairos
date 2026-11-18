#!/usr/bin/env python3
"""Disk-conscious EPIC-Kitchens download/preprocess loop.
Requires epic_downloader.py from the official dataset tooling. Uses one source video
at a time, creates a low-resolution copy, then removes the raw download.
"""
import argparse,csv,shutil,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest",type=Path); ap.add_argument("work",type=Path)
    ap.add_argument("--downloader",default="epic_downloader.py"); ap.add_argument("--extract-script",type=Path)
    ap.add_argument("--min-free-gb",type=float,default=8.0); args=ap.parse_args()
    raw=args.work/"raw"; low=args.work/"lowres"; done=args.work/"done"; logs=args.work/"logs"
    for p in (raw,low,done,logs):p.mkdir(parents=True,exist_ok=True)
    with args.manifest.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r.get("dataset","").lower() in {"epic_kitchens_100","epic","epic-kitchens-100"}]
    for row in rows:
        vid=row.get("video_id","").strip();
        if not vid or (done/(vid+".done")).exists():continue
        try:
            import shutil as sh
            free=sh.disk_usage(args.work).free/(1024**3)
            if free<args.min_free_gb: print(f"STOP: free space {free:.1f} GB below safety threshold"); return
            subprocess.run([sys.executable,args.downloader,"--videos","--specific-videos",vid,"--output-path",str(raw)],check=True)
            candidates=list(raw.rglob(vid+".mp4"));
            if not candidates: raise FileNotFoundError(f"Downloader did not produce {vid}.mp4")
            source=candidates[0]; dest=low/(vid+".mp4")
            subprocess.run(["ffmpeg","-nostdin","-y","-v","error","-i",str(source),"-vf","scale=-2:256,fps=15","-an","-c:v","libx264","-crf","26","-preset","veryfast",str(dest)],check=True)
            if args.extract_script:
                (args.work/"cache").mkdir(parents=True,exist_ok=True)
                subprocess.run([sys.executable,str(args.extract_script),str(dest),str(args.work/"cache"/(vid+".npz"))],check=True)
            (done/(vid+".done")).write_text("ok\n"); source.unlink(missing_ok=True)
            print(f"OK {vid}")
        except Exception as e:
            (logs/(vid+".log")).write_text(str(e)+"\n"); print(f"FAILED {vid}: {e}",file=sys.stderr)
if __name__=="__main__":main()
