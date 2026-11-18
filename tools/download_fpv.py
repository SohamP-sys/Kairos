#!/usr/bin/env python3
"""Download FPVSum videos from manifest IDs (<category>_<youtube-id>), if permitted."""
import argparse,csv,subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest",type=Path); ap.add_argument("output",type=Path); ap.add_argument("--limit",type=int); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with args.manifest.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r.get("dataset","").lower()=="fpvsum"]
    if args.limit:rows=rows[:args.limit]
    for row in rows:
        vid=row["video_id"].strip(); yt=vid.split("_",1)[1]
        target=args.output/(vid+".mp4")
        if target.exists():continue
        url="https://www.youtube.com/watch?v="+yt
        subprocess.run(["yt-dlp","-f","bestvideo[height<=480]+bestaudio/best[height<=480]","--merge-output-format","mp4","-o",str(target),url],check=False)
        print(f"{vid}: {'downloaded' if target.exists() else 'failed'}")
if __name__=="__main__":main()
