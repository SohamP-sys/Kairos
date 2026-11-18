from pathlib import Path
import argparse, json, os, sys
from pipeline import process_video

def main():
    p=argparse.ArgumentParser(description="Action-aware first-person video summarizer")
    p.add_argument("video"); p.add_argument("output")
    args=p.parse_args()
    team=os.environ.get("TEAM_ID_STR","TEAM_ID")
    try:process_video(Path(args.video),team,Path(args.output))
    except Exception as exc:
        # Fail-safe output attempt: preserve contract even for unreadable media.
        from assemble import assemble_prediction
        out=assemble_prediction(team,Path(args.video).stem,10.0)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
        print(f"[warning] {exc}; wrote safe fallback",file=sys.stderr)
if __name__=="__main__":main()
