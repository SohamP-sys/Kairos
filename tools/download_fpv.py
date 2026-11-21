#!/usr/bin/env python3
"""Download FPVSum videos from manifest IDs using yt-dlp."""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def youtube_video_id(manifest_id: str) -> str:
    _, separator, suffix = manifest_id.partition("_")
    if not separator:
        raise ValueError(f"FPVSum ID has no category prefix: {manifest_id!r}")

    video_id = suffix[:11]
    segment_suffix = suffix[11:]
    if not YOUTUBE_ID.fullmatch(video_id) or (
        segment_suffix and not re.fullmatch(r"_\d+", segment_suffix)
    ):
        raise ValueError(f"Unsupported FPVSum video ID: {manifest_id!r}")
    return video_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    rows = [row for row in rows if row.get("dataset", "").lower() == "fpvsum"]
    if args.limit:
        rows = rows[:args.limit]

    for row in rows:
        manifest_id = row["video_id"].strip()
        youtube_id = youtube_video_id(manifest_id)
        target = args.output / f"{manifest_id}.mp4"
        if target.exists():
            continue
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--continue",
                "-f",
                "bestvideo[height<=480]+bestaudio/best[height<=480]",
                "--merge-output-format",
                "mp4",
                "-o",
                str(target),
                url,
            ],
            check=True,
        )
        if not target.is_file():
            raise RuntimeError(f"yt-dlp did not create {target}")
        print(f"Downloaded {manifest_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
