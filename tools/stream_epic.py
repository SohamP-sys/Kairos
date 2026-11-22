#!/usr/bin/env python3
"""Compatibility wrapper for sequential EPIC inference and validated cleanup."""

import argparse

try:
    from .stream_manifest import main as stream_manifest_main
except ImportError:
    from stream_manifest import main as stream_manifest_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("work")
    parser.add_argument("--output-dir")
    parser.add_argument("--epic-downloader")
    parser.add_argument("--team-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--keep-videos", action="store_true")
    args = parser.parse_args()

    forwarded = [
        args.manifest,
        "--work-dir",
        args.work,
        "--dataset",
        "epic",
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    for option, value in (
        ("--output-dir", args.output_dir),
        ("--epic-downloader", args.epic_downloader),
        ("--team-id", args.team_id),
        ("--limit", str(args.limit) if args.limit is not None else None),
    ):
        if value is not None:
            forwarded.extend((option, value))
    if args.keep_videos:
        forwarded.append("--keep-videos")
    return stream_manifest_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
