#!/usr/bin/env python3
"""Validate Track 4 prediction files against the output contract and manifest."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from schema import validate_output


def _read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        required = {"video_id", "duration_sec", "dataset"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"{path} must have columns: {', '.join(sorted(required))}")
        entries = {}
        for line, row in enumerate(rows, start=2):
            video_id = (row.get("video_id") or "").strip()
            try:
                duration = float(row["duration_sec"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{line} has an invalid duration_sec.")
            if not video_id or not math.isfinite(duration) or duration <= 0 or video_id in entries:
                raise ValueError(f"{path}:{line} has an invalid or duplicate entry.")
            dataset = (row.get("dataset") or "").strip().lower()
            if dataset not in {"epic", "epic-kitchens-100", "epic_kitchens_100", "fpvsum"}:
                raise ValueError(f"{path}:{line} has an unsupported dataset: {dataset!r}.")
            entries[video_id] = (duration, dataset)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Track 4 JSON predictions.")
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--dir", dest="directory_option", type=Path)
    parser.add_argument("--file", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument("--classes", type=Path)
    args = parser.parse_args()
    directory = args.directory_option or args.directory
    if args.file is not None:
        if directory is not None:
            parser.error("use either --file or --dir, not both")
        if not args.file.is_file():
            parser.error(f"Prediction file not found: {args.file}")
    else:
        if directory is None:
            parser.error("provide --file or the output directory with --dir")
        if not directory.is_dir():
            parser.error(f"Output directory not found: {directory}")

    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")
    try:
        manifest = _read_manifest(args.manifest)
    except ValueError as exc:
        parser.error(str(exc))

    if args.file is not None:
        try:
            data = json.loads(args.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL {args.file.name}: {exc}")
            return 1
        video_id = args.file.stem
        duration, dataset = manifest.get(video_id, (None, None))
        valid, errors, warnings = validate_output(
            data,
            expected_video_id=video_id,
            manifest_duration=duration,
            classes_path=args.classes,
            dataset=dataset,
        )
        print(
            f"{'PASS' if valid else 'FAIL'} {args.file.name}"
            + (f": {' | '.join(errors)}" if errors else "")
        )
        if args.verbose and warnings:
            print(f"WARN {args.file.name}: {' | '.join(warnings)}")
        return 0 if valid else 1

    files = sorted(directory.glob("*.json"))
    files_by_id = {path.stem: path for path in files}
    invalid = 0
    warning_count = 0
    team_id = None

    for video_id, (duration, dataset) in manifest.items():
        if video_id not in files_by_id:
            invalid += 1
            print(f"FAIL {video_id}.json: required manifest output is missing.")

    for video_id, path in files_by_id.items():
        if manifest and video_id not in manifest:
            invalid += 1
            print(f"FAIL {path.name}: video_id is not present in the manifest.")
            continue
        duration, dataset = manifest.get(video_id, (None, None))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid += 1
            print(f"FAIL {path.name}: {exc}")
            continue

        valid, errors, warnings = validate_output(
            data,
            expected_video_id=video_id,
            manifest_duration=duration,
            classes_path=args.classes,
            dataset=dataset,
        )
        current_team_id = data.get("team_id") if isinstance(data, dict) else None
        if valid and team_id is None:
            team_id = current_team_id
        elif valid and current_team_id != team_id:
            valid = False
            errors.append(
                f"team_id {current_team_id!r} does not match the other outputs ({team_id!r})."
            )

        warning_count += len(warnings)
        if not valid:
            invalid += 1
            print(f"FAIL {path.name}: {' | '.join(errors)}")
        elif warnings:
            print(f"WARN {path.name}: {' | '.join(warnings)}")

    print(
        f"Validated {len(files)} files; manifest={len(manifest)} "
        f"invalid={invalid}; warnings={warning_count}"
    )
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
