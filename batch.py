#!/usr/bin/env python3
"""Generate and validate predictions for videos in the Track 4 manifest."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from assemble import assemble_prediction
from baseline import evenly_spaced_windows
from schema import validate_output

EPIC_DATASETS = {"epic", "epic-kitchens-100", "epic_kitchens_100"}
SUPPORTED_DATASETS = EPIC_DATASETS | {"fpvsum"}


def _read_manifest(path: Path) -> list[tuple[str, float, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "duration_sec", "dataset"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have columns: {', '.join(sorted(required))}")

        entries = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            if not video_id or video_id in seen:
                raise ValueError(f"{path}:{line} has an empty or duplicate video_id.")
            seen.add(video_id)
            try:
                duration = float(row["duration_sec"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{line} has an invalid duration_sec.")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{path}:{line} duration_sec must be positive.")
            dataset = (row.get("dataset") or "").strip().lower()
            if dataset not in SUPPORTED_DATASETS:
                raise ValueError(f"{path}:{line} has an unsupported dataset: {dataset!r}.")
            entries.append((video_id, duration, dataset))
    if not entries:
        raise ValueError(f"{path} contains no video entries.")
    return entries


def _normalize_output(
    result: dict,
    team_id: str,
    video_id: str,
    duration: float,
    dataset: str,
    classes_path: Path | None,
) -> dict:
    label = result.get("label") if isinstance(result, dict) else {}
    label_name = label.get("name", "other") if isinstance(label, dict) else "other"
    label_score = label.get("score", 0.5) if isinstance(label, dict) else 0.5

    if dataset in EPIC_DATASETS:
        label_name, label_score = "cooking", 1.0
    elif str(label_name).strip().lower() == "cooking":
        label_name, label_score = "other", 0.5

    return assemble_prediction(
        team_id,
        video_id,
        duration,
        label_name,
        label_score,
        result.get("actions", []) if isinstance(result, dict) else [],
        result.get("highlights", []) if isinstance(result, dict) else [],
        classes_path=classes_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch Track 4 predictions from the manifest."
    )
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Write explicitly marked fallback files for unavailable videos.",
    )
    parser.add_argument("--team-id", default=os.environ.get("TEAM_ID_STR", "TEAM_ID"))
    parser.add_argument("--classes", type=Path)
    args = parser.parse_args()

    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")
    if not args.team_id.strip():
        parser.error("--team-id must not be empty")
    try:
        entries = _read_manifest(args.manifest)
    except ValueError as exc:
        parser.error(str(exc))

    videos_by_id: dict[str, list[Path]] = {}
    if args.video_dir.is_dir():
        for path in args.video_dir.rglob("*.mp4"):
            videos_by_id.setdefault(path.stem, []).append(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    placeholders = 0
    written = 0
    runner = ROOT / "code" / "run_video.py"

    for video_id, duration, dataset in entries:
        sources = videos_by_id.get(video_id, [])
        output_path = args.output_dir / f"{video_id}.json"
        if len(sources) > 1:
            failures.append(video_id)
            print(f"FAIL {video_id}: multiple matching MP4 files found.")
            continue

        if not sources:
            if not args.fill_missing:
                failures.append(video_id)
                print(f"FAIL {video_id}: source video is missing.")
                continue
            result = {
                "actions": [],
                "highlights": evenly_spaced_windows(duration),
            }
            prediction = _normalize_output(
                result, args.team_id, video_id, duration, dataset, args.classes
            )
            placeholders += 1
            print(f"FALLBACK {video_id}: missing source; not a video prediction.")
        else:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    str(sources[0]),
                    str(output_path),
                    "--dataset",
                    dataset,
                    "--manifest-duration",
                    str(duration),
                ],
                check=False,
            )
            if completed.returncode:
                failures.append(video_id)
                print(
                    f"FAIL {video_id}: inference exited with status "
                    f"{completed.returncode}."
                )
                continue
            try:
                result = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(video_id)
                print(f"FAIL {video_id}: could not read prediction: {exc}")
                continue
            prediction = _normalize_output(
                result, args.team_id, video_id, duration, dataset, args.classes
            )

        valid, errors, warnings = validate_output(
            prediction,
            expected_video_id=video_id,
            manifest_duration=duration,
            classes_path=args.classes,
            dataset=dataset,
        )
        if not valid:
            failures.append(video_id)
            print(f"FAIL {video_id}: {' | '.join(errors)}")
            continue

        output_path.write_text(
            json.dumps(prediction, indent=2) + "\n", encoding="utf-8"
        )
        written += 1
        if warnings:
            print(f"WARN {video_id}: {' | '.join(warnings)}")

    print(
        f"Batch complete: manifest={len(entries)} written={written} "
        f"placeholders={placeholders} failures={len(failures)}"
    )
    if failures:
        print("Failures:", ", ".join(failures))
    if placeholders:
        print(
            "This batch is not submission-ready because it contains placeholders.",
            file=sys.stderr,
        )
    return 1 if failures or placeholders else 0


if __name__ == "__main__":
    raise SystemExit(main())
