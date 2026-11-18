#!/usr/bin/env python3
"""Run inference for every available video in a directory or manifest."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from assemble import assemble_prediction
from baseline import evenly_spaced_windows
from schema import validate_output

EPIC_DATASETS = {"epic_kitchens_100", "epic", "epic-kitchens-100"}
FPV_DATASETS = {"fpvsum"}


def _validate_prediction(
    output_path: Path,
    video_id: str,
    duration: float,
    dataset: str,
    team_id: str,
    classes_path: Path | None,
) -> list[str]:
    try:
        prediction = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read prediction JSON: {exc}"]

    ok, errors, _ = validate_output(
        prediction,
        expected_video_id=video_id,
        manifest_duration=duration if duration > 0 else None,
        classes_path=classes_path,
    )
    if not ok:
        return errors
    if prediction["team_id"] != team_id:
        errors.append(
            f"team_id {prediction['team_id']!r} does not match {team_id!r}"
        )
    label = prediction["label"]["name"]
    if dataset in EPIC_DATASETS and label != "cooking":
        errors.append(f"EPIC-KITCHENS label must be 'cooking', got {label!r}")
    if dataset in FPV_DATASETS and label == "cooking":
        errors.append("FPVSum videos cannot use the 'cooking' label")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch inference for videos, optionally using a manifest"
    )
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Write metadata-only placeholders for missing videos (not valid predictions)",
    )
    parser.add_argument(
        "--team-id",
        default=os.environ.get("TEAM_ID_STR", "TEAM_ID"),
    )
    parser.add_argument("--classes", type=Path)
    args = parser.parse_args()

    if not args.team_id.strip():
        parser.error("--team-id must not be empty")

    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            print(f"Manifest is empty: {args.manifest}", file=sys.stderr)
            return 1
        entries = []
        seen_video_ids = set()
        for row in rows:
            video_id = (row.get("video_id") or row.get("id") or "").strip()
            if not video_id:
                print("Manifest contains a row with an empty video_id.", file=sys.stderr)
                return 1
            if video_id in seen_video_ids:
                print(f"Manifest contains duplicate video_id {video_id!r}.", file=sys.stderr)
                return 1
            seen_video_ids.add(video_id)
            try:
                duration = float(row.get("duration_sec") or row.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0:
                print(f"Manifest duration is invalid for {video_id!r}.", file=sys.stderr)
                return 1
            dataset = (row.get("dataset") or "").strip().lower()
            if dataset not in EPIC_DATASETS | FPV_DATASETS:
                print(
                    f"Manifest dataset is unsupported for {video_id!r}: {dataset!r}",
                    file=sys.stderr,
                )
                return 1
            entries.append((video_id, duration, dataset))
    else:
        entries = [(path.stem, 0.0, "") for path in sorted(args.video_dir.glob("*.mp4"))]

    if not entries:
        print("No manifest entries or .mp4 inputs found.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    placeholders = 0
    written = 0
    for video_id, duration, dataset in entries:
        if not video_id:
            failures.append("manifest row has an empty video_id")
            continue

        source = args.video_dir / f"{video_id}.mp4"
        output = args.output_dir / f"{video_id}.json"
        if source.is_file():
            env = os.environ.copy()
            env["TEAM_ID_STR"] = args.team_id
            if dataset:
                env["KAIROS_DATASET"] = dataset
            result = subprocess.run(
                [str(ROOT / "run.sh"), str(source), str(output)],
                check=False,
                env=env,
            )
            if result.returncode:
                failures.append(f"{video_id}: inference exited with {result.returncode}")
                continue
        elif args.fill_missing and duration > 0:
            print(
                f"WARNING: {video_id} has no source video; writing a metadata-only "
                "placeholder that must not be submitted.",
                file=sys.stderr,
            )
            label = "cooking" if dataset in EPIC_DATASETS else "other"
            prediction = assemble_prediction(
                args.team_id,
                video_id,
                duration,
                label,
                0.3,
                [],
                evenly_spaced_windows(duration),
                classes_path=args.classes,
            )
            output.write_text(
                json.dumps(prediction, indent=2) + "\n", encoding="utf-8"
            )
            placeholders += 1
        else:
            failures.append(f"{video_id}: source video is missing ({source})")
            continue

        if not output.is_file():
            failures.append(f"{video_id}: inference did not create {output}")
            continue
        errors = _validate_prediction(
            output, video_id, duration, dataset, args.team_id, args.classes
        )
        if errors:
            failures.append(f"{video_id}: " + " | ".join(errors))
            continue
        written += 1

    print(
        f"Batch complete: manifest/inputs={len(entries)} "
        f"validated={written} placeholders={placeholders} failures={len(failures)}"
    )
    if failures:
        print("Failures:\n- " + "\n- ".join(failures), file=sys.stderr)
    if placeholders:
        print(
            "This batch is not submission-ready because it contains placeholders.",
            file=sys.stderr,
        )
    return 1 if failures or placeholders else 0


if __name__ == "__main__":
    raise SystemExit(main())
