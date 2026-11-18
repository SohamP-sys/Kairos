#!/usr/bin/env python3
"""Validate prediction files and, optionally, completeness against a manifest."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from schema import validate_output


def _read_manifest(path: Path) -> dict[str, tuple[float | None, str]]:
    expected = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            video_id = (row.get("video_id") or row.get("id") or "").strip()
            if not video_id:
                raise ValueError("manifest contains a row with an empty video_id")
            if video_id in expected:
                raise ValueError(f"manifest contains duplicate video_id {video_id!r}")
            try:
                duration = float(row.get("duration_sec") or row.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0:
                raise ValueError(f"manifest duration is invalid for {video_id!r}")
            dataset = (row.get("dataset") or "").strip().lower()
            if dataset not in {
                "epic_kitchens_100", "epic", "epic-kitchens-100", "fpvsum"
            }:
                raise ValueError(
                    f"manifest dataset is unsupported for {video_id!r}: {dataset!r}"
                )
            expected[video_id] = (duration if duration > 0 else None, dataset)
    if not expected:
        raise ValueError("manifest is empty")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--team-id")
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.json"))
    try:
        expected = _read_manifest(args.manifest) if args.manifest else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    expected_team_id = args.team_id
    seen_team_ids = set()
    failures = 0
    warnings = 0

    if expected is not None:
        file_ids = {file.stem for file in files}
        missing = sorted(set(expected) - file_ids)
        extra = sorted(file_ids - set(expected))
        if missing:
            failures += len(missing)
            print(f"FAIL missing manifest outputs ({len(missing)}): {', '.join(missing)}")
        if extra:
            failures += len(extra)
            print(f"FAIL outputs not in manifest ({len(extra)}): {', '.join(extra)}")

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            manifest_duration, dataset = (
                expected.get(path.stem, (None, "")) if expected is not None else (None, "")
            )
            ok, errors, file_warnings = validate_output(
                data,
                expected_video_id=path.stem,
                manifest_duration=manifest_duration,
                classes_path=args.classes,
            )
            if not isinstance(data, dict):
                team_id = None
                label_name = None
            else:
                team_id = data.get("team_id")
                label = data.get("label")
                label_name = label.get("name") if isinstance(label, dict) else None
            if isinstance(team_id, str) and team_id.strip():
                seen_team_ids.add(team_id)
            if expected_team_id and team_id != expected_team_id:
                errors.append(
                    f"team_id {team_id!r} does not match expected {expected_team_id!r}"
                )
            if expected is not None and dataset in {
                "epic_kitchens_100", "epic", "epic-kitchens-100"
            } and label_name != "cooking":
                errors.append(f"EPIC-KITCHENS label must be 'cooking', got {label_name!r}")
            if expected is not None and dataset == "fpvsum" and label_name == "cooking":
                errors.append("FPVSum videos cannot use the 'cooking' label")
            if file_warnings:
                warnings += len(file_warnings)
            if errors or not ok:
                failures += 1
                print(f"FAIL {path.name}: " + " | ".join(errors))
            elif file_warnings:
                print(f"WARN {path.name}: " + " | ".join(file_warnings))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}")

    if len(seen_team_ids) > 1:
        failures += 1
        print(f"FAIL inconsistent team_id values: {', '.join(sorted(seen_team_ids))}")
    if not files:
        failures += 1
        print(f"FAIL no JSON files found in {args.directory}")

    print(
        f"Validated {len(files)} files; invalid={failures}; warnings={warnings}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
