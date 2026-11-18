import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_LABELS = {
    "cooking", "bike", "bikepolo", "horse", "jumping", "motor",
    "parkour", "ski", "surfing", "boxing", "longboarding",
    "plane", "rockclimbing", "scuba", "skate", "other",
}


def load_supported_classes(classes_path: Optional[Path] = None) -> Dict[str, set]:
    candidates = []
    if classes_path is not None:
        candidates.append(Path(classes_path))
    here = Path(__file__).resolve().parent
    candidates.extend([
        Path("supported_classes.json"),
        here.parent / "supported_classes.json",
        here.parent.parent / "supported_classes.json",
    ])
    for p in candidates:
        if p.is_file():
            raw = json.loads(p.read_text(encoding="utf-8"))
            out: Dict[str, set] = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    out[k] = {str(x).lower() for x in v.keys()}
                elif isinstance(v, list):
                    out[k] = {str(x).lower() for x in v}
            return out
    return {}


def _is_two_decimals(val: float) -> bool:
    return abs(round(val, 2) - val) < 1e-6


def validate_output(
    data: Dict[str, Any],
    expected_video_id: Optional[str] = None,
    manifest_duration: Optional[float] = None,
    classes_path: Optional[Path] = None,
) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(data, dict):
        return False, ["Root JSON must be an object."], warnings

    required = {"team_id", "video_id", "duration_sec", "label", "actions", "highlights"}
    missing = required - set(data.keys())
    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")
        return False, errors, warnings

    team_id = data.get("team_id")
    if not isinstance(team_id, str) or not team_id.strip():
        errors.append("team_id must be a non-empty string.")

    video_id = data.get("video_id")
    if not isinstance(video_id, str) or not video_id.strip():
        errors.append("video_id must be a non-empty string.")
    elif expected_video_id and video_id != expected_video_id:
        errors.append(f"video_id '{video_id}' != expected '{expected_video_id}'.")

    dur = data.get("duration_sec")
    if not isinstance(dur, (int, float)) or isinstance(dur, bool) or dur <= 0:
        errors.append("duration_sec must be a positive number.")
        return False, errors, warnings
    dur = float(dur)
    if manifest_duration is not None and abs(dur - manifest_duration) > 0.5:
        errors.append(f"duration_sec {dur} differs from manifest {manifest_duration} by > 0.5s.")

    vocab = load_supported_classes(classes_path)
    allowed_labels = vocab.get("labels") or vocab.get("categories") or DEFAULT_LABELS
    allowed_verbs = vocab.get("verbs")
    allowed_nouns = vocab.get("nouns")

    label = data.get("label")
    if not isinstance(label, dict):
        errors.append("label must be an object with 'name' and 'score'.")
    else:
        lname = label.get("name")
        lscore = label.get("score")
        if not isinstance(lname, str) or lname != lname.lower() or lname not in allowed_labels:
            errors.append(f"Invalid label.name: {lname!r}")
        if not isinstance(lscore, (int, float)) or isinstance(lscore, bool) or not (0.0 <= float(lscore) <= 1.0):
            errors.append(f"Invalid label.score: {lscore!r}")

    actions = data.get("actions")
    if not isinstance(actions, list):
        errors.append("actions must be a list.")
    else:
        if len(actions) > 2000:
            errors.append(f"Too many actions: {len(actions)} > 2000.")
        for idx, act in enumerate(actions):
            if not isinstance(act, dict):
                errors.append(f"actions[{idx}] is not an object.")
                continue
            seg = act.get("segment")
            if not (isinstance(seg, list) and len(seg) == 2 and all(isinstance(x, (int, float)) for x in seg)):
                errors.append(f"actions[{idx}].segment must be [start, end].")
                continue
            s, e = float(seg[0]), float(seg[1])
            if not (0.0 <= s < e <= dur + 0.5):
                errors.append(f"actions[{idx}].segment [{s}, {e}] out of bounds [0, {dur + 0.5}].")
            if not (_is_two_decimals(s) and _is_two_decimals(e)):
                warnings.append(f"actions[{idx}].segment [{s}, {e}] has >2 decimal places.")
            verb, noun, score = act.get("verb"), act.get("noun"), act.get("score")
            if not isinstance(verb, str) or (allowed_verbs and verb not in allowed_verbs):
                errors.append(f"actions[{idx}] invalid verb: {verb!r}")
            if not isinstance(noun, str) or (allowed_nouns and noun not in allowed_nouns):
                errors.append(f"actions[{idx}] invalid noun: {noun!r}")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not (0.0 <= float(score) <= 1.0):
                errors.append(f"actions[{idx}] invalid score: {score!r}")

    highlights = data.get("highlights")
    if not isinstance(highlights, list) or len(highlights) == 0:
        errors.append("highlights must be a non-empty list.")
    else:
        prev_end = -1.0
        total_hl = 0.0
        has_min_len = False
        for idx, hl in enumerate(highlights):
            if not isinstance(hl, dict):
                errors.append(f"highlights[{idx}] is not an object.")
                continue
            seg = hl.get("segment")
            if not (isinstance(seg, list) and len(seg) == 2 and all(isinstance(x, (int, float)) for x in seg)):
                errors.append(f"highlights[{idx}].segment must be [start, end].")
                continue
            s, e = float(seg[0]), float(seg[1])
            if not (0.0 <= s < e <= dur + 0.5):
                errors.append(f"highlights[{idx}].segment [{s}, {e}] out of bounds [0, {dur + 0.5}].")
            if s < prev_end - 1e-6:
                errors.append(f"highlights[{idx}] overlaps or is unsorted ({s} < {prev_end}).")
            if not (_is_two_decimals(s) and _is_two_decimals(e)):
                errors.append(f"highlights[{idx}].segment [{s}, {e}] exceeds 2 decimal places.")
            length = round(e - s, 4)
            if length >= 4.0:
                has_min_len = True
            else:
                warnings.append(f"highlights[{idx}] length {length}s < 4.0s (ignored in scoring).")
            total_hl += length
            prev_end = e

        if not has_min_len:
            errors.append("highlights must contain at least one segment >= 4.0s.")
        if dur >= 27.0 and total_hl > 0.15 * dur + 1e-4:
            warnings.append(f"Total highlight length {total_hl:.2f}s exceeds 15% ({0.15 * dur:.2f}s).")

    return len(errors) == 0, errors, warnings
