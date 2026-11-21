from pathlib import Path
from typing import Any, Dict, List, Optional

from schema import DEFAULT_LABELS, load_supported_classes, validate_output

LABEL_ALIASES = {"skiing": "ski"}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def sanitize_highlights(
    raw_highlights: List[Dict[str, Any]],
    duration_sec: float,
    max_ratio: float = 0.145,
) -> List[Dict[str, List[float]]]:
    dur = round(max(float(duration_sec), 0.1), 2)
    min_len = min(4.05, dur)
    budget = min(dur, max(max_ratio * dur, min_len))

    segs: List[List[float]] = []
    for item in raw_highlights or []:
        if not isinstance(item, dict):
            continue
        seg = item.get("segment")
        if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
            continue
        try:
            s = _clamp(float(seg[0]), 0.0, dur)
            e = _clamp(float(seg[1]), 0.0, dur)
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            continue
        if e - s <= 0.01:
            continue
        if e - s < min_len:
            mid = 0.5 * (s + e)
            s = _clamp(mid - 0.5 * min_len, 0.0, max(0.0, dur - min_len))
            e = _clamp(s + min_len, 0.0, dur)
        segs.append([s, e, score])

    segs.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[float]] = []
    for s, e, score in segs:
        if not merged or s > merged[-1][1] + 0.05:
            merged.append([s, e, score])
        else:
            prev_s, prev_e, prev_score = merged[-1]
            merged[-1] = [prev_s, max(prev_e, e), max(prev_score, score)]

    while len(merged) > 1 and sum(x[1] - x[0] for x in merged) > budget:
        worst_idx = min(
            range(len(merged)),
            key=lambda i: merged[i][2] / max(merged[i][1] - merged[i][0], 0.1),
        )
        merged.pop(worst_idx)

    if len(merged) == 1 and (merged[0][1] - merged[0][0]) > budget:
        s, e, score = merged[0]
        target = max(budget, min_len)
        mid = 0.5 * (s + e)
        ns = _clamp(mid - 0.5 * target, 0.0, max(0.0, dur - target))
        ne = _clamp(ns + target, 0.0, dur)
        merged[0] = [ns, ne, score]

    final: List[Dict[str, List[float]]] = []
    prev_end = -1.0
    for s, e, _ in merged:
        rs = round(s, 2)
        re_ = round(e, 2)
        if rs <= prev_end:
            rs = round(prev_end + 0.01, 2)
        if re_ - rs < 4.0 and dur >= 4.05:
            re_ = round(min(dur, rs + 4.05), 2)
            if re_ - rs < 4.0:
                rs = round(max(0.0, re_ - 4.05), 2)
        if re_ > rs and (re_ - rs >= 4.0 or dur < 4.05) and rs > prev_end:
            final.append({"segment": [rs, re_]})
            prev_end = re_

    if not final:
        mid = 0.5 * dur
        rs = round(_clamp(mid - 0.5 * min_len, 0.0, max(0.0, dur - min_len)), 2)
        re_ = round(_clamp(rs + min_len, 0.0, dur), 2)
        final.append({"segment": [rs, re_]})

    return final


def sanitize_actions(
    raw_actions: List[Dict[str, Any]],
    duration_sec: float,
    vocab: Dict[str, set],
) -> List[Dict[str, Any]]:
    dur = round(max(float(duration_sec), 0.1), 2)
    allowed_verbs = vocab.get("verbs")
    allowed_nouns = vocab.get("nouns")

    cleaned: List[Dict[str, Any]] = []
    for act in raw_actions or []:
        if not isinstance(act, dict):
            continue
        seg = act.get("segment")
        if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
            continue
        try:
            s = round(_clamp(float(seg[0]), 0.0, dur), 2)
            e = round(_clamp(float(seg[1]), 0.0, dur), 2)
            raw_score = float(act.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        verb = str(act.get("verb", "")).strip().lower()
        noun = str(act.get("noun", "")).strip().lower()
        if not verb or not noun:
            continue
        if allowed_verbs is not None and verb not in allowed_verbs:
            continue
        if allowed_nouns is not None and noun not in allowed_nouns:
            continue
        cleaned.append({
            "segment": [s, e],
            "verb": verb,
            "noun": noun,
            "score": round(_clamp(raw_score, 0.0, 1.0), 4),
        })

    cleaned.sort(key=lambda x: x["score"], reverse=True)
    return cleaned[:2000]


def assemble_prediction(
    team_id: str,
    video_id: str,
    duration_sec: float,
    label_name: str = "other",
    label_score: float = 0.5,
    raw_actions: Optional[List[Dict[str, Any]]] = None,
    raw_highlights: Optional[List[Dict[str, Any]]] = None,
    max_ratio: float = 0.145,
    classes_path: Optional[Path] = None,
) -> Dict[str, Any]:
    dur = round(max(float(duration_sec), 0.1), 2)
    vocab = load_supported_classes(classes_path)
    allowed_labels = vocab.get("labels") or vocab.get("categories") or DEFAULT_LABELS
    fallback_label = "other" if "other" in allowed_labels else sorted(allowed_labels)[0]

    lname = str(label_name).strip().lower()
    lname = LABEL_ALIASES.get(lname, lname)
    if lname not in allowed_labels:
        lname = fallback_label
    try:
        lscore = round(_clamp(float(label_score), 0.0, 1.0), 4)
    except (TypeError, ValueError):
        lscore = 0.5

    out = {
        "team_id": team_id,
        "video_id": video_id,
        "duration_sec": dur,
        "label": {"name": lname, "score": lscore},
        "actions": sanitize_actions(raw_actions or [], dur, vocab),
        "highlights": sanitize_highlights(raw_highlights or [], dur, max_ratio=max_ratio),
    }

    ok, _, _ = validate_output(out, expected_video_id=video_id, classes_path=classes_path)
    if not ok:
        out["label"] = {"name": fallback_label, "score": 0.5}
        out["actions"] = []
        out["highlights"] = sanitize_highlights([], dur, max_ratio=max_ratio)
    return out
