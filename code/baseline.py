import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from assemble import assemble_prediction


def probe_duration(video_path: Path) -> float:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    val = res.stdout.strip()
    if val and val != "N/A":
        dur = float(val)
        if dur > 0:
            return dur
    raise RuntimeError(f"ffprobe failed for {video_path}: {res.stderr.strip()}")


def extract_motion_curve(video_path: Path, fps: int = 2, w: int = 64, h: int = 36) -> List[float]:
    frame_bytes = w * h
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"scale={w}:{h},fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    diffs: List[float] = []
    prev = proc.stdout.read(frame_bytes)
    while len(prev) == frame_bytes:
        curr = proc.stdout.read(frame_bytes)
        if len(curr) < frame_bytes:
            break
        diffs.append(sum(abs(a - b) for a, b in zip(prev, curr)) / frame_bytes)
        prev = curr
    proc.stdout.close()
    if proc.wait(timeout=30) != 0 or not diffs:
        raise RuntimeError("ffmpeg motion extraction failed")
    return [sum(diffs[i : i + fps]) / len(diffs[i : i + fps]) for i in range(0, len(diffs), fps)]


def select_motion_windows(sec_scores: List[float], duration_sec: float, win_len: float = 6.0) -> List[Dict[str, Any]]:
    dur = float(duration_sec)
    budget = 0.145 * dur
    w = min(win_len, max(4.05, budget), dur) if dur >= 4.05 else dur
    n = len(sec_scores)
    if n == 0:
        return []
    win_bins = max(1, int(round(w)))
    candidates = []
    for start_bin in range(n):
        end_bin = min(n, start_bin + win_bins)
        score = sum(sec_scores[start_bin:end_bin]) / max(1, end_bin - start_bin)
        s = float(start_bin)
        e = min(dur, s + w)
        if e - s < w and dur >= w:
            s, e = max(0.0, dur - w), dur
        candidates.append((score, s, e))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen: List[Dict[str, Any]] = []
    used = 0.0
    for score, s, e in candidates:
        if used + (e - s) > budget and chosen:
            continue
        if any(not (e <= c["segment"][0] - 0.1 or s >= c["segment"][1] + 0.1) for c in chosen):
            continue
        chosen.append({"segment": [round(s, 2), round(e, 2)], "score": round(score, 4)})
        used += e - s
        if used >= budget * 0.95:
            break
    return chosen


def evenly_spaced_windows(duration_sec: float) -> List[Dict[str, Any]]:
    dur = float(duration_sec)
    budget = 0.14 * dur
    win_len = 5.0
    count = max(1, int(budget // win_len))
    if count == 1:
        mid = 0.5 * dur
        return [{"segment": [round(max(0.0, mid - 2.5), 2), round(min(dur, mid + 2.5), 2)], "score": 0.5}]
    step = dur / (count + 1)
    out = []
    for i in range(1, count + 1):
        s = max(0.0, i * step - 0.5 * win_len)
        out.append({"segment": [round(s, 2), round(min(dur, s + win_len), 2)], "score": 0.5})
    return out


def run_baseline_pipeline(video_path: Path, team_id: str) -> Dict[str, Any]:
    video_id = video_path.stem
    try:
        dur = probe_duration(video_path)
    except Exception as exc:
        sys.stderr.write(f"[WARN] Duration probe failed ({exc}); using 10.0s fallback.\n")
        return assemble_prediction(team_id=team_id, video_id=video_id, duration_sec=10.0)
    try:
        raw_hl = select_motion_windows(extract_motion_curve(video_path), dur)
    except Exception as exc:
        sys.stderr.write(f"[WARN] Motion extraction failed ({exc}); using evenly spaced fallback.\n")
        raw_hl = evenly_spaced_windows(dur)
    return assemble_prediction(
        team_id=team_id, video_id=video_id, duration_sec=dur,
        label_name="other", label_score=0.5, raw_actions=[],
        raw_highlights=raw_hl, max_ratio=0.145,
    )
