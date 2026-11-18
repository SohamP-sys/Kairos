"""Extract 2 s / 4 s AVION logits for one EPIC video into an NPZ cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from avion_epic import AVIONEpicClassifier


def probe_duration(video_path: Path) -> float:
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return float(result.stdout.strip())


def decode_frames(video_path: Path, fps: float = 8.0) -> np.ndarray:
    import subprocess

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    width, height = map(int, probe.stdout.strip().split(","))
    scale = "scale=-2:224,crop=224:224" if width >= height else "scale=224:-2,crop=224:224"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps},{scale}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=True,
    )
    raw = proc.stdout
    if not raw:
        return np.empty((0, 224, 224, 3), dtype=np.uint8)

    frame_bytes = 224 * 224 * 3
    count = len(raw) // frame_bytes
    if count == 0:
        return np.empty((0, 224, 224, 3), dtype=np.uint8)
    frames = np.frombuffer(raw[:count * frame_bytes], dtype=np.uint8)
    return frames.reshape(count, 224, 224, 3)


def center_crop_224(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.empty((0, 224, 224, 3), dtype=np.uint8)
    if frames.shape[1:3] == (224, 224):
        return frames

    try:
        import cv2

        cropped = []
        for frame in frames:
            height, width = frame.shape[:2]
            scale = 224 / min(height, width)
            resized = cv2.resize(
                frame,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
            h, w = resized.shape[:2]
            y = (h - 224) // 2
            x = (w - 224) // 2
            cropped.append(resized[y:y + 224, x:x + 224])
        return np.asarray(cropped, dtype=np.uint8)
    except ImportError:
        # Center crop and resize without OpenCV using NumPy nearest-neighbor.
        cropped = []
        for frame in frames:
            height, width = frame.shape[:2]
            if width >= height:
                scale = 224.0 / height
                resized_width = int(round(width * scale))
                resized = np.asarray(
                    [
                        frame[
                            int(round(y / scale)),
                            np.minimum(
                                resized_width - 1,
                                np.arange(resized_width),
                            ),
                        ]
                        for y in range(224)
                    ]
                )
            else:
                scale = 224.0 / width
                resized_height = int(round(height * scale))
                resized = np.asarray(
                    [
                        frame[
                            np.minimum(resized_height - 1, np.arange(resized_height)),
                            int(round(x / scale)),
                        ].copy()
                        for x in range(224)
                    ]
                ).transpose(1, 0, 2)
            h, w = resized.shape[:2]
            y = max(0, (h - 224) // 2)
            x = max(0, (w - 224) // 2)
            cropped.append(resized[y:y + 224, x:x + 224])
        return np.asarray(cropped, dtype=np.uint8)


def uniform_clip_frames(
    frames: np.ndarray,
    start_sec: float,
    end_sec: float,
    fps: float,
    num_frames: int = 16,
) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    start_index = int(round(start_sec * fps))
    end_index = int(round(end_sec * fps))
    ids = np.linspace(
        start_index,
        max(start_index, end_index - 1),
        num_frames,
    )
    ids = np.clip(np.rint(ids).astype(int), 0, len(frames) - 1)
    if end_index <= start_index:
        ids[:] = np.clip(start_index, 0, len(frames) - 1)
    return frames[ids]


def make_windows(duration: float, scale: float) -> np.ndarray:
    starts = np.arange(0.0, max(0.0, duration - scale + 0.999), 1.0)
    if len(starts) == 0:
        starts = np.asarray([0.0])
    windows = []
    for start in starts:
        end = min(duration, start + scale)
        if end - start >= 0.5:
            windows.append([float(start), float(end)])
    return np.asarray(windows, dtype=np.float32)


def extract_avion_cache(
    video_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
    batch_size: int = 1,
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)
    checkpoint_path = Path(
        checkpoint_path
        or ROOT / "weights" / "avion" / "avion_finetune_cls_lavila_vitb_best.pt"
    )
    if batch_size < 1:
        raise ValueError("AVION batch size must be at least 1")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = probe_duration(video_path)
    sample_fps = 8.0
    print(f"[AVION] Decoding {video_path.name} at {sample_fps:g} fps...", flush=True)
    raw_frames = decode_frames(video_path, fps=sample_fps)
    frames = center_crop_224(raw_frames)
    del raw_frames
    if len(frames) == 0:
        raise RuntimeError(f"No video frames could be decoded from {video_path}")
    print(f"[AVION] Decoded {len(frames)} frames.", flush=True)

    print("[AVION] Loading checkpoint...", flush=True)
    classifier = AVIONEpicClassifier(
        checkpoint_path,
        ROOT / "third_party" / "AVION",
    )
    priors_path = ROOT / "weights" / "epic_priors.npz"
    if not classifier.enabled:
        raise RuntimeError(
            f"AVION model could not be loaded from {checkpoint_path}"
        )
    if not classifier.load_action_mapping(priors_path):
        raise RuntimeError("AVION action mapping could not be loaded")
    print(f"[AVION] Model ready on {classifier.device}.", flush=True)

    cache: dict[str, np.ndarray] = {
        "duration_sec": np.asarray([duration], dtype=np.float32),
    }
    for scale_tag, scale in (("2s", 2.0), ("4s", 4.0)):
        windows = make_windows(duration, scale)
        logits_batches = []
        progress_step = max(1, len(windows) // 20)
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start:start + batch_size]
            clips = np.asarray(
                [
                    uniform_clip_frames(
                        frames, window_start, window_end, sample_fps
                    )
                    for window_start, window_end in batch_windows
                ],
                dtype=np.uint8,
            )
            logits_batches.append(
                classifier.predict_joint_logits(
                    clips,
                    batch_size=batch_size,
                )
            )
            completed = min(start + len(batch_windows), len(windows))
            if completed == len(windows) or completed % progress_step < batch_size:
                print(f"[AVION] {scale_tag}: {completed}/{len(windows)} windows", flush=True)
            del clips
        joint_logits = np.concatenate(logits_batches, axis=0)
        verb_probs, noun_probs = classifier.marginalize(joint_logits)
        cache[f"windows_{scale_tag}"] = windows
        cache[f"joint_logits_{scale_tag}"] = joint_logits.astype(np.float32)
        cache[f"verb_logits_{scale_tag}"] = np.log(
            verb_probs.astype(np.float32) + 1e-8
        )
        cache[f"noun_logits_{scale_tag}"] = np.log(
            noun_probs.astype(np.float32) + 1e-8
        )
        print(
            f"[AVION] {video_path.name}: {scale_tag} "
            f"{len(windows)} windows extracted",
            flush=True,
        )
        del joint_logits, verb_probs, noun_probs, logits_batches

    motion = np.zeros(max(1, int(np.ceil(duration))), dtype=np.float32)
    drift = np.zeros(max(1, int(np.ceil(duration))), dtype=np.float32)
    cache["motion_1hz"] = motion
    cache["drift_1hz"] = drift

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **cache)
    print(f"[AVION] Wrote cache: {output_path}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "weights" / "avion" / "avion_finetune_cls_lavila_vitb_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    extract_avion_cache(
        args.video,
        args.output,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
