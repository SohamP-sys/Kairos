import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def probe_duration(video_path: Path, default_duration: float = 60.0) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15).decode().strip()
        val = float(out)
        if val > 0:
            return round(val, 2)
    except Exception:
        pass
    return round(float(default_duration), 2)


def decode_video_frames(
    video_path: Path,
    fps: float = 4.0,
    width: int = 224,
    height: int = 224,
    max_seconds: float = 2400.0,
) -> np.ndarray:
    try:
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(video_path),
            "-t", str(max_seconds),
            "-vf", f"fps={fps},scale={width}:{height}:flags=bicubic",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=180)
        frame_bytes = width * height * 3
        n_frames = len(raw) // frame_bytes
        if n_frames == 0:
            return np.zeros((0, height, width, 3), dtype=np.uint8)
        arr = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
        return arr.reshape((n_frames, height, width, 3))
    except Exception:
        return np.zeros((0, height, width, 3), dtype=np.uint8)


def extract_audio_1hz_features(video_path: Path, num_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
    n_sec = max(1, int(num_seconds))
    flux_1hz = np.zeros(n_sec, dtype=np.float32)
    highband_1hz = np.zeros(n_sec, dtype=np.float32)
    try:
        sr = 16000
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sr),
            "-f", "s16le", "pipe:1",
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=60)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size < sr:
            return flux_1hz, highband_1hz

        win_len = 512
        hop = 500
        freqs = np.fft.rfftfreq(win_len, d=1.0 / sr)
        high_mask = (freqs >= 2000.0) & (freqs <= 7500.0)
        prev_mag: Optional[np.ndarray] = None

        for sec in range(min(n_sec, pcm.size // sr)):
            chunk = pcm[sec * sr : (sec + 1) * sr]
            sec_flux, sec_hb = [], []
            for start in range(0, len(chunk) - win_len, hop):
                frame = chunk[start : start + win_len] * np.hanning(win_len)
                mag = np.abs(np.fft.rfft(frame))
                mag_norm = mag / (np.sum(mag) + 1e-6)
                if prev_mag is not None:
                    sec_flux.append(float(np.sum(np.maximum(0.0, mag_norm - prev_mag))))
                prev_mag = mag_norm
                sec_hb.append(float(np.mean(mag[high_mask])))
            if sec_flux:
                flux_1hz[sec] = float(np.mean(sec_flux))
            if sec_hb:
                highband_1hz[sec] = float(np.mean(sec_hb))
    except Exception:
        pass
    return flux_1hz, highband_1hz
