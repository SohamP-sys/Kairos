"""FPV highlight importance and duration-budget selector (dependency-light)."""

from __future__ import annotations

import numpy as np

from typing import Dict, List


def _z(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    return (x - x.mean()) / (x.std() + 1e-6)


def _smooth(x, radius=2):
    if len(x) == 0:
        return x

    sigma = max(1.0, float(radius))
    r = int(3 * sigma)

    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()

    return np.convolve(x, k, mode="same")[:len(x)]


def importance_curve(
    motion: List[float],
    audio_flux: List[float],
    highband: List[float],
    embeddings=None,
    prompt_curve=None,
) -> np.ndarray:

    n = max(
        len(motion),
        len(audio_flux),
        len(highband),
        len(embeddings) if embeddings is not None else 0,
        1,
    )

    def fit(a):
        a = np.asarray(a, dtype=np.float32)

        if len(a) == n:
            return a

        if len(a) == 0:
            return np.zeros(n, dtype=np.float32)

        return np.interp(
            np.linspace(0, len(a) - 1, n),
            np.arange(len(a)),
            a,
        ).astype(np.float32)

    mot = fit(motion)
    flux = fit(audio_flux)
    hb = fit(highband)

    nov = np.zeros(n, dtype=np.float32)

    if embeddings is not None and len(embeddings) > 1:
        e = np.asarray(embeddings, dtype=np.float32)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

        med = e.mean(0)
        med /= np.linalg.norm(med) + 1e-8

        nov = fit(1.0 - (e @ med))

    prom = fit(prompt_curve) if prompt_curve is not None else np.zeros(n, dtype=np.float32)

    # Normalize each component.
    zm = _z(mot)
    za = _z(flux)
    zh = _z(hb)
    zn = _z(nov)
    zp = _z(prom)

    # Weighted contributions.
    cm = 0.38 * zm
    ca = 0.28 * za
    ch = 0.12 * zh
    cn = 0.17 * zn
    cp = 0.05 * zp

    curve = cm + ca + ch + cn + cp
    curve = _smooth(curve, 2.0)

    # ==========================================================
    # DEBUG: COMPONENT CONTRIBUTION ANALYSIS
    # First 11 seconds vs remaining video.
    # ==========================================================

    split = min(11, n)

    if split > 0 and n > split:

        print("\n" + "=" * 65)
        print("[DEBUG] Importance component contribution analysis")
        print(f"Comparing first {split} seconds vs remaining {n-split} samples")
        print("=" * 65)

        components = [
            ("Motion       (weight 0.38)", cm),
            ("Audio flux   (weight 0.28)", ca),
            ("Highband     (weight 0.12)", ch),
            ("Novelty      (weight 0.17)", cn),
            ("Prompt       (weight 0.05)", cp),
        ]

        print(f"{'Component':<30} {'First 11s':>12} {'Rest':>12}")
        print("-" * 65)

        for name, contribution in components:
            first_mean = float(np.mean(contribution[:split]))
            rest_mean = float(np.mean(contribution[split:]))

            print(
                f"{name:<30} "
                f"{first_mean:>12.4f} "
                f"{rest_mean:>12.4f}"
            )

        first_total = float(np.mean(curve[:split]))
        rest_total = float(np.mean(curve[split:]))

        print("-" * 65)
        print(f"{'Combined curve (pre-normalized)':<30} "
              f"{first_total:>12.4f} {rest_total:>12.4f}")
        print("=" * 65 + "\n")

    # Smooth, then normalize to [0, 1].
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-6)

    return curve


def select_highlights(curve: np.ndarray, duration: float, budget_ratio: float = 0.148) -> List[Dict]:
    dur = float(duration)
    budget = max(min(dur, budget_ratio * dur), min(4.05, dur))

    if dur < 4.05:
        return [{"segment": [0.0, round(dur, 2)], "score": 1.0}]

    scores = np.asarray(curve, dtype=np.float32)
    candidates = []

    for length in (4.05, 6.0, 8.0):
        if length > budget + 1e-6 or length > dur:
            continue

        step = max(1, int(length // 2))
        max_start = int(np.floor(min(dur, len(scores)) - length + 1e-8))
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, step):
            end = min(dur, start + length)
            s = float(start)
            e = float(end)

            if e - s < 4.05:
                continue

            val = float(scores[start:min(len(scores), int(np.ceil(e)))].mean())
            candidates.append((val, s, e))

    if not candidates:
        idx = int(np.argmax(scores))
        s = max(0, min(dur - 4.05, idx - 2.025))
        return [{"segment": [round(s, 2), round(s + 4.05, 2)], "score": 0.5}]

    candidates.sort(key=lambda x: (x[2], x[1]))

    quantum = 0.5
    budget_bins = int(budget // quantum)
    n = len(candidates)

    ends = [c[2] for c in candidates]

    from bisect import bisect_right

    prev = []

    for i, (val, s, e) in enumerate(candidates):
        prev.append(bisect_right(ends, s - 1e-6, 0, i))

    dp = np.zeros((n + 1, budget_bins + 1), dtype=np.float32)
    take = np.zeros((n + 1, budget_bins + 1), dtype=np.uint8)

    weights = [
        max(1, int(np.ceil((e - s) / quantum - 1e-8)))
        for _, s, e in candidates
    ]

    values = [
        max(0.0, float(v)) * (e - s)
        for v, s, e in candidates
    ]

    for i in range(1, n + 1):
        w = weights[i - 1]
        pidx = prev[i - 1]

        for b in range(budget_bins + 1):
            best = dp[i - 1, b]

            if w <= b:
                inc = dp[pidx, b - w] + values[i - 1]

                if inc > best + 1e-8:
                    best = inc
                    take[i, b] = 1

            dp[i, b] = best

    chosen = []
    b = budget_bins
    i = n

    while i > 0:
        if take[i, b]:
            v, s, e = candidates[i - 1]
            chosen.append((v, s, e))
            b -= weights[i - 1]
            i = prev[i - 1]
        else:
            i -= 1

    chosen.reverse()

    if not chosen:
        v, s, e = candidates[0]
        chosen = [(v, s, e)]

    chosen.sort(key=lambda x: x[1])

    return [
        {
            "segment": [round(s, 2), round(e, 2)],
            "score": float(np.clip(v, 0, 1)),
        }
        for v, s, e in chosen
    ]
