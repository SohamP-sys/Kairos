"""Lightweight kernel temporal segmentation and strict duration-budget highlight selection."""

from __future__ import annotations

import numpy as np


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def kts_segments(features, min_len_sec=4.05, max_len_sec=20.0,
                 penalty=0.18, fps=1.0):
    """Approximate KTS segmentation using normalized features and DP segment costs."""

    x = _unit(features)
    n = len(x)

    if n == 0:
        return []

    min_n = max(1, int(np.ceil(min_len_sec * fps)))
    max_n = max(min_n, int(max_len_sec * fps))

    cs = np.vstack([
        np.zeros((1, x.shape[1]), dtype=np.float64),
        np.cumsum(x, axis=0, dtype=np.float64)
    ])
    c2 = np.r_[0.0, np.cumsum(np.sum(x * x, axis=1), dtype=np.float64)]

    def cost(a, b):
        z = cs[b] - cs[a]
        length = b - a
        return float((c2[b] - c2[a]) - np.dot(z, z) / max(1, length))

    dp = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=int)
    dp[0] = 0.0

    for b in range(min_n, n + 1):
        lo = max(0, b - max_n)
        hi = b - min_n

        for a in range(lo, hi + 1):
            if not np.isfinite(dp[a]):
                continue

            val = dp[a] + cost(a, b) + penalty
            if val < dp[b]:
                dp[b] = val
                prev[b] = a

    if prev[n] < 0:
        return [[0.0, float(n / fps)]]

    cuts = []
    b = n

    while b > 0 and prev[b] >= 0:
        a = int(prev[b])
        cuts.append([a / fps, b / fps])
        b = a

    cuts.reverse()

    if b > 0 and cuts:
        cuts[0][0] = 0.0
    elif b > 0:
        cuts = [[0.0, n / fps]]

    out = []
    for s, e in cuts:
        if out and e - out[-1][0] < min_len_sec:
            out[-1][1] = e
        else:
            out.append([s, e])

    if out and n / fps - out[-1][1] < min_len_sec:
        out[-1][1] = n / fps

    return out


def knapsack_highlights(segments, scores, duration,
                       budget_ratio=0.148, min_len=4.05):
    """
    Select non-overlapping highlights under a STRICT duration budget.

    If no valid candidate can fit, return an empty list rather than violating
    the budget by forcing a minimum-length highlight.
    """

    duration = float(duration)

    if duration <= 0 or not segments or not scores:
        return []

    # Strict budget: never inflate it to the minimum highlight length.
    budget = min(duration, max(0.0, budget_ratio * duration))

    if budget <= 0:
        return []

    items = []

    for (s, e), sc in zip(segments, scores):
        s = max(0.0, float(s))
        e = min(duration, float(e))
        d = e - s

        # Candidates must satisfy both the minimum duration and strict budget.
        if d >= min_len and d <= budget + 1e-9:
            items.append((s, e, max(0.0, float(sc)) * d))

    if not items:
        return []

    # Sort by ending time for weighted interval scheduling.
    items.sort(key=lambda x: (x[1], x[0]))
    ends = [it[1] for it in items]

    from bisect import bisect_right

    pred = [
        bisect_right(ends, it[0] + 1e-9, 0, i)
        for i, it in enumerate(items)
    ]

    # Conservative budget discretization: rounding duration UP means selected
    # segments cannot exceed the actual budget because of bin rounding.
    quantum = 0.5
    bins = int(np.floor(budget / quantum + 1e-9))

    if bins <= 0:
        return []

    weights = [
        max(1, int(np.ceil((it[1] - it[0]) / quantum - 1e-8)))
        for it in items
    ]

    values = [it[2] for it in items]

    n = len(items)
    dp = np.zeros((n + 1, bins + 1), dtype=np.float32)
    take = np.zeros((n + 1, bins + 1), dtype=np.uint8)

    for i, item in enumerate(items, start=1):
        w = weights[i - 1]
        p = pred[i - 1]

        for b in range(bins + 1):
            dp[i, b] = dp[i - 1, b]

            if w <= b:
                candidate_value = dp[p, b - w] + values[i - 1]

                if candidate_value > dp[i, b]:
                    dp[i, b] = candidate_value
                    take[i, b] = 1

    # Backtrack.
    chosen = []
    b = bins
    i = n

    while i > 0:
        if take[i, b]:
            s, e, v = items[i - 1]
            chosen.append((s, e, v))
            b -= weights[i - 1]
            i = pred[i - 1]
        else:
            i -= 1

    chosen.reverse()
    chosen.sort(key=lambda x: x[0])

    return [
        {
            "segment": [round(s, 2), round(e, 2)],
            "score": float(np.clip(v / max(1e-6, e - s), 0.0, 1.0))
        }
        for s, e, v in chosen
    ]
