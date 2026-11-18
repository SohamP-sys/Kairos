import numpy as np
from typing import Dict, List, Tuple
from assemble import sanitize_highlights


def select_epic_highlights(
    actions: List[Dict],
    duration_sec: float,
    budget_ratio: float = 0.145,
    mode: str = "noisy_or",
    gap_fill: bool = True,
) -> List[Dict]:
    dur = float(duration_sec)
    budget = budget_ratio * dur

    if budget < 4.05:
        if actions:
            best_act = max(actions, key=lambda a: a["score"])
            mid = 0.5 * (best_act["segment"][0] + best_act["segment"][1])
            cands = [{"segment": [max(0.0, mid - 2.025), min(dur, mid + 2.025)], "score": best_act["score"]}]
        else:
            cands = []
        return sanitize_highlights(cands, dur, max_ratio=budget_ratio)

    top_actions = sorted(actions, key=lambda a: a["score"], reverse=True)[:400]
    if not top_actions:
        return sanitize_highlights([], dur, max_ratio=budget_ratio)

    pair_counts: Dict[Tuple[str, str], int] = {}
    for act in top_actions:
        p = (act["verb"], act["noun"])
        pair_counts[p] = pair_counts.get(p, 0) + 1
    pair_weights = {p: 1.0 / np.sqrt(float(cnt)) for p, cnt in pair_counts.items()}

    candidates = []
    seen_grids = set()
    for act in top_actions[:120]:
        mid = 0.5 * (act["segment"][0] + act["segment"][1])
        for w_len in (4.05, 6.0, 8.0):
            if w_len > budget:
                continue
            s_grid = round(max(0.0, min(dur - w_len, mid - 0.5 * w_len)) * 2.0) / 2.0
            if (s_grid, w_len) in seen_grids:
                continue
            seen_grids.add((s_grid, w_len))
            e = min(dur, s_grid + w_len)
            if e - s_grid < 4.05:
                continue

            pair_probs: Dict[Tuple[str, str], float] = {}
            for inst in top_actions:
                is_s, is_e = inst["segment"]
                inter = max(0.0, min(e, is_e) - max(s_grid, is_s))
                if inter > 0.25:
                    cov_ratio = min(1.0, inter / max(0.5, is_e - is_s))
                    p_val = float(inst["score"]) * cov_ratio
                    pk = (inst["verb"], inst["noun"])
                    if mode == "max":
                        pair_probs[pk] = max(pair_probs.get(pk, 0.0), p_val)
                    else:
                        pair_probs[pk] = 1.0 - (1.0 - pair_probs.get(pk, 0.0)) * (1.0 - p_val)
            if pair_probs:
                candidates.append({"s": s_grid, "e": e, "len": e - s_grid, "pair_probs": pair_probs})

    if not candidates:
        return sanitize_highlights([], dur, max_ratio=budget_ratio)

    selected: List[Dict] = []
    covered_prob: Dict[Tuple[str, str], float] = {p: 0.0 for p in pair_counts}
    used_sec = 0.0

    while True:
        best_cand, best_rate, best_gain = None, 0.0, 0.0
        for cand in candidates:
            if used_sec + cand["len"] > budget:
                continue
            if any(not (cand["e"] + 0.5 <= sel["s"] or cand["s"] >= sel["e"] + 0.5) for sel in selected):
                continue
            gain = 0.0
            for pk, p_val in cand["pair_probs"].items():
                cur = covered_prob.get(pk, 0.0)
                new_p = max(cur, p_val) if mode == "max" else (1.0 - (1.0 - cur) * (1.0 - p_val))
                gain += pair_weights[pk] * (new_p - cur)
            rate = gain / cand["len"]
            if rate > best_rate:
                best_rate, best_gain, best_cand = rate, gain, cand

        if best_cand is None or best_gain < 1e-3:
            break
        selected.append({"s": best_cand["s"], "e": best_cand["e"], "score": best_gain})
        used_sec += best_cand["len"]
        for pk, p_val in best_cand["pair_probs"].items():
            cur = covered_prob.get(pk, 0.0)
            covered_prob[pk] = max(cur, p_val) if mode == "max" else (1.0 - (1.0 - cur) * (1.0 - p_val))

    if gap_fill and (budget - used_sec) >= 4.20:
        selected.sort(key=lambda x: x["s"])
        intervals = [(0.0, 0.0)] + [(x["s"], x["e"]) for x in selected] + [(dur, dur)]
        gaps = [(intervals[i + 1][0] - intervals[i][1], 0.5 * (intervals[i][1] + intervals[i + 1][0]))
                for i in range(len(intervals) - 1) if intervals[i + 1][0] - intervals[i][1] >= 10.0]
        gaps.sort(reverse=True)
        for _, mid in gaps:
            if budget - used_sec < 4.20:
                break
            s = max(0.0, min(dur - 4.10, mid - 2.05))
            e = s + 4.10
            if all(e + 0.2 <= sel["s"] or s >= sel["e"] + 0.2 for sel in selected):
                selected.append({"s": s, "e": e, "score": 0.05})
                used_sec += 4.10

    formatted = [{"segment": [x["s"], x["e"]], "score": x["score"]} for x in selected]
    return sanitize_highlights(formatted, dur, max_ratio=budget_ratio)
