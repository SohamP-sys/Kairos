import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_epic_vocab_and_priors(weights_dir: Optional[Path] = None) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Tuple[float, ...]]:
    prior_file = (Path(weights_dir) / "epic_priors.npz") if weights_dir else Path("weights/epic_priors.npz")
    if prior_file.is_file():
        d = np.load(prior_file)
        verbs = [str(x) for x in d["verb_keys"]]
        nouns = [str(x) for x in d["noun_keys"]]
        prior = d["prior"].astype(np.float32)
        widths = d["verb_widths"].astype(np.float32)
        hp = tuple(float(x) for x in d["hyperparams"][:5])
        return verbs, nouns, prior, widths, hp
    verbs = [f"verb_{i}" for i in range(97)]
    nouns = [f"noun_{i}" for i in range(300)]
    return verbs, nouns, np.ones((97, 300), dtype=np.float32) / 300.0, np.full(97, 2.8, dtype=np.float32), (0.55, 0.45, 0.25, 1.0, 1.0)


def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    scaled = logits / max(1e-3, float(temp))
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-8)


def compute_soft_actionness_gate(
    verb_probs: np.ndarray,
    motion_1hz: np.ndarray,
    drift_1hz: np.ndarray,
    windows: np.ndarray,
    gate_floor: float = 0.30,
) -> np.ndarray:
    n_w = len(windows)
    if n_w == 0:
        return np.zeros(0, dtype=np.float32)
    max_v = np.max(verb_probs, axis=-1)
    entropy = -np.sum(verb_probs * np.log(verb_probs + 1e-8), axis=-1) / np.log(max(2, verb_probs.shape[1]))
    conf = max_v * (1.0 - 0.5 * entropy)

    mot_norm = motion_1hz / (np.percentile(motion_1hz, 90) + 1e-5) if len(motion_1hz) > 0 else np.ones(1)
    drf_norm = drift_1hz / (np.percentile(drift_1hz, 90) + 1e-5) if len(drift_1hz) > 0 else np.ones(1)

    raw_gate = np.zeros(n_w, dtype=np.float32)
    for i, (ws, we) in enumerate(windows):
        s_i = int(max(0, min(len(mot_norm) - 1, ws)))
        e_i = int(max(s_i + 1, min(len(mot_norm), we)))
        m_val = float(np.clip(np.mean(mot_norm[s_i:e_i]), 0.0, 1.5))
        d_val = float(np.clip(np.mean(drf_norm[s_i:e_i]), 0.0, 1.5))
        raw_gate[i] = float(np.clip(0.50 * conf[i] + 0.35 * m_val + 0.15 * d_val, 0.0, 1.0))

    smoothed = np.copy(raw_gate)
    for i in range(1, n_w):
        smoothed[i] = 0.65 * smoothed[i] + 0.35 * smoothed[i - 1]
    for i in range(n_w - 2, -1, -1):
        smoothed[i] = 0.65 * smoothed[i] + 0.35 * smoothed[i + 1]
    return np.clip(gate_floor + (1.0 - gate_floor) * smoothed, gate_floor, 1.0)


def temporal_soft_nms(instances: List[Dict], sigma: float = 0.40, min_score: float = 0.01) -> List[Dict]:
    by_pair: Dict[Tuple[str, str], List[Dict]] = {}
    for inst in instances:
        by_pair.setdefault((inst["verb"], inst["noun"]), []).append(dict(inst))

    kept: List[Dict] = []
    for _, group in by_pair.items():
        pool = sorted(group, key=lambda x: x["score"], reverse=True)
        out_group = []
        while pool:
            best = pool.pop(0)
            out_group.append(best)
            bs, be = best["segment"]
            blen = max(1e-4, be - bs)
            new_pool = []
            for item in pool:
                s, e = item["segment"]
                inter = max(0.0, min(be, e) - max(bs, s))
                if inter > 0.0:
                    tiou = inter / max(1e-4, blen + (e - s) - inter)
                    item["score"] *= float(np.exp(-(tiou * tiou) / sigma))
                if item["score"] >= min_score:
                    new_pool.append(item)
            pool = sorted(new_pool, key=lambda x: x["score"], reverse=True)
        kept.extend(out_group)

    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept


def decode_epic_actions(
    cache: Dict[str, np.ndarray],
    weights_dir: Optional[Path] = None,
    multi_width: bool = False,
    top_k_per_window: int = 5,
    max_actions: int = 2000,
) -> List[Dict]:
    verbs_list, nouns_list, prior_mat, verb_widths, hp = load_epic_vocab_and_priors(weights_dir)
    alpha, beta, gamma, temp_v, temp_n = hp
    n_v, n_n = len(verbs_list), len(nouns_list)
    dur = float(cache["duration_sec"][0])

    log_prior = gamma * np.log(prior_mat + 1e-4)
    motion_1hz = cache.get("motion_1hz", np.ones(int(max(1, dur)), dtype=np.float32))
    drift_1hz = cache.get("drift_1hz", np.zeros(int(max(1, dur)), dtype=np.float32))
    raw_instances: List[Dict] = []
    prior_file = (Path(weights_dir) / "epic_priors.npz") if weights_dir else Path("weights/epic_priors.npz")
    action_verbs = action_nouns = None
    if prior_file.is_file():
        with np.load(prior_file, allow_pickle=False) as data:
            if "action_verb" in data and "action_noun" in data:
                action_verbs = data["action_verb"].astype(np.int32)
                action_nouns = data["action_noun"].astype(np.int32)

    for scale_tag, default_w in [("2s", 2.2), ("4s", 3.8)]:
        win_key = f"windows_{scale_tag}"
        if win_key not in cache:
            continue
        windows = cache[win_key]
        pair_curves: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        joint_key = f"joint_logits_{scale_tag}"
        if (
            joint_key in cache
            and action_verbs is not None
            and action_nouns is not None
            and cache[joint_key].shape[1] == len(action_verbs) == len(action_nouns)
        ):
            action_probs = _softmax(cache[joint_key])
            v_probs = np.zeros((len(windows), n_v), dtype=np.float32)
            for action_idx, verb_idx in enumerate(action_verbs):
                if 0 <= verb_idx < n_v:
                    v_probs[:, verb_idx] += action_probs[:, action_idx]
            gate = compute_soft_actionness_gate(
                v_probs, motion_1hz, drift_1hz, windows, gate_floor=0.30
            )

            k = min(max(1, top_k_per_window), action_probs.shape[1])
            top_actions = np.argpartition(
                action_probs, -k, axis=1
            )[:, -k:]
            for w_idx, (ws, we) in enumerate(windows):
                center = 0.5 * float(ws + we)
                order = top_actions[w_idx][
                    np.argsort(action_probs[w_idx, top_actions[w_idx]])[::-1]
                ]
                for action_idx in order:
                    vi = int(action_verbs[action_idx])
                    ni = int(action_nouns[action_idx])
                    if not (0 <= vi < n_v and 0 <= ni < n_n):
                        continue
                    score = float(action_probs[w_idx, action_idx]) * float(gate[w_idx])
                    if score > 0.0:
                        pair_curves.setdefault((vi, ni), []).append((center, score))
        else:
            v_probs = _softmax(cache[f"verb_logits_{scale_tag}"][:, :n_v], temp=temp_v)
            n_probs = _softmax(cache[f"noun_logits_{scale_tag}"][:, :n_n], temp=temp_n)

            noun_census = np.mean(n_probs, axis=0, keepdims=True)
            n_probs = n_probs * np.power(noun_census / (np.mean(noun_census) + 1e-8), 0.25)
            n_probs = n_probs / (np.sum(n_probs, axis=-1, keepdims=True) + 1e-8)

            gate = compute_soft_actionness_gate(
                v_probs, motion_1hz, drift_1hz, windows, gate_floor=0.30
            )
            for w_idx, (ws, we) in enumerate(windows):
                center = 0.5 * float(ws + we)
                top_v = np.argsort(v_probs[w_idx])[-3:]
                top_n = np.argsort(n_probs[w_idx])[-3:]
                cands = []
                for vi in top_v:
                    for ni in top_n:
                        log_sc = (
                            alpha * np.log(v_probs[w_idx, vi] + 1e-8)
                            + beta * np.log(n_probs[w_idx, ni] + 1e-8)
                            + log_prior[vi, ni]
                        )
                        prob = float(np.exp(np.clip(log_sc, -20.0, 0.0))) * float(gate[w_idx])
                        cands.append((prob, int(vi), int(ni)))
                cands.sort(reverse=True)
                for prob, vi, ni in cands[:top_k_per_window]:
                    pair_curves.setdefault((vi, ni), []).append((center, prob))

        for (vi, ni), pts in pair_curves.items():
            pts.sort(key=lambda x: x[0])
            times = [p[0] for p in pts]
            scores = [p[1] for p in pts]
            base_w = float(0.5 * verb_widths[vi] + 0.5 * default_w)
            widths = [0.7 * base_w, 1.5 * base_w] if multi_width else [base_w]

            for idx_p, (tc, sc) in enumerate(zip(times, scores)):
                if (idx_p > 0 and sc < scores[idx_p - 1]) or (idx_p < len(scores) - 1 and sc < scores[idx_p + 1]):
                    continue
                for w_mult_idx, width in enumerate(widths):
                    s = max(0.0, tc - 0.5 * width)
                    e = min(dur, tc + 0.5 * width)
                    if e - s < 0.25:
                        continue
                    raw_instances.append({
                        "segment": [round(s, 2), round(e, 2)],
                        "verb": verbs_list[vi],
                        "noun": nouns_list[ni],
                        "score": float(sc * (0.92 if w_mult_idx > 0 else 1.0)),
                    })

    deduped = temporal_soft_nms(raw_instances, sigma=0.40, min_score=1e-4)[:max_actions]
    for item in deduped:
        item["score"] = round(float(np.clip(item["score"], 0.0, 1.0)), 4)
    return deduped
