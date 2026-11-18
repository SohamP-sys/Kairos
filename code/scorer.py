import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_timestamp(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    return float(s)


def load_epic_annotations(csv_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"].strip()
            s = parse_timestamp(row["start_timestamp"])
            e = parse_timestamp(row["stop_timestamp"])
            if e <= s:
                continue
            by_video[vid].append({
                "segment": [s, e],
                "verb": row["verb"].strip().lower(),
                "noun": row["noun"].strip().lower(),
            })
    return dict(by_video)


def temporal_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0.0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0.0 else 0.0


def _compute_ap(recalls: List[float], precisions: List[float]) -> float:
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_action_map(
    preds_by_vid: Dict[str, Dict[str, Any]],
    gts_by_vid: Dict[str, List[Dict[str, Any]]],
    tiou_thresholds: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
) -> Dict[str, float]:
    gt_by_cls: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))
    pred_by_cls: Dict[Tuple[str, str], List[Tuple[float, str, List[float]]]] = defaultdict(list)

    for vid, gt_list in gts_by_vid.items():
        for g in gt_list:
            gt_by_cls[(g["verb"], g["noun"])][vid].append(g["segment"])
        pred_list = preds_by_vid.get(vid, {}).get("actions", [])
        for p in pred_list:
            pred_by_cls[(p["verb"], p["noun"])].append((float(p["score"]), vid, p["segment"]))

    if not gt_by_cls:
        return {"mAP": 0.0}

    per_thresh: Dict[str, float] = {}
    for thresh in tiou_thresholds:
        aps: List[float] = []
        for cls_key, vid_gts in gt_by_cls.items():
            total_gt = sum(len(segs) for segs in vid_gts.values())
            preds = sorted(pred_by_cls.get(cls_key, []), key=lambda x: x[0], reverse=True)
            matched = {vid: [False] * len(segs) for vid, segs in vid_gts.items()}
            tp, fp = [], []
            for _, vid, pseg in preds:
                gsegs = vid_gts.get(vid, [])
                best_iou, best_idx = 0.0, -1
                for idx, gseg in enumerate(gsegs):
                    if not matched[vid][idx]:
                        iou = temporal_iou(pseg, gseg)
                        if iou > best_iou:
                            best_iou, best_idx = iou, idx
                if best_iou >= thresh and best_idx >= 0:
                    matched[vid][best_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            cum_tp, cum_fp = 0, 0
            recs, precs = [], []
            for t, f in zip(tp, fp):
                cum_tp += t
                cum_fp += f
                recs.append(cum_tp / total_gt)
                precs.append(cum_tp / (cum_tp + cum_fp))
            aps.append(_compute_ap(recs, precs))
        per_thresh[f"mAP@{thresh:.1f}"] = sum(aps) / len(aps)

    per_thresh["mAP"] = sum(per_thresh.values()) / len(per_thresh)
    return per_thresh


def compute_action_coverage(
    preds_by_vid: Dict[str, Dict[str, Any]],
    gts_by_vid: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, float]:
    totals = {"any_overlap": 0.0, "min_1s": 0.0, "half_contained": 0.0, "penalized_any": 0.0}
    counted = 0

    for vid, gt_list in gts_by_vid.items():
        if not gt_list:
            continue
        counted += 1
        if vid not in preds_by_vid:
            continue
        pred = preds_by_vid[vid]
        dur = float(pred.get("duration_sec", 0.0))
        hls = [h["segment"] for h in pred.get("highlights", [])]
        hl_len = sum(max(0.0, h[1] - h[0]) for h in hls)
        penalty = min(1.0, (0.15 * dur) / hl_len) if (dur >= 27.0 and hl_len > 0.15 * dur) else 1.0

        distinct_pairs = {(g["verb"], g["noun"]) for g in gt_list}
        cap_any, cap_1s, cap_half = set(), set(), set()

        for g in gt_list:
            pair = (g["verb"], g["noun"])
            gs, ge = g["segment"]
            gdur = max(1e-6, ge - gs)
            ov = sum(max(0.0, min(ge, hs[1]) - max(gs, hs[0])) for hs in hls)
            if ov > 0.0:
                cap_any.add(pair)
            if ov >= min(1.0, 0.5 * gdur):
                cap_1s.add(pair)
            if ov >= 0.5 * gdur:
                cap_half.add(pair)

        n = len(distinct_pairs)
        totals["any_overlap"] += len(cap_any) / n
        totals["min_1s"] += len(cap_1s) / n
        totals["half_contained"] += len(cap_half) / n
        totals["penalized_any"] += (len(cap_any) / n) * penalty
        counted += 1

    if counted == 0:
        return totals
    return {k: v / counted for k, v in totals.items()}


def compute_highlight_f1(pred_highlights: Sequence[Dict[str, Any]], gt_mask: Sequence[int], fps: float = 1.0) -> float:
    n = len(gt_mask)
    if n == 0:
        return 0.0
    pred_mask = [0] * n
    for h in pred_highlights:
        s, e = h["segment"]
        i0 = max(0, int(round(s * fps)))
        i1 = min(n, int(round(e * fps)))
        for i in range(i0, i1):
            pred_mask[i] = 1
    tp = sum(1 for p, g in zip(pred_mask, gt_mask) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred_mask, gt_mask) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred_mask, gt_mask) if p == 0 and g == 1)
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0
