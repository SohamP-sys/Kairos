"""FPV branch: video category model + multi-modal importance + KTS + budget knapsack."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from preprocess import decode_video_frames, extract_audio_1hz_features
from baseline import extract_motion_curve
from fpv_select import importance_curve, select_highlights
from router_siglip import SigLIPRouter
from videomae_fpv import VideoMAEFpvClassifier, FPV_CLASSES
from kts import kts_segments, knapsack_highlights


def run_fpv_branch(
    video_path: Path,
    duration: float,
    router=None,
    video_model=None,
    frames=None,
    audio_flux=None,
    highband=None,
    motion=None,
    embeddings=None,
    sig_label=None,
    sig_score=None,
    sig_probs=None,
):

    router = router or SigLIPRouter(
        Path(__file__).resolve().parents[1] / "weights"
    )

    weights_dir = Path(
        os.environ.get(
            "KAIROS_WEIGHTS_DIR",
            Path(__file__).resolve().parents[1] / "weights",
        )
    ).expanduser()
    video_model = video_model or VideoMAEFpvClassifier(
        model_path=weights_dir / "videomae_kinetics"
    )

    # Keep frame extraction shared and low-cost; one frame per second at 224 px.
    if frames is None:
        frames = decode_video_frames(
            video_path,
            fps=1.0,
            width=224,
            height=224,
            max_seconds=max(2400.0, duration),
        )

    if audio_flux is None or highband is None:
        audio_flux, highband = extract_audio_1hz_features(
            video_path, int(np.ceil(duration))
        )

    # ==========================================================
    # SIGLIP CLASSIFICATION
    # ==========================================================

    if embeddings is None or sig_probs is None or sig_label is None:
        sig_label, sig_score, sig_probs, embeddings = router.route_and_label(
            frames, highband
        )

    print("\n" + "=" * 65)
    print("[DEBUG] SigLIP classification")
    print("=" * 65)
    print(f"Label: {sig_label}")
    print(f"Score: {sig_score:.5f}")

    print("\nSigLIP probabilities (top 10):")

    for cls, score in sorted(
        sig_probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]:
        print(f"{cls:<20} {float(score):.5f}")

    print(f"\nSigLIP probability sum: {sum(sig_probs.values()):.5f}")
    print(f"Embedding shape: {np.asarray(embeddings).shape}")
    print("=" * 65)

    # ==========================================================
    # MOTION
    # ==========================================================

    if motion is None:
        motion = extract_motion_curve(video_path)

    # ==========================================================
    # VIDEOMAE CLASSIFICATION
    # ==========================================================

    mae_probs = video_model.predict(frames)

    # ==========================================================
    # ENSEMBLE CLASSIFICATION
    # ==========================================================

    print("\n" + "=" * 65)
    print("[DEBUG] Ensemble inputs")
    print("=" * 65)

    print(f"SigLIP label:   {sig_label}")
    print(f"SigLIP score:   {sig_score:.5f}")
    print(f"VideoMAE label:  {max(mae_probs, key=mae_probs.get)}")
    print(f"VideoMAE score:  {max(mae_probs.values()):.5f}")

    print(f"\nSigLIP mass:    {sum(sig_probs.values()):.5f}")
    print(f"VideoMAE mass:  {sum(mae_probs.values()):.5f}")

    # Ensemble distributions only after both sources are available.
    # If VideoMAE is absent, retain semantic route.
    if sum(mae_probs.values()) > 0:

        combined = {
            c: (
                0.5 * float(sig_probs.get(c, 0))
                + 0.5 * float(mae_probs.get(c, 0))
            )
            for c in FPV_CLASSES
        }

        label = max(combined, key=combined.get)
        label_score = float(combined[label])

    else:
        label, label_score = sig_label, sig_score
        combined = {}

    print("\n[DEBUG] Ensemble output (top 10):")

    for cls, score in sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]:
        print(f"{cls:<20} {float(score):.5f}")

    print(f"\nFinal ensemble label: {label}")
    print(f"Final ensemble score: {label_score:.5f}")
    print("=" * 65 + "\n")

    # ==========================================================
    # PROMPT SIMILARITY CURVE
    # ==========================================================

    prompt_curve = (
        router.category_prompt_curve(embeddings, label)
        if len(embeddings)
        else None
    )

    # ==========================================================
    # FRAME-LEVEL IMPORTANCE
    # ==========================================================

    curve = importance_curve(
        motion,
        audio_flux,
        highband,
        embeddings,
        prompt_curve,
    )

    # ==========================================================
    # KTS TEMPORAL SEGMENTATION
    # ==========================================================

    if len(embeddings) > 0:
        segments = kts_segments(
            embeddings,
            min_len_sec=4.05,
            max_len_sec=20.0,
            penalty=0.18,
            fps=1.0,
        )
    else:
        segments = kts_segments(
            np.asarray(curve)[:, None],
            min_len_sec=4.05,
            max_len_sec=20.0,
            penalty=0.18,
            fps=1.0,
        )

    # ==========================================================
    # COMPUTE SEGMENT IMPORTANCE
    # ==========================================================

    scores = []

    for s, e in segments:
        a = max(0, int(s))
        b = min(len(curve), int(np.ceil(e)))

        scores.append(
            float(np.mean(curve[a:b])) if b > a else 0.0
        )

    # ==========================================================
    # DEBUG: RANK ALL CANDIDATE SEGMENTS
    # ==========================================================

    ranked = sorted(
        zip(segments, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n" + "=" * 60)
    print("[DEBUG] Segment ranking by importance")
    print("=" * 60)

    for rank, ((s, e), score) in enumerate(ranked, start=1):
        print(
            f"Rank {rank:02d} | "
            f"{s:6.2f}s - {e:6.2f}s | "
            f"score={score:.4f} | "
            f"duration={e-s:.2f}s"
        )

    print("=" * 60)

    # ==========================================================
    # SELECT HIGHLIGHTS UNDER DURATION BUDGET
    # ==========================================================

    highlights = knapsack_highlights(
        segments,
        scores,
        duration,
        budget_ratio=0.148,
        min_len=4.05,
    )
    if not highlights:
        highlights = select_highlights(curve, duration, budget_ratio=0.148)

    # ==========================================================
    # DEBUG: SELECTED HIGHLIGHTS
    # ==========================================================

    print("\n[DEBUG] Selected highlights:")

    if not highlights:
        print("  No segments selected.")

    for h in highlights:
        s, e = h["segment"]

        print(
            f"  {s:.2f}s - {e:.2f}s | "
            f"score={h['score']:.4f} | "
            f"duration={e-s:.2f}s"
        )

    print("=" * 60 + "\n")

    return (
        label,
        label_score,
        highlights,
        curve,
        {
            "siglip_label": sig_label,
            "siglip_score": sig_score,
            "videomae_enabled": video_model.enabled,
            "num_shots": len(segments),
        },
    )
