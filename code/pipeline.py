"""Video entry pipeline. Optional model/cache stages fail closed to safe heuristics."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from assemble import assemble_prediction
from baseline import (
    evenly_spaced_windows,
    extract_motion_curve,
    probe_duration,
    select_motion_windows,
)
from fpv_select import importance_curve, select_highlights
from preprocess import decode_video_frames, extract_audio_1hz_features
from router_siglip import SigLIPRouter
from schema import validate_output

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("KAIROS_WEIGHTS_DIR", ROOT / "weights")).expanduser()


def _load_epic_cache(video_id: str):
    path = WEIGHTS / "epic_cache" / (video_id + ".npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def _create_epic_cache(video_path: Path, video_id: str) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.extract_avion_epic import extract_avion_cache

    try:
        batch_size = int(os.environ.get("KAIROS_AVION_BATCH_SIZE", "8"))
    except ValueError as exc:
        raise ValueError("KAIROS_AVION_BATCH_SIZE must be a positive integer.") from exc
    if batch_size < 1:
        raise ValueError("KAIROS_AVION_BATCH_SIZE must be a positive integer.")

    cache_path = WEIGHTS / "epic_cache" / (video_id + ".npz")
    extract_avion_cache(
        video_path,
        cache_path,
        checkpoint_path=WEIGHTS / "avion" / "avion_finetune_cls_lavila_vitb_best.pt",
        batch_size=batch_size,
    )


def process_video(
    video_path: Path,
    team_id: str = "TEAM_ID",
    output_path: Path | None = None,
    dataset: str | None = None,
    manifest_duration: float | None = None,
) -> dict:
    video_path = Path(video_path)
    video_id = video_path.stem
    actual_duration = probe_duration(video_path)
    if manifest_duration is not None:
        manifest_duration = float(manifest_duration)
        if (
            not math.isfinite(manifest_duration)
            or manifest_duration <= 0
            or abs(actual_duration - manifest_duration) > 0.5
        ):
            raise ValueError(
                f"Video duration {actual_duration:.2f}s does not match manifest "
                f"duration {manifest_duration:.2f}s within 0.5s."
            )
    duration = manifest_duration if manifest_duration is not None else actual_duration
    dataset_name = (dataset or "").strip().lower()
    is_epic = dataset_name in {"epic", "epic-kitchens-100", "epic_kitchens_100"}
    is_fpvsum = dataset_name == "fpvsum"

    raw_motion = []
    try:
        raw_motion = extract_motion_curve(video_path)
    except Exception as exc:
        sys.stderr.write(f"[fallback] motion extraction: {exc}\n")

    raw_highlights = (
        select_motion_windows(raw_motion, duration)
        if raw_motion
        else evenly_spaced_windows(duration)
    )

    router = None
    label, label_score = ("cooking", 1.0) if is_epic else ("other", 0.25)
    class_probs = {}
    embeddings = np.zeros((0, 64), dtype=np.float32)
    frames = np.zeros((0, 224, 224, 3), dtype=np.uint8)
    flux = np.zeros(max(1, int(np.ceil(duration))), dtype=np.float32)
    highband = flux.copy()

    if not is_epic:
        router = SigLIPRouter(WEIGHTS)
        try:
            frames = decode_video_frames(
                video_path,
                fps=1.0,
                width=224,
                height=224,
                max_seconds=max(2400.0, duration),
            )
            flux, highband = extract_audio_1hz_features(
                video_path, int(np.ceil(duration))
            )
            label, label_score, class_probs, embeddings = router.route_and_label(
                frames, highband
            )
        except Exception as exc:
            sys.stderr.write(f"[fallback] router/audio: {exc}\n")

    if is_fpvsum and label == "cooking":
        from router_siglip import FPV_CLASSES

        fpv_probs = {name: float(class_probs.get(name, 0.0)) for name in FPV_CLASSES}
        label = max(fpv_probs, key=fpv_probs.get) if max(fpv_probs.values(), default=0.0) > 0 else "other"
        label_score = fpv_probs.get(label, 0.25)
        class_probs = fpv_probs

    raw_actions = []
    if label == "cooking":
        cache = _load_epic_cache(video_id)
        prior_file = WEIGHTS / "epic_priors.npz"
        class_file = ROOT / "supported_classes.json"

        if cache is None:
            try:
                sys.stderr.write(
                    "[AVION] No cache found; extracting windows now. "
                    "This may take several minutes on a long video.\n"
                )
                _create_epic_cache(video_path, video_id)
                cache = _load_epic_cache(video_id)
                if cache is None:
                    raise RuntimeError("Extractor completed without writing a cache")
            except Exception as exc:
                sys.stderr.write(f"[AVION] Extraction failed: {exc}\n")

        if cache is None:
            sys.stderr.write("[AVION] Using the motion fallback.\n")
        elif not prior_file.is_file() or not class_file.is_file():
            sys.stderr.write(
                "[info] EPIC cache found but class/prior files are missing; "
                "action decoding is disabled.\n"
            )
        else:
            try:
                from epic_actions import decode_epic_actions
                from epic_select import select_epic_highlights

                raw_actions = decode_epic_actions(cache, WEIGHTS, multi_width=False)
                raw_highlights = select_epic_highlights(
                    raw_actions, duration, budget_ratio=0.145, mode="noisy_or"
                )
            except Exception as exc:
                sys.stderr.write(f"[fallback] EPIC cached-logit stage: {exc}\n")
    else:
        try:
            from fpv_pipeline import run_fpv_branch

            label, label_score, raw_highlights, _, fpv_meta = run_fpv_branch(
                video_path,
                duration,
                router=router,
                frames=frames,
                audio_flux=flux,
                highband=highband,
                motion=raw_motion,
                embeddings=embeddings,
                sig_label=label,
                sig_score=label_score,
                sig_probs=class_probs,
            )
            sys.stderr.write(
                f"[FPV] VideoMAE={fpv_meta['videomae_enabled']} "
                f"shots={fpv_meta['num_shots']} label={label}\n"
            )
        except Exception as exc:
            sys.stderr.write(f"[fallback] FPV model/segmentation branch: {exc}\n")
            try:
                prompt_curve = (
                    router.category_prompt_curve(embeddings, label)
                    if len(embeddings)
                    else None
                )
                curve = importance_curve(
                    raw_motion, flux, highband, embeddings, prompt_curve
                )
                raw_highlights = select_highlights(curve, duration, 0.148)
            except Exception as fallback_exc:
                sys.stderr.write(f"[fallback] FPV importance: {fallback_exc}\n")

    result = assemble_prediction(
        team_id,
        video_id,
        duration,
        label,
        label_score,
        raw_actions,
        raw_highlights,
        max_ratio=0.145,
    )
    ok, errors, _ = validate_output(
        result,
        expected_video_id=video_id,
        manifest_duration=manifest_duration,
        dataset=dataset,
    )
    if not ok:
        sys.stderr.write("[validation fallback] " + "; ".join(errors) + "\n")
        fallback_label = "cooking" if is_epic else "other"
        result = assemble_prediction(
            team_id,
            video_id,
            duration,
            fallback_label,
            1.0 if is_epic else 0.2,
            [],
            evenly_spaced_windows(duration),
            max_ratio=0.145,
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
