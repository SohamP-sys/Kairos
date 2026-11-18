# Track 4 — Action-Aware Video Summarization

Samsung PRISM XR Hackathon 2026 — project starter/fallback pipeline.

## Status and honesty note
This package is runnable and includes a deterministic motion/audio/visual-feature baseline, optional local SigLIP zero-shot classification, an EPIC cached-logit action decoder, a real AVION ViT-B adapter, a highlight selector, output assembly, and a local validator. **It does not yet include a trained actionness gate, measured EPIC calibration, official FPVSum annotations, or the organizer's official validator.** Model accuracy must not be described as verified until real dataset evaluation is run.

## Run

Requirements: Python 3.10+ (tested target), `ffmpeg` and `ffprobe` available on PATH.

```bash
python -m pip install -r requirements.txt
./run.sh input.mp4 output.json
```

Set `TEAM_ID_STR` before running to use the same non-empty team ID for all submissions:

```bash
export TEAM_ID_STR="YOUR_TEAM_ID"
./run.sh input.mp4 output.json
```

The output JSON has `team_id`, `video_id`, `duration_sec`, `label`, `actions`, and `highlights`. All model paths are local/offline. By default no model download is attempted. To enable the SigLIP model download during setup, set `ALLOW_MODEL_DOWNLOAD=1` before launching (not recommended in the competition's offline inference environment). If a local SigLIP checkpoint exists at `weights/siglip_vit_b_16.pt`, the router tries to load it.

## Architecture

- **Decode / duration:** `ffprobe` for duration; `ffmpeg` grayscale motion extraction.
- **Fallback highlight baseline:** top-motion non-overlapping windows, safe sanitization.
- **Optional FPV route/classifier:** SigLIP text prompt ensemble, if local model and `open_clip` are installed; otherwise conservative `other` label.
- **Branch routing:** a `cooking` route uses the EPIC cached-logit decoder when its AVION cache and class/prior files are present. Until then, it keeps the cooking label and uses the motion fallback instead of being sent through the sports classifier.
- **FPV highlights:** combined motion, audio spectral flux/high-band energy, embedding novelty, and optional category prompt similarity. These are heuristic signals; no annotation-based training is performed.
- **EPIC action decoder:** `epic_actions.py` consumes a per-video `.npz` cache with window timestamps and verb/noun logits. It applies priors, soft actionness gate, peaks, and temporal soft-NMS. `tools/extract_avion_epic.py` produces caches from the real AVION ViT-B checkpoint.
- **EPIC highlights:** instance-aware noisy-OR/max greedy selection in `epic_select.py`.
- **Fallback ladder:** optional stages fail independently; final assembly clamps timestamps, sanitizes labels/actions/highlights, and writes JSON.

## Optional EPIC cache contract
Place an NPZ at `weights/epic_cache/<video_id>.npz`. It must contain:

- `duration_sec` (single numeric value)
- `windows_2s`, `windows_4s` arrays shaped `[N,2]` with start/end seconds
- `verb_logits_2s`, `noun_logits_2s`, `verb_logits_4s`, `noun_logits_4s`
- optional `motion_1hz`, `drift_1hz`

Action class keys and pair priors are loaded from `weights/epic_priors.npz` when available. Without that file, the decoder uses placeholder keys and uniform priors, so decoded action labels are not meaningful. Use the organizer vocabulary before packaging.

## AVION EK100 extraction

When a video is routed to cooking, `./run.sh` automatically extracts and stores a
missing AVION cache at `weights/epic_cache/<video_id>.npz`, then immediately runs
the EPIC decoder. The first run takes longer; later runs reuse the cache. To
precompute a cache manually (optional):

```bash
python tools/extract_avion_epic.py input.mp4 weights/epic_cache/input.npz
```

The extractor loads the MD5-verified checkpoint at
`weights/avion/avion_finetune_cls_lavila_vitb_best.pt`, uniformly samples 16
frames per window, and marginalizes the checkpoint's 3806 joint-action logits
into EPIC verb and noun probabilities. It uses CPU/MPS/CUDA as available and
does not require flash-attention at inference. PyTorch/timm deprecation warnings
during model import are non-fatal; extraction progress is printed separately.

## Disk-conscious data utilities

- `tools/stream_epic.py` is an optional script to sequentially download, resize, and delete EPIC originals. It requires the official `epic_downloader.py`, manifest, and permission to download data. Review its settings and disk threshold before running.
- `tools/download_fpv.py` uses `yt-dlp`; dataset availability, access rules, and licensing apply.

These scripts do not run during inference.


## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use a synthetic video and a corrupt file to exercise fallback behavior. The official organizer validator should still be run on every output before submission.

## Reproducibility and limitations

- Inference is deterministic for the fallback path.
- No video ID, filename, or dataset label is used as a prediction feature. Filename stem is used only for required output `video_id` and cache lookup.
- This baseline is not trained or calibrated on hackathon annotations.
- Small videos under 4.05 s cannot satisfy the minimum highlight length; the validator/scoring special case needs organizer clarification.
- Current highlight selector uses temporal windows, not semantic KTS shots.
- `TEAM_ID` is a placeholder; configure `TEAM_ID_STR` before submission.


## Architecture update (Nov 18, 2026)

The FPV branch is now wired in the code path and includes: optional local VideoMAE Kinetics classifier (`code/videomae_fpv.py`), SigLIP frame semantics, multi-signal importance curve, feature-based KTS-style temporal segmentation (`code/kts.py`), and duration-budget knapsack highlight selection (`code/fpv_pipeline.py`). Model dependencies are listed separately in `requirements-models.txt`; install CUDA-compatible PyTorch first. Place a locally downloaded VideoMAE checkpoint under `weights/videomae_kinetics/` (Hugging Face model folder including config and weights).

**Important status:** this FPV integration has been code-wired and unit-tested only in the no-checkpoint fallback. VideoMAE inference with real weights is not verified. Kinetics-to-FPVSum mapping is conservative and may need dataset evaluation. SigLIP also requires its local checkpoint.

**AVION is now integrated for cache extraction.** The ViT-B checkpoint loads, produces 3806 joint-action logits, marginalizes them to the official 97 verb and 300 noun classes, and passes a synthetic forward test. Real EPIC validation-mAP and coverage benchmarks, calibration, and throughput still remain to be measured.

### Installing optional model packages

Base setup does not install heavyweight model dependencies. For the RTX 3050 machine, first install the PyTorch build matching your CUDA setup using the official PyTorch selector, then from the project root run:

```bash
python -m pip install -r requirements-models.txt
```

VideoMAE local model directory expected: `weights/videomae_kinetics/`. SigLIP local model folder expected: `weights/siglip/`. At inference, the pipeline does not download models.

## Apple Silicon Mac setup (MPS)

This project can be set up on Apple Silicon Macs for local model inference. The Mac can use PyTorch's MPS backend where supported; model files are downloaded explicitly, then inference is offline. For the 8 GB MacBook Air, do not keep multiple large models in memory while experimenting.

```bash
./setup_mac.sh
source .venv/bin/activate
python tools/download_models.py
```

The download helper stores model snapshots under `weights/siglip/` and `weights/videomae_kinetics/`. `run.sh` is still configured for offline inference. VideoMAE maps Kinetics labels conservatively to the FPVSum categories; unmapped classes are ignored rather than guessed.

### AVION status (not falsely marked complete)

`tools/download_avion.py` downloads and MD5-verifies the EK100 fine-tuned ViT-B checkpoint.
`tools/extract_avion_epic.py` runs 2-second and 4-second windows and writes the NPZ cache
consumed by the EPIC branch. The adapter is synthetic-forward tested; real benchmark
accuracy and throughput are still not reported.

## Model availability check

After setup and download, run:

```bash
python tools/environment_check.py
PYTHONPATH=code python -c "from router_siglip import SigLIPRouter; from videomae_fpv import VideoMAEFpvClassifier; print('SigLIP:', SigLIPRouter().enabled); print('VideoMAE:', VideoMAEFpvClassifier().enabled)"
```

This reports whether local models load; it is not an accuracy benchmark.
