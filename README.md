# Track 4 — Action-Aware Video Summarization

Samsung PRISM XR Hackathon 2026.

## Status

The project has dataset-specific inference paths: AVION ViT-B for EPIC-KITCHENS
and SigLIP/VideoMAE plus temporal highlight selection for FPVSum. It also keeps
heuristic fallback behavior for unavailable models, unreadable media, or failed
inference stages. Real EPIC mAP/coverage and FPVSum F1/label-accuracy benchmarks
have not been reported; do not treat smoke tests as accuracy results.

## Architecture

- **EPIC-KITCHENS-100:** manifest metadata selects the EPIC route. AVION extracts
  action scores over short temporal windows; the EPIC decoder maps these to
  supported verb/noun actions, and the action-aware selector chooses highlights.
- **FPVSum:** SigLIP provides frame-level category semantics and VideoMAE provides
  an optional Kinetics action signal. The FPV pipeline combines model scores and
  temporal features to select non-overlapping highlights within the duration
  budget.
- **Fallbacks:** if a model, cache, or media stage fails, the pipeline can use
  motion/feature-based highlights and safe output assembly instead. These
  fallbacks improve robustness; they are not AVION predictions or evidence of
  model accuracy.
- **Output:** one JSON object with `team_id`, `video_id`, `duration_sec`,
  `label`, `actions`, and `highlights`. Dataset-specific schema and timing rules
  are checked against the manifest.

## AVION windows and cache

`tools/extract_avion_epic.py` decodes frames at 8 fps and extracts 16 frames for
each clip. It evaluates 2-second and 4-second windows at a 1-second stride.
The default cache stores window timestamps and separate verb/noun log-marginal
scores. The decoder also accepts joint-action logits when supplied by another
cache producer, but the current extractor does not write those joint logits.

For one EPIC video, a missing cache is created automatically during inference.
The AVION checkpoint and source tree must be available locally:

- `weights/avion/avion_finetune_cls_lavila_vitb_best.pt`
- `third_party/AVION/`
- `weights/epic_priors.npz`

The checkpoint downloader verifies its published MD5 prefix. AVION can run on
CUDA, MPS, or CPU; Windows setup prepares its optional FlashAttention imports so
the configured non-FlashAttention inference path can load.

## Setup and single-video inference

Python 3.10+ and `ffmpeg`/`ffprobe` are required. On macOS/Linux:

```bash
python -m pip install -r requirements.txt
export TEAM_ID_STR="YOUR_TEAM_ID"
./run.sh input.mp4 output.json
```

On Windows, `run.sh` is not the entry point; use the Python runner after setup:

```powershell
python code\run_video.py input.mp4 output.json --dataset fpvsum
```

For EPIC, pass the dataset and official manifest duration:

```powershell
python code\run_video.py input.mp4 output.json `
  --dataset epic_kitchens_100 --manifest-duration 561.53
```

For EPIC cache extraction, AVION processes eight windows per batch by default
and marginalizes the checkpoint's 3806 joint-action logits into verb and noun
probabilities. If CUDA runs out of memory, it halves the batch size and retries.
Set `KAIROS_AVION_BATCH_SIZE` to a smaller positive integer for a more
conservative batch size.

`TEAM_ID_STR` sets the output team ID. Inference is configured to use local
models; download and prepare checkpoints before running rather than relying on
network access during inference.

## Windows NVIDIA setup

Create the project virtual environment first. Then run the setup script from the
repository root, choosing a work drive with sufficient free space:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1 `
  -WorkDir D:\Kairos-Track4-work -TeamId YOUR_TEAM_ID
```

The script installs CUDA 12.6 PyTorch and model/download dependencies; clones
and prepares AVION; downloads the AVION, SigLIP, and VideoMAE checkpoints; and
checks CUDA access and model forward passes. It places weights and temporary
files under `-WorkDir`. The Python environment remains in `.venv`.

## Run and validate the full manifest

The checked-in [`manifest.csv`](manifest.csv) lists 187 videos (138 EPIC-KITCHENS
and 49 FPVSum). For sources already downloaded and named by manifest ID:

```powershell
python batch.py path\to\videos outputs --team-id YOUR_TEAM_ID
python validate.py --dir outputs
```

Batch inference searches the video directory recursively, enforces dataset
labels, and checks probed duration against the manifest. Missing inputs or
failed inference produce a non-zero batch status. `--fill-missing` writes
metadata-only placeholders for development and is not suitable for submission.
Run the organizer-provided validator on the final submission as well.

To download, infer, validate, and remove source videos incrementally instead of
storing the full EPIC collection, use:

```powershell
python tools\stream_manifest.py manifest.csv `
  --work-dir D:\Kairos-Track4-work --team-id YOUR_TEAM_ID
```

The streaming runner checks required local models, processes manifest entries,
validates each output before publishing it, and removes only its own downloaded
source directory after success. It can prefetch one next video when disk space
allows; a failed stage retains the source for retry. The full run still
downloads every listed video over time. EPIC and FPVSum access and licensing
rules apply.

## Weights, outputs, and limitations

Large checkpoints, raw videos, and generated outputs are intentionally not
stored in the source repository. See [`weights/README.md`](weights/README.md)
for expected local model paths and `KAIROS_WEIGHTS_DIR` behavior. Include the
required checkpoints and the complete validated output set in the final
submission package, then test it offline.

The system has not yet been benchmarked against the official scored metrics.
FPV category mapping and heuristic fallback behavior need evaluation on the
provided development data. The local validator checks structure and manifest
consistency, but does not calculate the competition scores.

## Tests

```bash
python -m unittest discover -s tests -v
```
