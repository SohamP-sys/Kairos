# Project Status — 18 Nov 2026

## Integrated in the code path
- Video decoding and duration probe via ffmpeg/ffprobe.
- Heuristic motion baseline and fallback ladder.
- Optional SigLIP prompt routing (requires local checkpoint/open_clip).
- Real AVION ViT-B EK100 fine-tuned checkpoint loading and 3806-action marginalization.
- Official EPIC 98-verb/301-noun vocabulary and train-derived joint prior mapping.
- FPV branch now calls: optional VideoMAE Kinetics inference, SigLIP semantics, audio/motion importance curve, KTS-style feature segmentation, and budget knapsack highlight selection.
- Output assembly and in-process schema validation.
- Unit tests cover no-checkpoint fallbacks and synthetic inputs.

## Not yet complete / not verified
- VideoMAE inference on real weights and its Kinetics-to-FPVSum mapping.
- SigLIP inference on real local weights.
- AVION extraction has a working ViT-B loader and synthetic forward pass; real EPIC benchmark coverage and throughput are not yet measured.
- Official supported_classes.json and validate.py have not been included in this build.
- No real dataset benchmark, checkpoint sanity run, calibration, or hardware throughput measurement.

## Build note
The current FPV additions are a real code integration, but this remains an engineering prototype. Do not describe it as the fully complete hackathon system until the missing pretrained-model paths are run and EPIC/AVION is integrated.

## Supplied metric priorities
- EPIC action coverage: 40%.
- EPIC action mAP: 25%.
- FPVSum highlight frame-level F1: 25%.
- FPVSum video label accuracy: 10%.
- EPIC is therefore 65% of the total score and should receive tuning priority.
- The details folder contains scoring definitions, but no empirical project baseline or measured ablation values.
