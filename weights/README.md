# Model weights
No model weights are included in this repository package. Keep local, offline checkpoints here.

- `siglip_vit_b_16.pt`: optional SigLIP checkpoint for zero-shot category scoring.
- `epic_priors.npz`: optional verb-noun prior and calibration metadata expected by `epic_actions.py`.
- `epic_cache/<video_id>.npz`: optional per-video AVION logits cache for action decoding.

The current package is a runnable baseline without these files. Do not claim model stages are active until the corresponding checkpoint/cache has been tested.
