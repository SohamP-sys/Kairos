# Model weights and caches

Large checkpoints and per-video caches are intentionally excluded from Git.
By default, the project looks under this `weights/` directory. Set
`KAIROS_WEIGHTS_DIR` to use another local directory; the Windows setup and
streaming scripts use a separate work-drive directory for these files.

## Required or optional files

- `epic_priors.npz`: EPIC action vocabulary mapping and verb/noun priors used by
  the AVION action decoder. Required for EPIC action decoding and streaming-run
  preflight.
- `avion/avion_finetune_cls_lavila_vitb_best.pt`: AVION ViT-B checkpoint for
  EPIC-KITCHENS. Download with `tools/download_avion.py`; the script checks the
  published MD5 prefix before accepting the checkpoint.
- `siglip/`: local SigLIP model snapshot used for FPVSum frame/category
  semantics.
- `videomae_kinetics/`: local VideoMAE Kinetics checkpoint used by the FPVSum
  path.
- `epic_cache/<video_id>.npz`: optional AVION per-video window cache. The
  extractor evaluates 2-second and 4-second clips at a 1-second stride and
  stores timestamps plus verb/noun log-marginal scores. A missing cache is
  generated on demand; the streaming runner removes a cache it created after
  successfully publishing that video's prediction.

The AVION source tree is separate from the weights directory and defaults to
`third_party/AVION`. Set `KAIROS_AVION_ROOT` to override it. On Windows,
`tools/setup_windows.ps1` clones the source and prepares the optional
FlashAttention imports for inference without FlashAttention.

Checkpoints are loaded locally during inference; install/download them during
setup, not in the offline prediction run. The project still has heuristic
fallback paths when optional models or inference stages are unavailable, but
fallback outputs are not model predictions and must not be used as competition
results.
