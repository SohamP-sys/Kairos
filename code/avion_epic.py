"""AVION ViT-B action recognition adapter for EPIC-KITCHENS windows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np


class AVIONEpicClassifier:
    def __init__(
        self,
        checkpoint: Path | str,
        avion_root: Path | str,
        device: str | None = None,
    ):
        self.checkpoint = Path(checkpoint)
        self.avion_root = Path(avion_root)
        self.device = device
        self.model = None
        self.enabled = False
        self.action_verbs: np.ndarray | None = None
        self.action_nouns: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint.is_file():
            return
        try:
            import torch

            if torch.cuda.is_available():
                default_device = "cuda"
            elif torch.backends.mps.is_available():
                default_device = "mps"
            else:
                default_device = "cpu"
            self.device = self.device or default_device

            sys.path.insert(0, str(self.avion_root))
            from avion.models.transformer import VisionTransformer

            import torch.nn as nn

            vision_model = VisionTransformer(
                224,
                16,
                768,
                12,
                12,
                4,
                num_frames=16,
                patch_dropout=0.0,
                drop_path_rate=0.0,
                use_fast_conv1=True,
                use_flash_attn=False,
            )

            class AVIONActionClassifier(nn.Module):
                def __init__(self, visual):
                    super().__init__()
                    self.visual = visual
                    self.dropout = nn.Dropout(0.5)
                    self.fc_cls = nn.Linear(visual.width, 3806)

                def forward(self, image):
                    image_embed = self.visual(image)
                    if isinstance(image_embed, list):
                        image_embed = image_embed[0]
                    return self.fc_cls(self.dropout(image_embed))

            model = AVIONActionClassifier(vision_model)

            checkpoint = torch.load(
                str(self.checkpoint),
                map_location="cpu",
                weights_only=False,
            )
            state_dict = {
                key.removeprefix("module."): value
                for key, value in checkpoint["state_dict"].items()
            }
            remapped = {}
            for key, value in state_dict.items():
                if ".attn.Wqkv." in key:
                    key = key.replace(
                        ".attn.Wqkv.",
                        ".attn.in_proj_",
                    )
                elif ".mlp.fc1." in key:
                    key = key.replace(".mlp.fc1.", ".mlp.c_fc.")
                elif ".mlp.fc2." in key:
                    key = key.replace(".mlp.fc2.", ".mlp.c_proj.")
                remapped[key] = value
            state_dict = remapped
            filtered = {
                key: value
                for key, value in state_dict.items()
                if key.startswith("visual.")
                or key.startswith("fc_cls.")
            }
            filtered = {
                ("visual." + key.split(".", 1)[1])
                if key.startswith("visual.")
                else key: value
                for key, value in filtered.items()
            }
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            if len(unexpected) > 0:
                raise RuntimeError(
                    f"Unexpected AVION checkpoint tensors: {unexpected[:5]}"
                )
            if missing:
                raise RuntimeError(
                    f"Missing AVION checkpoint tensors: {missing[:5]}"
                )
            model.to(self.device).eval()
            self.model = model
            self.enabled = True
        except Exception as exc:
            print(f"[AVION disabled] {exc}")
            self.model = None
            self.enabled = False

    def load_action_mapping(self, priors_path: Path | str) -> bool:
        import numpy as np

        path = Path(priors_path)
        if not path.is_file():
            return False
        with np.load(path, allow_pickle=False) as data:
            if "action_verb" not in data or "action_noun" not in data:
                return False
            self.action_verbs = data["action_verb"].copy()
            self.action_nouns = data["action_noun"].copy()
        return (
            len(self.action_verbs) == 3806
            and len(self.action_nouns) == 3806
        )

    def predict_joint_logits(
        self,
        clips: np.ndarray,
        batch_size: int = 4,
    ) -> np.ndarray:
        if not self.enabled:
            return np.empty((0, 3806), dtype=np.float32)

        import torch

        clips = np.asarray(clips, dtype=np.float32)
        if clips.ndim != 5 or len(clips) == 0:
            return np.empty((0, 3806), dtype=np.float32)

        mean = np.asarray(
            [108.3272985, 116.7460125, 104.09373615000001],
            dtype=np.float32,
        )
        std = np.asarray(
            [68.5005327, 66.6321579, 70.32316305],
            dtype=np.float32,
        )

        all_logits = []
        with torch.inference_mode():
            for start in range(0, len(clips), batch_size):
                batch = clips[start:start + batch_size]
                tensor = torch.from_numpy(batch).to(self.device)
                tensor = tensor.permute(0, 4, 1, 2, 3)
                tensor = (tensor - torch.from_numpy(mean).view(1, 3, 1, 1, 1).to(self.device))
                tensor = tensor / torch.from_numpy(std).view(1, 3, 1, 1, 1).to(self.device)
                logits = self.model(tensor)
                all_logits.append(logits.detach().cpu().numpy())
        return np.concatenate(all_logits, axis=0).astype(np.float32)

    def marginalize(
        self,
        joint_logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if joint_logits.size == 0:
            return (
                np.empty((0, 97), dtype=np.float32),
                np.empty((0, 300), dtype=np.float32),
            )
        if self.action_verbs is None or self.action_nouns is None:
            raise RuntimeError("AVION action mapping has not been loaded")

        joint = np.asarray(joint_logits, dtype=np.float32)
        joint = joint - np.max(joint, axis=-1, keepdims=True)
        probs = np.exp(joint)
        probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-8)

        verb_probs = np.zeros((len(probs), 97), dtype=np.float32)
        noun_probs = np.zeros((len(probs), 300), dtype=np.float32)
        for action_index, (verb, noun) in enumerate(
            zip(self.action_verbs, self.action_nouns)
        ):
            verb_probs[:, verb] += probs[:, action_index]
            noun_probs[:, noun] += probs[:, action_index]

        verb_probs /= np.sum(verb_probs, axis=-1, keepdims=True) + 1e-8
        noun_probs /= np.sum(noun_probs, axis=-1, keepdims=True) + 1e-8
        return verb_probs, noun_probs
