"""Local Hugging Face VideoMAE Kinetics classifier with Mac MPS/CUDA support."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# Shared FPV taxonomy.
# "other" captures activities that do not map to one of the target classes.
FPV_CLASSES = [
    "cooking",
    "bike",
    "bikepolo",
    "boxing",
    "horse",
    "jumping",
    "longboarding",
    "motor",
    "parkour",
    "plane",
    "rockclimbing",
    "scuba",
    "ski",
    "skate",
    "surfing",
    "other",
]


# Map normalized Kinetics labels to hackathon categories.
ALIASES = {
    "cooking": {
        "frying vegetables",
        "tossing salad",
        "scrambling eggs",
        "cooking egg",
        "cooking chicken",
        "cooking sausages",
        "cooking meat",
        "cooking on campfire",
        "cooking",
        "making pizza",
        "making pasta",
        "making sushi",
        "making sandwich",
        "making salad",
        "cutting vegetables",
        "chopping vegetables",
        "chopping onions",
        "chopping garlic",
        "food preparation",
        "preparing food",
        "baking cookies",
        "baking cake",
        "baking bread",
        "frying food",
        "stir frying",
        "stirring",
        "grilling",
        "grilling meat",
        "grilling chicken",
        "grilling fish",
        "making pancakes",
        "making waffles",
        "making omelette",
        "making omelet",
        "making coffee",
        "making tea",
        "making a sandwich",
        "making a salad",
        "making dumplings",
        "making noodles",
        "making soup",
        "making ice cream",
        "making chocolate",
        "making cheese",
        "making butter",
        "making dough",
        "kneading dough",
        "mixing ingredients",
        "mixing batter",
        "peeling potatoes",
        "peeling fruit",
        "washing vegetables",
        "washing fruits",
        "scraping food",
        "slicing meat",
        "slicing vegetables",
        "slicing bread",
        "cutting meat",
        "cutting fruit",
        "cutting cake",
        "decorating cake",
        "icing cake",
        "food carving",
        "food plating",
        "serving food",
        "setting table",
        "eating",
        "drinking",
        "tasting food",
        "opening a bottle",
        "opening a can",
        "pouring milk",
        "pouring juice",
        "pouring water",
        "pouring coffee",
        "pouring beer",
        "pouring wine",
        "pouring something into a cup",
        "pouring something into a bowl",
        "pouring something into a pan",
        "pouring something into a pot",
        "pouring something into a glass",
        "pouring something onto a plate",
        "pouring something onto food",
        "pouring sauce",
        "pouring oil",
        "pouring syrup",
        "pouring batter",
        "pouring ingredients",
        "pouring liquid",
        "pouring",
    },

    "bike": {
        "biking",
        "bicycle riding",
        "mountain biking",
        "cycling",
        "riding a bike",
        "riding mountain bike",
        "bmx",
    },

    "bikepolo": {
        "playing polo",
        "bicycle polo",
        "bike polo",
    },

    "boxing": {
        "boxing",
        "punching bag",
        "punching",
    },

    "horse": {
        "horse riding",
        "horseback riding",
        "riding horse",
        "riding a horse",
    },

    "jumping": {
        "skydiving",
        "bungee jumping",
        "base jumping",
        "cliff diving",
        "jumping into water",
        "paragliding",
    },

    "longboarding": {
        "longboarding",
        "longboard",
        "skateboarding longboard",
    },

    "motor": {
        "motorcycling",
        "motorbike riding",
        "riding motorcycle",
        "riding a motorcycle",
        "riding dirt bike",
        "driving motorcycle",
    },

    "parkour": {
        "parkour",
        "freerunning",
    },

    "plane": {
        "flying airplane",
        "piloting airplane",
        "flying aircraft",
        "airplane piloting",
    },

    "rockclimbing": {
        "rock climbing",
        "climbing a rope",
        "climbing wall",
        "ice climbing",
    },

    "scuba": {
        "scuba diving",
        "underwater diving",
        "diving underwater",
    },

        "ski": {
        "ski",
        "skiing",
        "snow skiing",
        "skiing downhill",
        "skiing not slalom or crosscountry",
        "skiing crosscountry",
        "skiing slalom",
        "ski jumping",
        "cross country skiing",
        "downhill skiing",
        "slalom skiing",
    },

    "skate": {
        "skateboarding",
        "skateboard",
        "street skateboarding",
    },

    "surfing": {
        "surfing",
        "surfing water",
        "surfing on water",
    },
}

def norm_label(s: str) -> str:
    """Normalize labels to lowercase alphanumeric text and collapse whitespace."""
    s = str(s).lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

LABEL_TO_CLASS = {
    norm_label(alias): cls
    for cls, aliases in ALIASES.items()
    for alias in aliases
}


class VideoMAEFpvClassifier:
    def __init__(self, model_path=None, device=None):
        self.model_path = Path(
            model_path
            or Path(__file__).resolve().parents[1]
            / "weights"
            / "videomae_kinetics"
        )

        self.device = device
        self.model = self.processor = None
        self.id2label = {}
        self.enabled = False

        if not self.model_path.exists():
            print(f"[VideoMAE] Model directory not found: {self.model_path}")
            return

        try:
            import torch

            from transformers import (
                VideoMAEImageProcessor,
                VideoMAEForVideoClassification,
            )

            self.device = self.device or (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            print(f"[VideoMAE] Loading model on device: {self.device}")

            self.processor = VideoMAEImageProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )

            self.model = VideoMAEForVideoClassification.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            ).to(self.device).eval()

            self.id2label = {
                int(k): str(v).lower()
                for k, v in self.model.config.id2label.items()
            }

            self.enabled = True
            print("[VideoMAE] Model loaded successfully.")

        except Exception as exc:
            print(f"[VideoMAE disabled] {exc}")
            self.model = self.processor = None
            self.enabled = False

    def predict(self, frames):
        # Every output category exists even if it receives zero probability.
        out = {c: 0.0 for c in FPV_CLASSES}

        if not self.enabled or len(frames) == 0:
            return out

        import torch

        n = int(getattr(self.model.config, "num_frames", 16))

        ids = np.linspace(
            0,
            len(frames) - 1,
            n,
        ).round().astype(int)

        clip = [frames[i] for i in ids]

        inputs = self.processor(
            clip,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(
                logits.float(),
                dim=-1,
            ).cpu().numpy()

        # ======================================================
        # DEBUG 1: RAW KINETICS TOP 10
        # ======================================================

        top_indices = np.argsort(probs)[::-1][:10]

        print("\n" + "=" * 65)
        print("[DEBUG] VideoMAE - RAW KINETICS TOP 10")
        print("=" * 65)

        for rank, i in enumerate(top_indices, 1):
            label = self.id2label.get(int(i), "unknown")
            print(
                f"{rank:02d}. "
                f"{label:<45} "
                f"{probs[i]:.5f}"
            )

        # ======================================================
        # MAP KINETICS PROBABILITIES
        # ======================================================

        mapped_mass = 0.0

        for i, p in enumerate(probs):
            raw_label = self.id2label.get(i, "")
            label = norm_label(raw_label)

            cls = LABEL_TO_CLASS.get(label)

            if cls is not None:
                out[cls] += float(p)
                mapped_mass += float(p)

        # ======================================================
        # PRESERVE UNMAPPED PROBABILITY MASS AS "OTHER"
        # ======================================================

        unmapped_mass = max(
            0.0,
            1.0 - mapped_mass,
        )

        out["other"] = unmapped_mass

        total_mass = sum(out.values())

        # ======================================================
        # DEBUG 2: MASS ACCOUNTING
        # ======================================================

        print("\n[DEBUG] VideoMAE probability mass accounting")
        print("-" * 65)
        print(f"Total raw probability mass: {float(probs.sum()):.5f}")
        print(f"Mapped probability mass:    {mapped_mass:.5f}")
        print(f"Unmapped probability mass:  {unmapped_mass:.5f}")
        print(f"Output probability mass:    {total_mass:.5f}")
        print("Normalization:              NONE")
        print("-" * 65)

        # ======================================================
        # DEBUG 3: MAPPED TOP 5 + OTHER
        # ======================================================

        print("\n[DEBUG] VideoMAE - OUTPUT TOP 6")

        top_mapped = sorted(
            out.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:6]

        for rank, (label, score) in enumerate(
            top_mapped,
            1,
        ):
            print(
                f"{rank:02d}. "
                f"{label:<20} "
                f"{score:.5f}"
            )

        print("=" * 65 + "\n")

        return out
