"""Hugging Face SigLIP zero-shot router and frame embedding extractor.

Weights are loaded from a local folder only at inference. Download them separately with
`tools/download_models.py`; no model download is triggered by run.sh.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


FPV_CLASSES = ["bike", "bikepolo", "boxing", "horse", "jumping", "longboarding", "motor", "parkour", "plane", "rockclimbing", "scuba", "ski", "skate", "surfing"]
PROMPTS: Dict[str, List[str]] = {
    "cooking": ["first person view of someone preparing food in a kitchen", "hands chopping vegetables or cooking with pots and pans"],
    "bike": ["first person view of cycling a bicycle on a trail or road", "riding a mountain bike down a trail"],
    "bikepolo": ["bicycle polo match with riders holding mallets", "bike polo players hitting a ball with mallets"],
    "boxing": ["first person view boxing with gloves, sparring or punching a bag", "boxing fight throwing punches"],
    "horse": ["first person view riding a horse with reins and horse ears visible", "horseback riding along a trail"],
    "jumping": ["first person view skydiving or jumping from a high platform", "cliff jumping into water"],
    "longboarding": ["longboard skateboard downhill riding on a road", "riding a longboard carving turns"],
    "motor": ["first person motorcycle or dirt bike riding", "motorcycle racing through turns"],
    "parkour": ["first person parkour freerunning jumping and vaulting over obstacles", "urban parkour on walls and rails"],
    "plane": ["small airplane pilot cockpit view while flying", "first person flying a light aircraft"],
    "rockclimbing": ["first person rock climbing hands gripping holds on a cliff", "climbing a rock wall with ropes"],
    "scuba": ["first person underwater scuba diving with fish and coral", "underwater diving with bubbles and marine life"],
    "ski": ["first person skiing down a snowy mountain", "alpine skiing on a snow slope with ski poles"],
    "skate": ["street skateboarding at a skatepark doing tricks", "skateboard flip tricks and grinds"],
    "surfing": ["first person surfing on an ocean wave", "surfboard riding inside a breaking wave"],
}
ACTION_PROMPTS = {k: p[0] for k, p in PROMPTS.items()}

class SigLIPRouter:
    def __init__(self, weights_dir: Optional[Path] = None, device: Optional[str] = None):
        self.weights_dir = Path(weights_dir or Path(__file__).resolve().parents[1] / "weights")
        self.model_dir = self.weights_dir / "siglip"
        self.device = device
        self.model = self.processor = None
        self.text_features: Dict[str, np.ndarray] = {}
        self.action_prompt_features: Dict[str, np.ndarray] = {}
        self.enabled = False
        self._load()

    def _load(self):
        if not self.model_dir.exists():
            return
        try:
            import torch
            from transformers import SiglipModel, SiglipProcessor
            self.device = self.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            self.model = SiglipModel.from_pretrained(str(self.model_dir), local_files_only=True).to(self.device).eval()
            self.processor = SiglipProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
            self.enabled = True
            with torch.no_grad():
                for cls, prompts in PROMPTS.items():
                    inputs = self.processor(text=prompts, padding="max_length", return_tensors="pt").to(self.device)
                    f = self.model.get_text_features(**inputs); f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
                    avg = f.mean(0); avg = avg / (avg.norm() + 1e-8)
                    self.text_features[cls] = avg.detach().cpu().numpy()
                    one = self.processor(text=[ACTION_PROMPTS[cls]], padding="max_length", return_tensors="pt").to(self.device)
                    af = self.model.get_text_features(**one)[0]; af = af / (af.norm() + 1e-8)
                    self.action_prompt_features[cls] = af.detach().cpu().numpy()
        except Exception as exc:
            print(f"[SigLIP disabled] {exc}"); self.model = self.processor = None; self.enabled = False

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0: return np.zeros((0, 64), dtype=np.float32)
        if self.enabled:
            try:
                import torch
                out=[]
                with torch.no_grad():
                    for j in range(0,len(frames),8):
                        inputs=self.processor(images=list(frames[j:j+8]),return_tensors="pt").to(self.device)
                        f=self.model.get_image_features(**inputs); f=f/(f.norm(dim=-1,keepdim=True)+1e-8)
                        out.append(f.detach().cpu().numpy())
                return np.concatenate(out).astype(np.float32)
            except Exception as exc: print(f"[SigLIP image embedding fallback] {exc}")
        out=[]
        for f in frames:
            a=f.astype(np.float32)/255.0; h,w,_=a.shape
            h4=h-h%4; w4=w-w%4; a=a[:h4,:w4]
            grid=a.reshape(4,h4//4,4,w4//4,3).mean((1,3)).reshape(-1)
            stats=np.concatenate([a.mean((0,1)),a.std((0,1)),np.mean(np.abs(np.diff(a,axis=0)),axis=(0,1)),np.mean(np.abs(np.diff(a,axis=1)),axis=(0,1))])
            v=np.concatenate([grid,stats]); v=v/(np.linalg.norm(v)+1e-8); out.append(v)
        return np.asarray(out,dtype=np.float32)

    def category_prompt_curve(self, embs: np.ndarray, label_name: str) -> np.ndarray:
        if self.enabled and label_name in self.action_prompt_features and embs.shape[1] == len(self.action_prompt_features[label_name]):
            return (embs @ self.action_prompt_features[label_name]).astype(np.float32)
        return np.zeros(len(embs), dtype=np.float32)

    def route_and_label(self, frames: np.ndarray, audio_highband_1hz=None):
        embs=self.encode_frames(frames)
        if len(embs)==0: return "other",0.0,{"other":1.0},embs
        if self.enabled and self.text_features:
            ids=np.linspace(0,len(embs)-1,min(24,len(embs)),dtype=int); mean=embs[ids].mean(0); mean/=np.linalg.norm(mean)+1e-8
            sims={k:float(mean@v) for k,v in self.text_features.items()}; vals=np.array(list(sims.values()))*12
            ex=np.exp(vals-vals.max()); probs={k:float(v/ex.sum()) for k,v in zip(sims,ex)}
            best=max(probs,key=probs.get); return best,probs[best],probs,embs
        return "other",0.25,{"other":0.25},embs
