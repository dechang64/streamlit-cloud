from __future__ import annotations
# ── python/analysis/feature_extractor.py ──
"""
DINOv2 PCB Feature Extractor
==============================
Self-supervised feature extraction for PCB images.

Use cases:
- PCB similarity search: find boards with similar defect patterns
- Domain shift detection: compare PCBs from different factories
- Zero-shot defect clustering: group similar defects without labels
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional
from torchvision import transforms


class DINOv2PCBExtractor(nn.Module):
    """DINOv2 feature extractor for PCB images."""

    MODEL_DIMS = {"vits14": 384, "base": 768, "large": 1024}

    def __init__(self, model_name: str = "facebook/dinov2-base", device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.variant = model_name.split("/")[-1]
        self.dim = self.MODEL_DIMS.get(self.variant, 768)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_backbone(self):
        if not hasattr(self, "backbone") or self.backbone is None:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(self.model_name)
            self.backbone.eval()
            self.backbone.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._load_backbone()
        outputs = self.backbone(x)
        return outputs.last_hidden_state[:, 0]

    def extract(self, image_path: str) -> np.ndarray:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(self.device)
        feat = self.forward(x)
        return feat.squeeze(0).cpu().numpy()
