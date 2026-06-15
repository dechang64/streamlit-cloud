from __future__ import annotations
# ── python/analysis/gradcam.py ──
"""
Grad-CAM for PCB Defect Classification
========================================
Visual explanations: "Why did the model flag this as a short circuit?"

Critical for manufacturing:
- Quality engineers need to understand model decisions
- False positive analysis: "The model flagged this normal pad as a defect because..."
- Root cause: "The defect region overlaps with a high-density trace area"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class GradCAM:
    """Grad-CAM for PCB defect classification models."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        if target_layer is None:
            target_layer = self._auto_detect(model)
        self.target_layer = target_layer
        self._register_hooks()

    def _auto_detect(self, model):
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise ValueError("No Conv2d found")

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        out[0, target_class].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def generate_defect_report(
    heatmap: np.ndarray,
    defect_type: str,
    confidence: float,
    severity: str,
    morphology: Optional[dict] = None,
) -> str:
    """Human-readable defect analysis report."""
    lines = [
        f"## PCB Defect Analysis Report",
        f"",
        f"**Defect Type**: {defect_type.replace('_', ' ').title()}",
        f"**Confidence**: {confidence:.1%}",
        f"**Severity**: {severity.upper()}",
        f"",
        f"### Visual Attention",
    ]

    h, w = heatmap.shape
    center = heatmap[h//4:3*h//4, w//4:3*w//4].mean()
    edge = (heatmap.mean() - center * 0.25) / 0.75
    if center > edge:
        lines.append(f"- Model focused on **central defect region**")
    else:
        lines.append(f"- Model focused on **boundary/transition region** (subtle defect)")

    if morphology:
        lines.append(f"")
        lines.append(f"### Defect Morphology")
        lines.append(f"- Area: {morphology.get('area', 0)} px²")
        lines.append(f"- Circularity: {morphology.get('circularity', 0):.2f}")
        lines.append(f"- Solidity: {morphology.get('solidity', 0):.2f}")

    if severity == "critical":
        lines.append(f"")
        lines.append(f"⚠️ **CRITICAL**: Immediate inspection required. This defect may cause board failure.")

    lines.append(f"")
    lines.append(f"*Defect-FL Grad-CAM Analysis*")
    return "\n".join(lines)
