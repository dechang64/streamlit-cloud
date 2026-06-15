# ── python/analysis/detector.py ──
"""
YOLOv11 PCB Defect Detector
=============================
Federated defect detection for PCB manufacturing.

Defect types (6 classes):
  - short_circuit: Unintended connection between traces
  - open_circuit: Broken trace or missing connection
  - spurious_copper: Extra copper deposit
  - missing_hole: Drill hole not present
  - spur: Small copper protrusion
  - good: No defect (for normal samples)

FL Integration:
  - Backbone aggregated via FedAvg across factories
  - Detection head stays local (factory-specific defect distributions)
  - Class imbalance handled via weighted loss per client
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class DefectDetection:
    """Single PCB defect detection result."""
    bbox: list[float]
    class_name: str
    class_id: int
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    area: float
    severity: str  # "critical", "major", "minor"

    def to_dict(self) -> dict:
        return asdict(self)


class PCBDefectDetector:
    """YOLOv11-based PCB defect detector."""

    DEFECT_CLASSES = [
        "short_circuit", "open_circuit", "spurious_copper",
        "missing_hole", "spur", "good",
    ]

    SEVERITY_MAP = {
        "short_circuit": "critical",
        "open_circuit": "critical",
        "spurious_copper": "major",
        "missing_hole": "major",
        "spur": "minor",
        "good": "none",
    }

    def __init__(self, model_size: str = "n", device: Optional[str] = None):
        self.model_size = model_size
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(f"yolo11{self.model_size}.pt")

    def train_local(self, images_dir: str, epochs: int = 50, imgsz: int = 640):
        """Client-side local training on factory's PCB data."""
        self._ensure_model()
        results = self.model.train(
            data=f"{images_dir}/data.yaml",
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
        )
        return results

    def detect(self, image) -> list[DefectDetection]:
        """Detect defects in PCB image."""
        self._ensure_model()
        results = self.model(image)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls)]
                detections.append(DefectDetection(
                    bbox=box.xyxy[0].tolist(),
                    class_name=cls_name,
                    class_id=int(box.cls),
                    confidence=float(box.conf),
                    cx=float(box.xywh[0][0]),
                    cy=float(box.xywh[0][1]),
                    width=float(box.xywh[0][2]),
                    height=float(box.xywh[0][3]),
                    area=float(box.xywh[0][2] * box.xywh[0][3]),
                    severity=self.SEVERITY_MAP.get(cls_name, "minor"),
                ))
        return detections

    def export_backbone_weights(self) -> dict:
        """Export backbone weights for FedAvg."""
        self._ensure_model()
        backbone = {}
        for name, param in self.model.model.state_dict().items():
            if any(f"model.{i}." in name for i in range(10)):
                backbone[name] = param.clone()
        return backbone

    def load_backbone_weights(self, state_dict: dict):
        """Load aggregated backbone weights."""
        self._ensure_model()
        current = self.model.model.state_dict()
        for name, param in state_dict.items():
            if name in current:
                current[name] = param
        self.model.model.load_state_dict(current)

    def summary(self, detections: list[DefectDetection]) -> dict:
        if not detections:
            return {"total": 0, "defects": 0, "severity": {}}
        defects = [d for d in detections if d.class_name != "good"]
        severity_counts = {}
        for d in defects:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1
        return {
            "total": len(detections),
            "defects": len(defects),
            "defect_rate": len(defects) / max(len(detections), 1),
            "severity": severity_counts,
            "avg_confidence": float(np.mean([d.confidence for d in detections])),
        }
