# ── python/analysis/segmentor.py ──
"""
SAM2 PCB Defect Segmentor
===========================
Pixel-level segmentation of PCB defects using SAM2.

Pipeline:
  1. YOLO detects defects → bounding boxes
  2. SAM2 uses boxes as prompts → pixel-level masks
  3. Extract defect morphology (area, perimeter, shape)

PCB-specific metrics:
  - Defect area (pixels)
  - Trace width deviation (for open/short circuits)
  - Hole diameter deviation (for missing holes)
  - Copper coverage ratio
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class DefectSegmentation:
    """Segmented PCB defect with morphology."""
    mask: np.ndarray
    area: int
    perimeter: float
    circularity: float
    solidity: float
    aspect_ratio: float
    bbox: list[float]
    centroid: tuple[float, float]
    defect_type: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask")
        return d


class PCBDefectSegmentor:
    """SAM2-based PCB defect segmentor."""

    def __init__(self, checkpoint: str = "sam2_hiera_small.pt"):
        self.predictor = None
        self.checkpoint = checkpoint

    def _ensure_predictor(self):
        if self.predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor(build_sam2(self.checkpoint))

    def segment(self, image: np.ndarray, boxes: list[list[float]],
                defect_types: Optional[list[str]] = None) -> list[DefectSegmentation]:
        self._ensure_predictor()
        self.predictor.set_image(image)

        results = []
        for i, box in enumerate(boxes):
            box_np = np.array(box, dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box_np, multimask_output=False)
            mask = masks[0]
            metrics = self._compute_morphology(mask, box)
            dtype = defect_types[i] if defect_types and i < len(defect_types) else "unknown"
            results.append(DefectSegmentation(mask=mask, defect_type=dtype, **metrics))
        return results

    def _compute_morphology(self, mask: np.ndarray, bbox: list[float]) -> dict:
        area = int(mask.sum())
        if area == 0:
            return {"area": 0, "perimeter": 0, "circularity": 0, "solidity": 0,
                    "aspect_ratio": 1, "bbox": bbox, "centroid": (0, 0)}

        try:
            import cv2
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True) if contours else 0
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
            hull = cv2.convexHull(contours[0]) if contours else None
            hull_area = cv2.contourArea(hull) if hull is not None else area
            solidity = area / hull_area if hull_area > 0 else 0
            aspect_ratio = 1.0
            if contours and len(contours[0]) >= 5:
                ellipse = cv2.fitEllipse(contours[0])
                major, minor = ellipse[1]
                aspect_ratio = major / minor if minor > 0 else 1.0
        except ImportError:
            perimeter = circularity = solidity = aspect_ratio = 0

        ys, xs = np.where(mask)
        centroid = (float(np.mean(xs)), float(np.mean(ys)))
        return {"area": area, "perimeter": float(perimeter), "circularity": float(circularity),
                "solidity": float(solidity), "aspect_ratio": float(aspect_ratio),
                "bbox": bbox, "centroid": centroid}
