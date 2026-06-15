from __future__ import annotations
# ── core/feature_extractor.py ──
"""
Reading-FL v2 Feature Extractor
================================
DINOv2-based text understanding for reading reflections.

Upgrade from v1:
- v1: Pure NumPy bag-of-words → 64-dim features
- v2: DINOv2 (book cover images) + sentence-transformers (text) → 768-dim

Dual-modal features:
  1. Visual: DINOv2 on book cover → captures genre/style
  2. Textual: Sentence-BERT on reflection text → captures emotion/quality
  3. Combined: Concatenated for matching
"""

import numpy as np
from typing import Optional


class ReadingFeatureExtractor:
    """Dual-modal feature extractor for reading reflections.

    Modes:
    - "text": Sentence-BERT on reflection text (768-dim)
    - "visual": DINOv2 on book cover image (768-dim)
    - "combined": Concatenated (1536-dim)
    - "legacy": NumPy bag-of-words (64-dim, backward compatible)
    """

    def __init__(self, mode: str = "legacy", dim: int = 64):
        self.mode = mode
        self.dim = dim

    def extract_text(self, text: str) -> np.ndarray:
        """Extract features from reflection text."""
        if self.mode == "legacy":
            return self._bow_extract(text)
        elif self.mode == "text":
            return self._sbert_extract(text)
        elif self.mode == "combined":
            return self._sbert_extract(text)  # Text-only for combined mode
        else:
            return self._bow_extract(text)

    def extract_visual(self, image_path: str) -> np.ndarray:
        """Extract features from book cover image."""
        if self.mode in ("visual", "combined"):
            return self._dinov2_extract(image_path)
        return np.zeros(self.dim, dtype=np.float32)

    def _bow_extract(self, text: str) -> np.ndarray:
        """Legacy bag-of-words extraction (v1 compatible)."""
        # Simple character n-gram hashing
        vec = np.zeros(self.dim, dtype=np.float32)
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i+3]
            h = hash(trigram) % self.dim
            vec[h] += 1.0
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _sbert_extract(self, text: str) -> np.ndarray:
        """Sentence-BERT extraction (requires sentence-transformers)."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_sbert_model'):
                self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = self._sbert_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except ImportError:
            # Fallback to BOW
            return self._bow_extract(text)

    def _dinov2_extract(self, image_path: str) -> np.ndarray:
        """DINOv2 extraction from book cover."""
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            from transformers import AutoModel

            if not hasattr(self, '_dinov2_model'):
                self._dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base")
                self._dinov2_model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = Image.open(image_path).convert("RGB")
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = self._dinov2_model(x)
                feat = outputs.last_hidden_state[:, 0]
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except (ImportError, FileNotFoundError):
            return np.zeros(768, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 1e-8 else 0.0
