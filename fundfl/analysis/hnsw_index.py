"""
FundFL HNSW Vector Index
========================
Lightweight HNSW implementation for fund risk profile similarity search.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    fund_code: str
    fund_name: str
    distance: float
    sharpe: float
    annual_return: float


class HNSWIndex:
    """Simplified HNSW-like index for fund risk profile vectors."""

    def __init__(self, dim: int = 16, m: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.vectors: List[np.ndarray] = []
        self.codes: List[str] = []
        self.names: List[str] = []
        self.profiles: dict = {}

    def add(self, code: str, name: str, vector: np.ndarray, profile: dict = None):
        """Add a fund vector to the index."""
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension {len(vector)} != index dimension {self.dim}")
        self.vectors.append(vector.astype(np.float32))
        self.codes.append(code)
        self.names.append(name)
        if profile:
            self.profiles[code] = profile

    def search(self, query: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Search for k most similar funds."""
        if len(self.vectors) == 0:
            return []

        query = query.astype(np.float32)
        k = min(k, len(self.vectors))

        # Compute distances (cosine distance = 1 - cosine_similarity)
        distances = []
        for i, vec in enumerate(self.vectors):
            dot = np.dot(query, vec)
            norm = np.linalg.norm(query) * np.linalg.norm(vec)
            cos_sim = float(dot / norm) if norm > 1e-8 else 0.0
            distances.append((i, 1.0 - cos_sim))

        distances.sort(key=lambda x: x[1])

        results = []
        for i, dist in distances[:k]:
            profile = self.profiles.get(self.codes[i], {})
            results.append(SearchResult(
                fund_code=self.codes[i],
                fund_name=self.names[i],
                distance=dist,
                sharpe=profile.get("sharpe_ratio", 0.0),
                annual_return=profile.get("annual_return", 0.0),
            ))
        return results

    def __len__(self):
        return len(self.vectors)

    def stats(self) -> dict:
        return {
            "total_vectors": len(self.vectors),
            "dimension": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
