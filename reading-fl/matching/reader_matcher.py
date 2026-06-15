from __future__ import annotations
"""
Reading-FL Reader Matcher

Matches readers with similar reading tastes using HNSW vector search.
Supports anonymous matching with double-opt-in.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .hnsw_index import HNSWIndex


class ReaderMatcher:
    """
    HNSW-based reader matching system.

    Each reader has a "reading profile vector" — an aggregated embedding
    of all their reflections and reading behavior. Two readers with similar
    profile vectors have similar reading tastes.

    Matching is anonymous and requires double opt-in:
    - Reader A sees anonymous excerpts from similar readers
    - If A "likes" a reader, that reader sees A's anonymous excerpts
    - Only when both opt in is the connection established

    Privacy:
    - Profile vectors are stored locally per campus (FL)
    - Only embeddings are shared for matching, never raw reflections
    - Reader IDs are hashed, not plaintext
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self.index = HNSWIndex(
            dim=embedding_dim,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        self.embedding_dim = embedding_dim

        # Reader ID -> profile vector (running average)
        self.profiles: Dict[str, np.ndarray] = {}
        # Reader ID -> number of reflections (for running average)
        self.reflection_counts: Dict[str, int] = {}

        # Pending match requests: reader_a -> set of reader_b who liked a
        self.pending_requests: Dict[str, set] = defaultdict(set)
        # Confirmed matches: frozenset({a, b})
        self.confirmed_matches: set = set()

        self._n_matches = 0

    def update_profile(
        self,
        reader_id: str,
        reflection_embedding: np.ndarray,
    ):
        """
        Incrementally update a reader's profile vector.

        Uses running average: new_profile = (old * n + new) / (n + 1)
        """
        if reader_id in self.profiles:
            n = self.reflection_counts[reader_id]
            old_profile = self.profiles[reader_id]
            new_profile = (old_profile * n + reflection_embedding) / (n + 1)
            self.profiles[reader_id] = new_profile
            self.reflection_counts[reader_id] = n + 1
        else:
            self.profiles[reader_id] = reflection_embedding.copy()
            self.reflection_counts[reader_id] = 1

    def rebuild_index(self):
        """Rebuild HNSW index from current profiles."""
        self.index = HNSWIndex(
            dim=self.embedding_dim,
            M=self.index.M,
            ef_construction=self.index.ef_construction,
            ef_search=self.index.ef_search,
        )
        for reader_id, profile in self.profiles.items():
            self.index.add(
                profile,
                metadata={
                    "reader_id": reader_id,
                    "n_reflections": self.reflection_counts[reader_id],
                }
            )

    def find_similar_readers(
        self,
        reader_id: str,
        k: int = 10,
        exclude_confirmed: bool = True,
    ) -> List[Tuple[float, str, dict]]:
        """
        Find readers with similar reading tastes.

        Returns:
            List of (distance, reader_id, metadata) sorted by similarity.
        """
        if reader_id not in self.profiles:
            return []

        query = self.profiles[reader_id]
        results = self.index.search(query, k=k + 20)

        # Filter out self and already confirmed matches
        filtered = []
        for dist, idx, meta in results:
            rid = meta.get("reader_id", "")
            if rid == reader_id:
                continue
            if exclude_confirmed and frozenset({reader_id, rid}) in self.confirmed_matches:
                continue
            filtered.append((dist, rid, meta))

        return filtered[:k]

    def find_similar_readers_cross_campus(
        self,
        reader_id: str,
        campus_id: str,
        k: int = 10,
    ) -> List[Tuple[float, str, dict]]:
        """Find similar readers from OTHER campuses (for FL cross-campus matching)."""
        results = self.find_similar_readers(reader_id, k=k * 3)
        return [
            (d, rid, meta) for d, rid, meta in results
            if meta.get("campus_id") != campus_id
        ][:k]

    def send_match_request(self, from_reader: str, to_reader: str) -> bool:
        """
        Send a match request (Reader A "likes" Reader B).

        If B has already sent a request to A, match is confirmed automatically.
        """
        if frozenset({from_reader, to_reader}) in self.confirmed_matches:
            return False  # Already matched

        if to_reader in self.pending_requests.get(from_reader, set()):
            return False  # Already sent

        # Check if this is a mutual like
        if from_reader in self.pending_requests.get(to_reader, set()):
            # Double opt-in! Confirm match.
            self.confirmed_matches.add(frozenset({from_reader, to_reader}))
            self.pending_requests[to_reader].discard(from_reader)
            self._n_matches += 1
            return True  # Match confirmed!

        # Otherwise, pending
        self.pending_requests[from_reader].add(to_reader)
        return False  # Pending, waiting for other side

    def get_pending_requests(self, reader_id: str) -> List[str]:
        """Get readers who have sent match requests to this reader."""
        return list(self.pending_requests.get(reader_id, set()))

    def get_confirmed_matches(self, reader_id: str) -> List[str]:
        """Get all confirmed matches for a reader."""
        matches = []
        for pair in self.confirmed_matches:
            if reader_id in pair:
                other = (pair - {reader_id}).pop()
                matches.append(other)
        return matches

    def get_stats(self) -> dict:
        return {
            "n_readers": len(self.profiles),
            "n_pending_requests": sum(len(v) for v in self.pending_requests.values()),
            "n_confirmed_matches": self._n_matches,
            "avg_reflections_per_reader": (
                np.mean(list(self.reflection_counts.values()))
                if self.reflection_counts else 0
            ),
        }
