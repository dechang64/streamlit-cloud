from __future__ import annotations
"""
Reading-FL Resonance Detector

Detects "high-resonance" excerpts — passages that consistently
trigger deep emotional responses across readers.

These are the excerpts that end up on coffee cup sleeves (坐忘·咖).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class ResonanceDetector:
    """
    Detects excerpts with high emotional resonance.

    Resonance score = f(depth, breadth, consistency)

    - Depth: How deeply did individual readers respond?
      (measured by reflection length, emotion intensity, personal connection)

    - Breadth: How many different readers responded?
      (a passage that moves 50 people is more resonant than one that moves 1)

    - Consistency: Do readers respond with similar emotions?
      (high consistency = the passage has a clear emotional identity)

    - Cross-campus: Does it resonate across different campus cultures?
      (cross-campus resonance = universal appeal)
    """

    def __init__(
        self,
        depth_weight: float = 0.30,
        breadth_weight: float = 0.25,
        consistency_weight: float = 0.20,
        cross_campus_weight: float = 0.25,
        min_reflections: int = 3,
    ):
        self.depth_weight = depth_weight
        self.breadth_weight = breadth_weight
        self.consistency_weight = consistency_weight
        self.cross_campus_weight = cross_campus_weight
        self.min_reflections = min_reflections

        # excerpt_id -> accumulated signals
        self.signals: Dict[str, dict] = {}

    def add_reflection(
        self,
        excerpt_id: str,
        reader_id: str,
        campus_id: str,
        depth_score: float,
        emotion_label: str,
        quality_score: float,
    ):
        """Record a reflection's signal for an excerpt."""
        if excerpt_id not in self.signals:
            self.signals[excerpt_id] = {
                "depths": [],
                "reader_ids": set(),
                "campus_ids": set(),
                "emotions": [],
                "qualities": [],
            }

        sig = self.signals[excerpt_id]
        sig["depths"].append(depth_score)
        sig["reader_ids"].add(reader_id)
        sig["campus_ids"].add(campus_id)
        sig["emotions"].append(emotion_label)
        sig["qualities"].append(quality_score)

    def compute_resonance(self, excerpt_id: str) -> Optional[float]:
        """
        Compute resonance score for an excerpt.

        Returns None if not enough reflections yet.
        """
        sig = self.signals.get(excerpt_id)
        if sig is None or len(sig["depths"]) < self.min_reflections:
            return None

        # 1. Depth: average depth score
        depth = np.mean(sig["depths"])

        # 2. Breadth: number of unique readers (log-scaled)
        n_readers = len(sig["reader_ids"])
        breadth = min(1.0, np.log(1 + n_readers) / np.log(20))

        # 3. Consistency: entropy of emotion distribution (lower = more consistent)
        emotion_counts = defaultdict(int)
        for e in sig["emotions"]:
            emotion_counts[e] += 1
        total = sum(emotion_counts.values())
        entropy = -sum(
            (c / total) * np.log(c / total + 1e-10)
            for c in emotion_counts.values()
        )
        max_entropy = np.log(6)  # 6 emotions
        consistency = 1.0 - (entropy / max_entropy)  # Lower entropy = higher consistency

        # 4. Cross-campus: how many different campuses?
        n_campuses = len(sig["campus_ids"])
        cross_campus = min(1.0, n_campuses / 3.0)  # Normalize by max 3 campuses

        # Weighted combination
        resonance = (
            self.depth_weight * depth +
            self.breadth_weight * breadth +
            self.consistency_weight * consistency +
            self.cross_campus_weight * cross_campus
        )

        return round(float(resonance), 4)

    def get_top_resonant(
        self,
        k: int = 10,
        min_reflections: int = 3,
    ) -> List[Tuple[str, float, dict]]:
        """
        Get top-k most resonant excerpts.

        Returns:
            List of (excerpt_id, resonance_score, stats) sorted by resonance.
        """
        results = []
        for excerpt_id in self.signals:
            score = self.compute_resonance(excerpt_id)
            if score is not None:
                sig = self.signals[excerpt_id]
                results.append((
                    excerpt_id,
                    score,
                    {
                        "n_reflections": len(sig["depths"]),
                        "n_readers": len(sig["reader_ids"]),
                        "n_campuses": len(sig["campus_ids"]),
                        "avg_depth": round(np.mean(sig["depths"]), 3),
                        "top_emotion": max(
                            set(sig["emotions"]),
                            key=sig["emotions"].count
                        ),
                    }
                ))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_coffee_sleeve_candidates(
        self,
        k: int = 5,
    ) -> List[Tuple[str, float, dict]]:
        """
        Get candidates for 坐忘·咖 coffee sleeve printing.

        Criteria:
        - High resonance score
        - Cross-campus appeal (resonates in 2+ campuses)
        - Consistent emotional identity (not all over the place)
        - At least 5 reflections
        """
        candidates = self.get_top_resonant(k=k * 3, min_reflections=5)
        return [
            (eid, score, stats)
            for eid, score, stats in candidates
            if stats["n_campuses"] >= 2 and stats["avg_depth"] > 0.4
        ][:k]

    def get_stats(self) -> dict:
        return {
            "n_excerpts_tracked": len(self.signals),
            "n_with_enough_data": sum(
                1 for sig in self.signals.values()
                if len(sig["depths"]) >= self.min_reflections
            ),
        }
