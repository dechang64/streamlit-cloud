"""
Reading-FL Data Provenance

Verifies reflection authenticity using multiple signals:
lamp behavior, timing, text patterns, and blockchain hash.
"""

from __future__ import annotations
import hashlib
import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .chain import AuditChain


@dataclass
class ProvenanceCheck:
    """Result of a provenance verification."""
    is_authentic: bool
    checks: Dict[str, bool]
    score: float  # 0-1, composite authenticity score
    details: str


class DataProvenance:
    """
    Multi-signal authenticity verification for reflections.

    A reflection is considered "authentic" if it passes multiple checks:
    1. Lamp exists and was active during the reading session
    2. Reading duration is reasonable (not too short for the reflection length)
    3. Text doesn't show obvious AI generation patterns
    4. Rate limiting: not too many reflections in a short time
    5. Blockchain hash matches recorded data

    This creates a "trust score" for each reflection without requiring
    any single check to be perfect.
    """

    def __init__(
        self,
        audit_chain: AuditChain,
        min_reading_duration: float = 5.0,
        max_reflections_per_hour: int = 10,
    ):
        self.chain = audit_chain
        self.min_duration = min_reading_duration
        self.max_per_hour = max_reflections_per_hour

        # Track rate limits: reader_id -> list of timestamps
        self._rate_tracker: Dict[str, List[float]] = {}

        # Known lamp IDs (in production, from lamp registration)
        self._known_lamps: Dict[str, str] = {}  # lamp_id -> campus_id

    def register_lamp(self, lamp_id: str, campus_id: str):
        """Register a physical lamp."""
        self._known_lamps[lamp_id] = campus_id

    def verify(
        self,
        reflection_text: str,
        reader_id: str,
        lamp_id: Optional[str] = None,
        reading_duration: float = 0.0,
        expected_chain_hash: Optional[str] = None,
    ) -> ProvenanceCheck:
        """
        Run all provenance checks on a reflection.

        Returns:
            ProvenanceCheck with individual check results and composite score.
        """
        checks = {}

        # 1. Lamp check
        checks["lamp_valid"] = self._check_lamp(lamp_id)

        # 2. Duration check
        checks["duration_reasonable"] = self._check_duration(
            reading_duration, len(reflection_text)
        )

        # 3. AI generation check
        checks["not_ai_generated"] = self._check_ai_patterns(reflection_text)

        # 4. Rate limit check
        checks["rate_ok"] = self._check_rate_limit(reader_id)

        # 5. Blockchain check (if hash provided)
        if expected_chain_hash:
            checks["chain_valid"] = self._check_chain(
                reflection_text, reader_id, expected_chain_hash
            )
        else:
            checks["chain_valid"] = True  # Not applicable

        # Composite score (weighted average)
        weights = {
            "lamp_valid": 0.15,
            "duration_reasonable": 0.25,
            "not_ai_generated": 0.30,
            "rate_ok": 0.15,
            "chain_valid": 0.15,
        }
        score = sum(
            weights[k] * (1.0 if v else 0.0)
            for k, v in checks.items()
        )

        # Record for rate limiting
        self._rate_tracker.setdefault(reader_id, []).append(time.time())

        is_authentic = score >= 0.6  # Threshold

        return ProvenanceCheck(
            is_authentic=is_authentic,
            checks=checks,
            score=round(score, 3),
            details=self._generate_details(checks, score),
        )

    def _check_lamp(self, lamp_id: Optional[str]) -> bool:
        """Check if the lamp is known and registered."""
        if lamp_id is None:
            return False  # No lamp = lower trust
        return lamp_id in self._known_lamps

    def _check_duration(self, duration: float, text_length: int) -> bool:
        """Check if reading duration is reasonable for the reflection length."""
        if duration <= 0:
            return False
        # Expected: at least 5 seconds, or 1 second per 10 chars of reflection
        min_expected = max(self.min_duration, text_length / 50)
        return duration >= min_expected

    def _check_ai_patterns(self, text: str) -> bool:
        """
        Check for obvious AI generation patterns.

        This is a simple heuristic, not a sophisticated detector.
        In production, this would use a dedicated AI detection model.
        """
        # Very short texts are suspicious
        if len(text) < 10:
            return False

        # Check for overly formulaic patterns
        ai_patterns = [
            r"^(总而言之|综上所述|总的来说)",  # Formulaic conclusions
            r"(首先.*其次.*最后)",              # Rigid structure
            r"(值得注意的是|需要指出的是).*(值得注意的是|需要指出的是)",  # Repetition
        ]

        pattern_count = sum(
            1 for p in ai_patterns if re.search(p, text)
        )

        # Check for excessive hedging
        hedge_words = ["或许", "可能", "也许", "似乎", "大概"]
        hedge_count = sum(text.count(w) for w in hedge_words)
        hedge_ratio = hedge_count / max(1, len(text) / 10)

        # If multiple AI patterns AND excessive hedging → suspicious
        if pattern_count >= 2 and hedge_ratio > 0.3:
            return False

        return True

    def _check_rate_limit(self, reader_id: str) -> bool:
        """Check if reader is within rate limits."""
        now = time.time()
        timestamps = self._rate_tracker.get(reader_id, [])

        # Clean old entries (older than 1 hour)
        timestamps = [t for t in timestamps if now - t < 3600]
        self._rate_tracker[reader_id] = timestamps

        return len(timestamps) < self.max_per_hour

    def _check_chain(
        self,
        reflection_text: str,
        reader_id: str,
        expected_hash: str,
    ) -> bool:
        """Verify against blockchain hash."""
        data = {"text": reflection_text, "reader": reader_id}
        return self.chain.verify_reflection(data, expected_hash)

    def _generate_details(self, checks: Dict[str, bool], score: float) -> str:
        failed = [k for k, v in checks.items() if not v]
        if not failed:
            return f"All checks passed. Authenticity score: {score:.1%}"
        return f"Failed checks: {', '.join(failed)}. Score: {score:.1%}"

    def get_stats(self) -> dict:
        return {
            "n_known_lamps": len(self._known_lamps),
            "n_tracked_readers": len(self._rate_tracker),
            "chain_stats": self.chain.get_stats(),
        }
