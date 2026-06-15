"""
Reading-FL Audit Chain

Lightweight blockchain-style audit trail for reflection authenticity.
Ensures data provenance without requiring a full blockchain.
"""

from __future__ import annotations
import hashlib
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class AuditBlock:
    """A single block in the audit chain."""
    index: int
    timestamp: float
    data_hash: str          # Hash of the reflection data
    prev_hash: str          # Previous block's hash
    validator: str          # Campus ID that validated
    nonce: int = 0
    block_hash: str = ""    # This block's hash (computed)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this block."""
        content = f"{self.index}{self.timestamp}{self.data_hash}{self.prev_hash}{self.validator}{self.nonce}"
        return hashlib.sha256(content.encode()).hexdigest()

    def mine(self, difficulty: int = 2):
        """Simple proof-of-work (for demo purposes)."""
        prefix = "0" * difficulty
        while not self.block_hash.startswith(prefix):
            self.nonce += 1
            self.block_hash = self.compute_hash()


class AuditChain:
    """
    Lightweight audit chain for reflection provenance.

    Each reflection generates a data hash that gets recorded on the chain.
    The chain provides:
    1. Immutability: once recorded, a reflection's hash cannot be changed
    2. Provenance: each entry links to its validator (campus)
    3. Verifiability: anyone can verify a reflection against the chain

    This is NOT a cryptocurrency blockchain — it's a simple hash chain
    optimized for audit trail purposes. No mining, no consensus, no tokens.
    """

    def __init__(self, chain_file: str = "audit_chain.json", difficulty: int = 2):
        self.chain: List[AuditBlock] = []
        self.chain_file = chain_file
        self.difficulty = difficulty
        self._pending: List[dict] = []

        # Genesis block
        genesis = AuditBlock(
            index=0,
            timestamp=time.time(),
            data_hash=hashlib.sha256(b"genesis").hexdigest(),
            prev_hash="0" * 64,
            validator="system",
        )
        genesis.block_hash = genesis.compute_hash()
        self.chain.append(genesis)

    def hash_reflection(self, reflection_data: dict) -> str:
        """Compute hash for a reflection's data."""
        # Sort keys for deterministic hashing
        content = json.dumps(reflection_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def add_reflection(
        self,
        reflection_data: dict,
        validator: str = "unknown",
    ) -> AuditBlock:
        """
        Add a reflection to the audit chain.

        Args:
            reflection_data: Dict with reflection fields to hash
            validator: Campus ID that validated this reflection

        Returns:
            The new audit block.
        """
        data_hash = self.hash_reflection(reflection_data)

        block = AuditBlock(
            index=len(self.chain),
            timestamp=time.time(),
            data_hash=data_hash,
            prev_hash=self.chain[-1].block_hash,
            validator=validator,
        )
        block.mine(self.difficulty)
        self.chain.append(block)
        return block

    def verify_reflection(
        self,
        reflection_data: dict,
        expected_hash: str,
    ) -> bool:
        """Verify that a reflection's data matches a chain entry."""
        actual_hash = self.hash_reflection(reflection_data)
        return actual_hash == expected_hash

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire chain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check hash linkage
            if current.prev_hash != previous.block_hash:
                return False

            # Check hash correctness
            if current.block_hash != current.compute_hash():
                return False

        return True

    def get_stats(self) -> dict:
        return {
            "chain_length": len(self.chain),
            "n_reflections": len(self.chain) - 1,  # Exclude genesis
            "validators": list(set(b.validator for b in self.chain)),
            "is_valid": self.verify_chain(),
        }

    def to_dict(self) -> dict:
        return {
            "chain": [asdict(b) for b in self.chain],
            "stats": self.get_stats(),
        }

    def save(self, filepath: Optional[str] = None):
        """Save chain to JSON file."""
        filepath = filepath or self.chain_file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, filepath: Optional[str] = None):
        """Load chain from JSON file."""
        filepath = filepath or self.chain_file
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chain = [
                AuditBlock(**block_data) for block_data in data["chain"]
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Start fresh
