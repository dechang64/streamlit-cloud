"""
FundFL Audit Chain
==================
Immutable audit trail for fund risk computations.
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple


@dataclass
class AuditBlock:
    """Single block in the audit chain."""
    index: int
    timestamp: float
    fund_code: str
    action: str
    data_hash: str
    prev_hash: str
    block_hash: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AuditChain:
    """Simple blockchain-style audit chain for fund risk computations."""

    def __init__(self):
        self.chain: List[AuditBlock] = []
        self._create_genesis()

    def _create_genesis(self):
        genesis = AuditBlock(
            index=0,
            timestamp=time.time(),
            fund_code="GENESIS",
            action="chain_init",
            data_hash=hashlib.sha256(b"genesis").hexdigest(),
            prev_hash="0" * 64,
            block_hash="",
        )
        genesis.block_hash = self._compute_hash(genesis)
        self.chain.append(genesis)

    @staticmethod
    def _compute_hash(block: AuditBlock) -> str:
        data = f"{block.index}{block.timestamp}{block.fund_code}{block.action}{block.data_hash}{block.prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def add_block(self, fund_code: str, action: str, data: dict) -> AuditBlock:
        """Add a new block to the chain."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        prev_block = self.chain[-1]
        block = AuditBlock(
            index=len(self.chain),
            timestamp=time.time(),
            fund_code=fund_code,
            action=action,
            data_hash=data_hash,
            prev_hash=prev_block.block_hash,
            block_hash="",
            metadata={"data_preview": str(data)[:200]},
        )
        block.block_hash = self._compute_hash(block)
        self.chain.append(block)
        return block

    def verify(self) -> Tuple[bool, str]:
        """Verify chain integrity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            # Verify hash
            expected_hash = self._compute_hash(current)
            if current.block_hash != expected_hash:
                return False, f"Block {i}: hash mismatch"

            # Verify chain link
            if current.prev_hash != prev.block_hash:
                return False, f"Block {i}: chain link broken"

        return True, "Chain integrity verified"

    def get_blocks(self, fund_code: Optional[str] = None) -> List[AuditBlock]:
        """Get blocks, optionally filtered by fund code."""
        if fund_code is None:
            return self.chain
        return [b for b in self.chain if b.fund_code == fund_code]

    @property
    def length(self) -> int:
        return len(self.chain)

    def to_dict(self) -> dict:
        return {
            "length": len(self.chain),
            "blocks": [b.to_dict() for b in self.chain],
        }
