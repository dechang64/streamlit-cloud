"""
Configuration loader for Reading-FL.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    embed_dim: int = 128
    hidden_dim: int = 256
    backbone_layers: int = 2
    dropout: float = 0.1
    max_seq_length: int = 256
    vocab_size: int = 10000


@dataclass
class EmotionHeadConfig:
    num_classes: int = 6
    labels: list = field(default_factory=lambda: [
        "moved", "thinking", "resonance", "confused", "disagree", "calm"
    ])


@dataclass
class QualityHeadConfig:
    output_dim: int = 1


@dataclass
class MatchingHeadConfig:
    output_dim: int = 64


@dataclass
class HeadsConfig:
    emotion: EmotionHeadConfig = field(default_factory=EmotionHeadConfig)
    quality: QualityHeadConfig = field(default_factory=QualityHeadConfig)
    matching: MatchingHeadConfig = field(default_factory=MatchingHeadConfig)


@dataclass
class FLConfig:
    num_rounds: int = 20
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    aggregation: str = "task_aware"
    temperature: float = 0.1
    min_clients: int = 2


@dataclass
class HNSWConfig:
    dim: int = 64
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50


@dataclass
class PrototypeConfig:
    n_per_domain: int = 10
    update_threshold: float = 0.3
    min_reflections: int = 5


@dataclass
class AuditConfig:
    enabled: bool = True
    difficulty: int = 4


@dataclass
class CampusConfig:
    id: str = ""
    name: str = ""
    reader_count: int = 30
    style_weight: float = 0.5


@dataclass
class DataConfig:
    num_books: int = 20
    reflections_per_reader: int = 3
    avg_excerpt_length: int = 80
    avg_reflection_length: int = 60
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    federated: FLConfig = field(default_factory=FLConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    campuses: list = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        campuses = []
        for c in raw.get("campuses", []):
            campuses.append(CampusConfig(**c))

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            heads=HeadsConfig(
                emotion=EmotionHeadConfig(**raw.get("heads", {}).get("emotion", {})),
                quality=QualityHeadConfig(**raw.get("heads", {}).get("quality", {})),
                matching=MatchingHeadConfig(**raw.get("heads", {}).get("matching", {})),
            ),
            federated=FLConfig(**raw.get("federated", {})),
            hnsw=HNSWConfig(**raw.get("hnsw", {})),
            prototype=PrototypeConfig(**raw.get("prototype", {})),
            audit=AuditConfig(**raw.get("audit", {})),
            campuses=campuses,
            data=DataConfig(**raw.get("data", {})),
        )

    @classmethod
    def default(cls) -> "Config":
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if config_path.exists():
            return cls.from_yaml(str(config_path))
        return cls()
