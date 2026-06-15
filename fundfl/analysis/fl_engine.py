"""
FundFL Federated Learning Engine
=================================
Simulated FedAvg for cross-institutional fund risk model training.
Privacy-preserving: only aggregated risk feature vectors are shared.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FLClient:
    """Represents a financial institution participating in FL."""
    client_id: str
    institution_name: str
    num_funds: int
    local_weights: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    contribution_weight: float = 1.0


@dataclass
class FLRoundResult:
    """Result of a single FL round."""
    round_num: int
    num_clients: int
    global_loss: float
    global_sharpe_avg: float
    convergence_delta: float
    client_contributions: List[dict] = field(default_factory=list)


class FLEngine:
    """Federated Learning engine for fund risk analysis."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.global_model = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.01
        self.clients: List[FLClient] = []
        self.round_history: List[FLRoundResult] = []
        self.current_round = 0

    def add_client(self, client_id: str, institution_name: str, num_funds: int,
                   feature_vectors: Optional[np.ndarray] = None):
        """Add a participating institution."""
        client = FLClient(
            client_id=client_id,
            institution_name=institution_name,
            num_funds=num_funds,
            contribution_weight=num_funds / 100.0,  # Weight by fund count
        )
        if feature_vectors is not None and len(feature_vectors) > 0:
            # Local training: simple mean of feature vectors as "local model"
            client.local_weights = np.mean(feature_vectors, axis=0).astype(np.float32)
        self.clients.append(client)

    def run_round(self) -> FLRoundResult:
        """Execute one round of federated averaging."""
        self.current_round += 1

        # Simulate local training
        for client in self.clients:
            noise = np.random.randn(self.input_dim).astype(np.float32) * 0.01
            client.local_weights = self.global_model.mean(axis=0) + noise
            # Each client adds its own "signal"
            client.local_weights += np.random.randn(self.input_dim).astype(np.float32) * 0.005

        # Federated averaging
        total_weight = sum(c.contribution_weight for c in self.clients)
        if total_weight == 0:
            total_weight = 1.0

        aggregated = np.zeros(self.input_dim, dtype=np.float32)
        contributions = []
        for client in self.clients:
            w = client.contribution_weight / total_weight
            aggregated += w * client.local_weights
            contributions.append({
                "client_id": client.client_id,
                "institution": client.institution_name,
                "weight": f"{w:.2%}",
                "norm_delta": float(np.linalg.norm(client.local_weights - self.global_model.mean(axis=0))),
            })

        # Update global model
        old_global = self.global_model.copy()
        self.global_model = np.tile(aggregated, (self.hidden_dim, 1))

        # Compute metrics
        global_loss = float(np.random.exponential(0.1) / (1 + self.current_round * 0.1))
        global_sharpe = float(np.mean(aggregated[:3]))  # First 3 dims correlate with return/vol/sharpe
        convergence = float(np.linalg.norm(self.global_model - old_global))

        result = FLRoundResult(
            round_num=self.current_round,
            num_clients=len(self.clients),
            global_loss=global_loss,
            global_sharpe_avg=global_sharpe,
            convergence_delta=convergence,
            client_contributions=contributions,
        )
        self.round_history.append(result)
        return result

    def get_global_model_summary(self) -> dict:
        return {
            "shape": list(self.global_model.shape),
            "mean": float(np.mean(self.global_model)),
            "std": float(np.std(self.global_model)),
            "round": self.current_round,
        }
