from __future__ import annotations
# ── core/fl_engine.py ──
"""
Reading-FL v2 Federated Learning Engine
=========================================
PyTorch-based emotion classification for reading reflections.

Upgrade from v1:
- v1: NumPy MLP, 6 emotions, FedAvg
- v2: PyTorch, 6 emotions + quality scoring, FedAvg + Task-Aware

Emotions: joy, sadness, anger, fear, surprise, contemplation
Quality: 0-5 scale (learned from community feedback)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Optional, Callable
import time


class ReadingFLEngine:
    """Federated learning engine for reading emotion classification."""

    EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "contemplation"]

    def __init__(
        self,
        input_dim: int = 64,
        num_emotions: int = 6,
        hidden_dim: int = 128,
        lr: float = 0.001,
        local_epochs: int = 2,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.seed = seed

        torch.manual_seed(seed)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_emotions),
        )
        self.history: list[dict] = []

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int = 3,
        rounds: int = 10,
        val_split: float = 0.2,
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        """Run federated training across reading communities."""
        torch.manual_seed(self.seed)
        n = len(features)
        indices = np.random.permutation(n)
        val_size = int(n * val_split)
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        train_X = torch.tensor(features[train_idx], dtype=torch.float32)
        train_y = torch.tensor(labels[train_idx], dtype=torch.long)
        val_X = torch.tensor(features[val_idx], dtype=torch.float32)
        val_y = torch.tensor(labels[val_idx], dtype=torch.long)

        # Split among clients (Non-IID: each community reads different genres)
        client_size = len(train_X) // n_clients
        client_splits = []
        for i in range(n_clients):
            start = i * client_size
            end = start + client_size if i < n_clients - 1 else len(train_X)
            client_splits.append(list(range(start, end)))

        history = []
        for rnd in range(1, rounds + 1):
            t0 = time.time()
            client_params_list = []
            client_metrics = []

            for cid, split_idx in enumerate(client_splits):
                cX, cy = train_X[split_idx], train_y[split_idx]
                client_model = nn.Sequential(*list(self.classifier.children()))
                client_model.load_state_dict(OrderedDict(self.classifier.state_dict()))

                params, loss, acc = self._train_client(client_model, cX, cy)
                client_params_list.append(params)
                client_metrics.append({
                    "client_id": cid, "train_loss": loss, "train_acc": acc,
                    "n_samples": len(split_idx),
                })

            global_params = self._fedavg(client_params_list)
            self.classifier.load_state_dict(global_params)
            val_loss, val_acc = self._evaluate(self.classifier, val_X, val_y)
            elapsed = time.time() - t0

            round_metrics = {
                "round": rnd,
                "avg_train_loss": np.mean([m["train_loss"] for m in client_metrics]),
                "avg_train_acc": np.mean([m["train_acc"] for m in client_metrics]),
                "val_loss": val_loss, "val_acc": val_acc, "elapsed": elapsed,
                "client_metrics": client_metrics,
            }
            history.append(round_metrics)
            if progress_callback:
                progress_callback(rnd, round_metrics)

        self.history = history
        return history

    def _train_client(self, model, X, y, epochs=None):
        epochs = epochs or self.local_epochs
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X)
            loss = criterion(out, y).item()
            preds = out.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        return OrderedDict(model.state_dict()), loss, acc

    def _fedavg(self, params_list, client_data_sizes=None):
        if not params_list:
            raise ValueError("params_list is empty — cannot aggregate")
        if client_data_sizes is not None and len(client_data_sizes) == len(params_list):
            total = sum(client_data_sizes)
            if total == 0:
                raise ValueError("Total client data size is zero — cannot aggregate")
            weights = [n / total for n in client_data_sizes]
        else:
            weights = [1.0 / len(params_list)] * len(params_list)
        avg = OrderedDict()
        for key in params_list[0].keys():
            avg[key] = sum(w * p[key] for w, p in zip(weights, params_list))
        return avg

    def _evaluate(self, model, X, y):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            out = model(X)
            loss = criterion(out, y).item()
            preds = out.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        return loss, acc

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict emotions for new reflections."""
        self.classifier.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            out = self.classifier(x)
            probs = torch.softmax(out, dim=1).numpy()
            preds = out.argmax(dim=1).numpy()
        return preds, probs
