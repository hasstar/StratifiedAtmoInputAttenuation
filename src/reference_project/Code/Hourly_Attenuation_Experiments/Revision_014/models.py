"""Frozen ten-model definitions for local hourly clear-sky attenuation.

This module never reads a target or event column. The caller supplies numeric
arrays prepared using only the current training fold.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "_vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR


MODEL_NAMES = (
    "elastic_net",
    "svr_rbf",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "transformer",
    "gcn",
    "bilstm",
)
NEURAL_MODELS = frozenset(("mlp", "transformer", "gcn", "bilstm"))
SCALED_MODELS = frozenset(("elastic_net", "svr_rbf", *NEURAL_MODELS))
N_BASE = 188
N_LEVELS = 10
N_LEVEL_FEATURES = 16
CPU_THREADS = 2


def candidate_configs(name: str) -> list[dict]:
    """Exactly four predetermined configurations for every candidate."""
    grids = {
        "elastic_net": [
            {"alpha": 0.01, "l1_ratio": 0.25},
            {"alpha": 0.1, "l1_ratio": 0.25},
            {"alpha": 0.01, "l1_ratio": 0.75},
            {"alpha": 0.1, "l1_ratio": 0.75},
        ],
        "svr_rbf": [
            {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
            {"C": 10.0, "epsilon": 0.1, "gamma": "scale"},
            {"C": 10.0, "epsilon": 0.3, "gamma": "scale"},
            {"C": 100.0, "epsilon": 0.3, "gamma": "scale"},
        ],
        "random_forest": [
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 16, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
        ],
        "xgboost": [
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
            {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03},
            {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05},
        ],
        "lightgbm": [
            {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03},
            {"n_estimators": 500, "num_leaves": 15, "learning_rate": 0.05},
            {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.03},
            {"n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05},
        ],
        "catboost": [
            {"iterations": 300, "depth": 4, "learning_rate": 0.03},
            {"iterations": 500, "depth": 4, "learning_rate": 0.05},
            {"iterations": 300, "depth": 6, "learning_rate": 0.03},
            {"iterations": 500, "depth": 6, "learning_rate": 0.05},
        ],
    }
    neural = [
        {"epochs": 60, "lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-4},
        {"epochs": 60, "lr": 3e-4, "dropout": 0.1, "weight_decay": 1e-4},
        {"epochs": 80, "lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-3},
        {"epochs": 80, "lr": 3e-4, "dropout": 0.2, "weight_decay": 1e-3},
    ]
    if name in NEURAL_MODELS:
        return [dict(x) for x in neural]
    return [dict(x) for x in grids[name]]


class FlatMLP(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()


class VerticalProfileNet(nn.Module):
    """One-hour vertical model; removed levels never become new GCN edges."""

    def __init__(self, kind: str, active_positions: tuple[int, ...], dropout: float):
        super().__init__()
        self.kind = kind
        self.active_positions = active_positions
        self.base = nn.Sequential(nn.Linear(N_BASE, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU())
        self.project = nn.Linear(N_LEVEL_FEATURES, 32)
        self.absolute_pressure_embedding = nn.Parameter(torch.zeros(N_LEVELS, 32))
        nn.init.normal_(self.absolute_pressure_embedding, std=0.02)
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        elif kind == "gcn":
            self.gcn1 = nn.Linear(32, 32)
            self.gcn2 = nn.Linear(32, 32)
            n = len(active_positions)
            adjacency = torch.eye(n)
            for i, p in enumerate(active_positions):
                for j, q in enumerate(active_positions):
                    if abs(p - q) == 1:
                        adjacency[i, j] = 1.0
            if n:
                degree = adjacency.sum(dim=1).clamp(min=1).pow(-0.5)
                adjacency = degree[:, None] * adjacency * degree[None, :]
            self.register_buffer("adjacency", adjacency)
        elif kind == "bilstm":
            self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
            self.lstm_projection = nn.Linear(64, 32)
        else:
            raise ValueError(f"Unknown vertical model: {kind}")
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x[:, :N_BASE])
        n = len(self.active_positions)
        if n:
            levels = x[:, N_BASE:].reshape(-1, n, N_LEVEL_FEATURES)
            levels = self.project(levels)
            levels = levels + self.absolute_pressure_embedding[list(self.active_positions)]
            if self.kind == "transformer":
                vertical = self.encoder(levels).mean(dim=1)
            elif self.kind == "gcn":
                h = torch.relu(self.gcn1(torch.matmul(self.adjacency, levels)))
                vertical = torch.relu(self.gcn2(torch.matmul(self.adjacency, h))).mean(dim=1)
            else:
                _, (last_hidden, _) = self.lstm(levels)
                vertical = self.lstm_projection(
                    torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
                )
        else:
            vertical = torch.zeros_like(base)
        return self.head(torch.cat((base, vertical), dim=1)).flatten()


class TorchRegressor:
    def __init__(self, kind: str, config: dict, seed: int,
                 n_features: int, active_positions: tuple[int, ...]):
        self.kind = kind
        self.config = config
        self.seed = seed
        self.n_features = n_features
        self.active_positions = active_positions
        self.net: nn.Module | None = None
        self.target_mean = 0.0
        self.target_scale = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchRegressor":
        torch.set_num_threads(CPU_THREADS)
        torch.manual_seed(self.seed)
        if self.kind == "mlp":
            self.net = FlatMLP(self.n_features, self.config["dropout"])
        else:
            self.net = VerticalProfileNet(
                self.kind, self.active_positions, self.config["dropout"]
            )
        self.target_mean = float(np.mean(y))
        self.target_scale = max(float(np.std(y)), 1e-6)
        xx = torch.as_tensor(np.asarray(x, dtype=np.float32))
        yy = torch.as_tensor(((y - self.target_mean) / self.target_scale).astype(np.float32))
        optimiser = torch.optim.AdamW(
            self.net.parameters(), lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.config["epochs"]):
            order = torch.randperm(len(xx), generator=generator)
            for batch in order.split(128):
                optimiser.zero_grad(set_to_none=True)
                loss = torch.mean((self.net(xx[batch]) - yy[batch]) ** 2)
                loss.backward()
                optimiser.step()
        self.net.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Torch model has not been fitted")
        with torch.no_grad():
            xx = torch.as_tensor(np.asarray(x, dtype=np.float32))
            pred = np.concatenate([
                self.net(chunk).numpy() for chunk in xx.split(512)
            ])
        return pred.astype(float) * self.target_scale + self.target_mean


def make_estimator(name: str, config: dict, seed: int,
                   n_features: int, active_positions: tuple[int, ...]):
    if name == "elastic_net":
        return ElasticNet(**config, max_iter=20000, tol=1e-4, random_state=seed)
    if name == "svr_rbf":
        return SVR(kernel="rbf", **config)
    if name == "random_forest":
        return RandomForestRegressor(**config, n_jobs=CPU_THREADS, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            **config, objective="reg:squarederror", tree_method="hist",
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=CPU_THREADS, random_state=seed, verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            **config, max_depth=-1, min_child_samples=20, reg_lambda=1.0,
            n_jobs=CPU_THREADS, random_state=seed, verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            **config, loss_function="RMSE", thread_count=CPU_THREADS,
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
    if name in NEURAL_MODELS:
        return TorchRegressor(name, config, seed, n_features, active_positions)
    raise ValueError(f"Unknown model: {name}")
