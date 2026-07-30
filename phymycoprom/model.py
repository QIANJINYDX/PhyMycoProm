from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PhyMycoPromConfig:
    """Locked core configuration used for the main PhyMycoProm-TSS model."""

    model_name: str = "PhyMycoProm-TSS"
    sequence_length: int = 300
    tss_index: int = 249
    embedding_dimension: int = 1536
    hidden_dimension: int = 256
    dropout: float = 0.2
    batch_size: int = 512
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip: float = 5.0
    threshold: float = 0.5
    n_splits: int = 5
    split_seed: int = 20260720
    training_seeds: tuple[int, ...] = (13, 42, 20260720)

    @classmethod
    def from_json(cls, path: str | Path) -> "PhyMycoPromConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if "training_seeds" in payload:
            payload["training_seeds"] = tuple(
                int(value) for value in payload["training_seeds"]
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not 0 <= self.tss_index < self.sequence_length:
            raise ValueError("tss_index must lie inside the input sequence")
        if self.embedding_dimension <= 0 or self.hidden_dimension <= 0:
            raise ValueError("embedding and hidden dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not 0 <= self.minimum_learning_rate_ratio <= 1:
            raise ValueError("minimum_learning_rate_ratio must lie in [0, 1]")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least two")
        if not self.training_seeds:
            raise ValueError("at least one training seed is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhyMycoPromHead(nn.Module):
    """The 393,729-parameter classifier trained on frozen E-TSS embeddings."""

    def __init__(
        self,
        embedding_dimension: int = 1536,
        hidden_dimension: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
