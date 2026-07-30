from __future__ import annotations

import hashlib
import random
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .model import PhyMycoPromConfig, PhyMycoPromHead, parameter_count


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fold_seed(root_seed: int, fold: int) -> int:
    """Match the seed derivation used in the locked manuscript experiment."""

    return int(root_seed + fold * 1009 + 500009)


def embedding_batch(
    embeddings: np.ndarray,
    row_to_unique: np.ndarray,
    record_indices: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(
            embeddings[row_to_unique[np.asarray(record_indices, dtype=np.int64)]],
            dtype=np.float32,
        )
    )


def fit_standardizer(
    embeddings: np.ndarray,
    row_to_unique: np.ndarray,
    train_indices: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(embeddings.shape[1])
    total = np.zeros(dimension, dtype=np.float64)
    total_sq = np.zeros(dimension, dtype=np.float64)
    seen = 0
    for start in range(0, len(train_indices), chunk_size):
        batch = embedding_batch(
            embeddings,
            row_to_unique,
            train_indices[start : start + chunk_size],
        )
        total += batch.sum(axis=0, dtype=np.float64)
        total_sq += np.square(batch, dtype=np.float32).sum(
            axis=0,
            dtype=np.float64,
        )
        seen += len(batch)
    if seen == 0:
        raise ValueError("cannot fit a standardizer on an empty training set")
    mean64 = total / seen
    variance = np.maximum(total_sq / seen - np.square(mean64), 0.0)
    scale64 = np.sqrt(variance)
    scale64[scale64 < 1e-6] = 1.0
    return mean64.astype(np.float32), scale64.astype(np.float32)


def train_one_head(
    *,
    config: PhyMycoPromConfig,
    embeddings: np.ndarray,
    row_to_unique: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: str,
    seed: int,
) -> tuple[PhyMycoPromHead, list[dict[str, float]], dict[str, Any]]:
    resolved_device = resolve_device(device)
    seed_everything(seed)
    model = PhyMycoPromHead(
        config.embedding_dimension,
        config.hidden_dimension,
        config.dropout,
    ).to(resolved_device)
    expected_parameters = (
        config.embedding_dimension * config.hidden_dimension
        + config.hidden_dimension
        + config.hidden_dimension
        + 1
    )
    if parameter_count(model) != expected_parameters:
        raise RuntimeError("unexpected classifier-head parameter count")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * config.minimum_learning_rate_ratio,
    )
    train_labels = labels[train_indices]
    positives = int(np.sum(train_labels == 1))
    negatives = int(np.sum(train_labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("training partition must contain both classes")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negatives / positives,
            dtype=torch.float32,
            device=resolved_device,
        )
    )
    mean_tensor = torch.from_numpy(mean).to(resolved_device)
    scale_tensor = torch.from_numpy(scale).to(resolved_device)
    generator = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        order = generator.permutation(train_indices)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            values = torch.from_numpy(
                embedding_batch(embeddings, row_to_unique, indices)
            ).to(resolved_device)
            values = (values - mean_tensor) / scale_tensor
            targets = torch.from_numpy(labels[indices].astype(np.float32)).to(
                resolved_device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip,
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": loss_sum / max(seen, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

    runtime = {
        "device": str(resolved_device),
        "training_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count(model),
        "seed": int(seed),
    }
    return model, history, runtime


def predict_probabilities(
    *,
    model: PhyMycoPromHead,
    config: PhyMycoPromConfig,
    embeddings: np.ndarray,
    row_to_unique: np.ndarray,
    record_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: str,
) -> np.ndarray:
    resolved_device = resolve_device(device)
    model.eval()
    mean_tensor = torch.from_numpy(mean).to(resolved_device)
    scale_tensor = torch.from_numpy(scale).to(resolved_device)
    probabilities = np.empty(len(record_indices), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(record_indices), config.batch_size):
            indices = record_indices[start : start + config.batch_size]
            values = torch.from_numpy(
                embedding_batch(embeddings, row_to_unique, indices)
            ).to(resolved_device)
            logits = model((values - mean_tensor) / scale_tensor)
            probabilities[start : start + len(indices)] = (
                torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            )
    return probabilities


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or len(labels) == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "n": int(len(labels)),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "threshold": float(threshold),
    }


def state_dict_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}
