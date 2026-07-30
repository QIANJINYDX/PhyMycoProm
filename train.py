#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phymycoprom.data import (
    load_aligned_embeddings,
    load_benchmark,
    load_or_create_folds,
)
from phymycoprom.model import PhyMycoPromConfig
from phymycoprom.training import (
    binary_metrics,
    fit_standardizer,
    fold_seed,
    predict_probabilities,
    sha256_file,
    state_dict_on_cpu,
    train_one_head,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-embedding PhyMycoProm-TSS head with grouped "
            "five-fold, three-seed out-of-fold evaluation."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--record-map", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/core")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PhyMycoPromConfig.from_json(args.config)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_benchmark(args.data, sequence_length=config.sequence_length)
    embeddings, row_to_unique = load_aligned_embeddings(
        frame,
        args.embeddings,
        args.record_map,
        expected_dimension=config.embedding_dimension,
    )
    folds, fold_source = load_or_create_folds(
        frame,
        assignments_path=args.fold_assignments,
        n_splits=config.n_splits,
        random_state=config.split_seed,
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    seed_probabilities: dict[int, np.ndarray] = {}
    history_rows: list[dict[str, float | int]] = []
    runtime_rows: list[dict[str, object]] = []
    checkpoint_root = output_dir / "checkpoints"

    for root_seed in config.training_seeds:
        oof = np.full(len(frame), np.nan, dtype=np.float32)
        for fold in range(config.n_splits):
            train_indices = np.flatnonzero(folds != fold)
            test_indices = np.flatnonzero(folds == fold)
            mean, scale = fit_standardizer(
                embeddings,
                row_to_unique,
                train_indices,
            )
            model_seed = fold_seed(root_seed, fold)
            model, history, runtime = train_one_head(
                config=config,
                embeddings=embeddings,
                row_to_unique=row_to_unique,
                labels=labels,
                train_indices=train_indices,
                mean=mean,
                scale=scale,
                device=args.device,
                seed=model_seed,
            )
            oof[test_indices] = predict_probabilities(
                model=model,
                config=config,
                embeddings=embeddings,
                row_to_unique=row_to_unique,
                record_indices=test_indices,
                mean=mean,
                scale=scale,
                device=args.device,
            )

            checkpoint_path = checkpoint_root / f"seed_{root_seed}" / f"fold_{fold}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": config.model_name,
                    "root_seed": int(root_seed),
                    "model_seed": int(model_seed),
                    "fold": int(fold),
                    "config": config.to_dict(),
                    "model_state_dict": state_dict_on_cpu(model),
                    "standardizer_mean": mean,
                    "standardizer_scale": scale,
                    "history": history,
                },
                checkpoint_path,
            )
            for row in history:
                history_rows.append(
                    {
                        "root_seed": int(root_seed),
                        "model_seed": int(model_seed),
                        "fold": int(fold),
                        **row,
                    }
                )
            runtime_rows.append(
                {
                    "root_seed": int(root_seed),
                    "fold": int(fold),
                    **runtime,
                    "checkpoint": str(checkpoint_path.relative_to(output_dir)),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[training] root_seed={root_seed} fold={fold} complete",
                flush=True,
            )
        if not np.isfinite(oof).all():
            raise RuntimeError(f"seed {root_seed} does not cover every OOF record")
        seed_probabilities[int(root_seed)] = oof

    ensemble = np.mean(
        np.stack(
            [seed_probabilities[seed] for seed in config.training_seeds],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)
    predictions = (ensemble >= config.threshold).astype(np.int8)
    overall = binary_metrics(labels, ensemble, threshold=config.threshold)
    per_species = {
        species: binary_metrics(
            labels[frame["species"].eq(species).to_numpy()],
            ensemble[frame["species"].eq(species).to_numpy()],
            threshold=config.threshold,
        )
        for species in sorted(frame["species"].unique())
    }
    per_fold = {
        str(fold): binary_metrics(
            labels[folds == fold],
            ensemble[folds == fold],
            threshold=config.threshold,
        )
        for fold in range(config.n_splits)
    }

    output_columns = [
        column
        for column in (
            "sequence_id",
            "species",
            "label",
            "negative_type",
            "cv_group",
        )
        if column in frame.columns
    ]
    oof_frame = frame[output_columns].copy()
    oof_frame["fold"] = folds
    for seed in config.training_seeds:
        oof_frame[f"probability_seed_{seed}"] = seed_probabilities[seed]
    oof_frame["probability"] = ensemble
    oof_frame["prediction"] = predictions
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    pd.DataFrame(runtime_rows).to_csv(
        output_dir / "runtime.csv",
        index=False,
    )
    frame[["sequence_id", "species", "label", "cv_group"]].assign(fold=folds).to_csv(
        output_dir / "fold_assignments.csv", index=False
    )

    metrics = {
        "model": config.model_name,
        "overall": overall,
        "per_species": per_species,
        "per_fold": per_fold,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model_name,
        "config": config.to_dict(),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(str(args.data)),
        "embeddings": str(args.embeddings.resolve()),
        "embeddings_sha256": sha256_file(str(args.embeddings)),
        "record_map": str(args.record_map.resolve()),
        "record_map_sha256": sha256_file(str(args.record_map)),
        "fold_source": fold_source,
        "record_count": int(len(frame)),
        "unique_embedding_count": int(len(embeddings)),
        "overall_metrics": overall,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
