from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from phymycoprom.data import load_or_create_folds
from phymycoprom.model import PhyMycoPromConfig, PhyMycoPromHead, parameter_count
from phymycoprom.ntv3 import encode_dna
from phymycoprom.training import (
    fit_standardizer,
    fold_seed,
    predict_probabilities,
    train_one_head,
)


class CoreModelTests(unittest.TestCase):
    def test_locked_head_parameter_count(self) -> None:
        config = PhyMycoPromConfig()
        model = PhyMycoPromHead(
            config.embedding_dimension,
            config.hidden_dimension,
            config.dropout,
        )
        self.assertEqual(parameter_count(model), 393_729)

    def test_ntv3_encoding(self) -> None:
        encoded = encode_dna(["ACGT" * 75])
        self.assertEqual(tuple(encoded.shape), (1, 384))
        self.assertEqual(encoded[0, :4].tolist(), [6, 8, 9, 7])
        self.assertTrue((encoded[0, 300:] == 1).all())

    def test_generated_folds_keep_groups_intact(self) -> None:
        rows = []
        for index in range(40):
            rows.append(
                {
                    "sequence_id": f"seq-{index}",
                    "species": "Ath" if index % 2 == 0 else "Sce",
                    "label": (index // 2) % 2,
                    "cv_group": f"group-{index}",
                }
            )
        frame = pd.DataFrame(rows)
        folds, _ = load_or_create_folds(
            frame,
            assignments_path=None,
            n_splits=5,
            random_state=20260720,
        )
        self.assertEqual(set(folds), set(range(5)))
        self.assertTrue(
            pd.DataFrame({"group": frame["cv_group"], "fold": folds})
            .groupby("group")["fold"]
            .nunique()
            .eq(1)
            .all()
        )

    def test_streaming_standardizer_matches_expanded_rows(self) -> None:
        embeddings = np.asarray(
            [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
            dtype=np.float32,
        )
        row_to_unique = np.asarray([0, 1, 2, 0], dtype=np.int64)
        indices = np.arange(4, dtype=np.int64)
        mean, scale = fit_standardizer(
            embeddings,
            row_to_unique,
            indices,
            chunk_size=2,
        )
        expanded = embeddings[row_to_unique]
        np.testing.assert_allclose(mean, expanded.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(scale, expanded.std(axis=0), rtol=1e-6)

    def test_fold_seed_rule(self) -> None:
        self.assertEqual(fold_seed(13, 0), 500_022)
        self.assertEqual(fold_seed(13, 4), 504_058)

    def test_training_and_prediction_smoke(self) -> None:
        config = PhyMycoPromConfig(
            embedding_dimension=4,
            hidden_dimension=3,
            dropout=0.0,
            batch_size=4,
            epochs=1,
            n_splits=2,
            training_seeds=(7,),
        )
        embeddings = np.random.default_rng(7).normal(size=(12, 4)).astype(np.float32)
        row_to_unique = np.arange(12, dtype=np.int64)
        labels = np.asarray([0, 1] * 6, dtype=np.int64)
        train_indices = np.arange(8, dtype=np.int64)
        test_indices = np.arange(8, 12, dtype=np.int64)
        mean, scale = fit_standardizer(
            embeddings,
            row_to_unique,
            train_indices,
        )
        model, history, runtime = train_one_head(
            config=config,
            embeddings=embeddings,
            row_to_unique=row_to_unique,
            labels=labels,
            train_indices=train_indices,
            mean=mean,
            scale=scale,
            device="cpu",
            seed=7,
        )
        probabilities = predict_probabilities(
            model=model,
            config=config,
            embeddings=embeddings,
            row_to_unique=row_to_unique,
            record_indices=test_indices,
            mean=mean,
            scale=scale,
            device="cpu",
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(runtime["parameter_count"], 19)
        self.assertEqual(probabilities.shape, (4,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


if __name__ == "__main__":
    unittest.main()
