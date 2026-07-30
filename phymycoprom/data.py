from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {
    "sequence_id",
    "sequence",
    "species",
    "label",
    "cv_group",
}


def load_benchmark(
    path: str | Path,
    *,
    sequence_length: int = 300,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("sequence_id must be complete and unique")
    if frame["cv_group"].isna().any():
        raise ValueError("cv_group must be complete")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("label must contain only 0 and 1")
    pattern = rf"[ACGT]{{{sequence_length}}}"
    if not frame["sequence"].astype(str).str.fullmatch(pattern).all():
        raise ValueError(
            f"every sequence must contain exactly {sequence_length} A/C/G/T bases"
        )
    return frame.reset_index(drop=True)


def load_aligned_embeddings(
    frame: pd.DataFrame,
    embeddings_path: str | Path,
    record_map_path: str | Path,
    *,
    expected_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != expected_dimension
        or embeddings.dtype != np.float32
    ):
        raise ValueError(
            "embeddings must be a float32 matrix with shape "
            f"(n_unique, {expected_dimension}); observed {embeddings.shape} "
            f"{embeddings.dtype}"
        )

    record_map = pd.read_csv(record_map_path)
    required = {"sequence_id", "unique_index"}
    missing = required.difference(record_map.columns)
    if missing:
        raise ValueError(f"record map is missing columns: {sorted(missing)}")
    if record_map["sequence_id"].duplicated().any():
        raise ValueError("record map contains duplicate sequence_id values")
    lookup = record_map.set_index("sequence_id")["unique_index"]
    aligned = lookup.reindex(frame["sequence_id"])
    if aligned.isna().any():
        examples = frame.loc[aligned.isna(), "sequence_id"].head().tolist()
        raise ValueError(f"record map does not cover dataset IDs: {examples}")
    row_to_unique = aligned.to_numpy(dtype=np.int64)
    if row_to_unique.min(initial=0) < 0 or row_to_unique.max(initial=-1) >= len(
        embeddings
    ):
        raise ValueError("record map contains out-of-range unique indices")
    return embeddings, row_to_unique


def validate_folds(
    frame: pd.DataFrame,
    folds: np.ndarray,
    *,
    n_splits: int,
) -> None:
    folds = np.asarray(folds, dtype=np.int64)
    if folds.shape != (len(frame),):
        raise ValueError("fold array does not match dataset length")
    if set(np.unique(folds)) != set(range(n_splits)):
        raise ValueError(f"fold values must be 0..{n_splits - 1}")
    group_fold_counts = (
        pd.DataFrame({"cv_group": frame["cv_group"], "fold": folds})
        .groupby("cv_group", observed=True)["fold"]
        .nunique()
    )
    if group_fold_counts.gt(1).any():
        raise ValueError("cv_group leakage detected across folds")
    for fold in range(n_splits):
        labels = frame.loc[folds == fold, "label"]
        if set(labels.unique()) != {0, 1}:
            raise ValueError(f"fold {fold} does not contain both classes")


def load_or_create_folds(
    frame: pd.DataFrame,
    *,
    assignments_path: str | Path | None,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, str]:
    if assignments_path is not None:
        assignments = pd.read_csv(assignments_path)
        required = {"sequence_id", "fold"}
        missing = required.difference(assignments.columns)
        if missing:
            raise ValueError(f"fold assignments are missing columns: {sorted(missing)}")
        if assignments["sequence_id"].duplicated().any():
            raise ValueError("fold assignments contain duplicate sequence IDs")
        lookup = assignments.set_index("sequence_id")["fold"]
        aligned = lookup.reindex(frame["sequence_id"])
        if aligned.isna().any():
            raise ValueError("fold assignments do not cover every dataset record")
        folds = aligned.to_numpy(dtype=np.int64)
        source = str(Path(assignments_path).resolve())
    elif "fold" in frame.columns and frame["fold"].notna().all():
        folds = frame["fold"].to_numpy(dtype=np.int64)
        source = "dataset:fold"
    else:
        strata = frame["species"].astype(str) + "::" + frame["label"].astype(str)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        folds = np.full(len(frame), -1, dtype=np.int64)
        dummy = np.zeros((len(frame), 1), dtype=np.uint8)
        for fold, (_, test_indices) in enumerate(
            splitter.split(dummy, y=strata, groups=frame["cv_group"])
        ):
            folds[test_indices] = fold
        source = f"generated:StratifiedGroupKFold(seed={random_state})"
    validate_folds(frame, folds, n_splits=n_splits)
    return folds, source
