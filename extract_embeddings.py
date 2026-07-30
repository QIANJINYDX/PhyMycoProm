#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from phymycoprom.data import load_benchmark
from phymycoprom.model import PhyMycoPromConfig
from phymycoprom.ntv3 import (
    encode_dna,
    extract_tss_state,
    load_ntv3,
    resolve_device,
)
from phymycoprom.training import sha256_file


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen NTv3 TSS-anchored embeddings for PhyMycoProm."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Local NTv3 checkpoint path or a compatible Hugging Face model ID.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/etss")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    config = PhyMycoPromConfig.from_json(args.config)
    frame = load_benchmark(args.data, sequence_length=config.sequence_length)
    row_to_unique, unique_sequences = pd.factorize(
        frame["sequence"],
        sort=False,
    )
    unique_sequences = unique_sequences.astype(str).tolist()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    partial_path = output_dir / "embeddings.partial.npy"
    record_map_path = output_dir / "record_map.csv"
    manifest_path = output_dir / "manifest.json"
    artifacts = (embeddings_path, partial_path, record_map_path, manifest_path)
    existing = [path for path in artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output artifacts already exist; use a new directory or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    device = resolve_device(args.device)
    model = load_ntv3(
        args.model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(unique_sequences), config.embedding_dimension),
    )
    for start in range(0, len(unique_sequences), args.batch_size):
        stop = min(start + args.batch_size, len(unique_sequences))
        input_ids = encode_dna(unique_sequences[start:stop]).long().to(device)
        values = extract_tss_state(
            model,
            input_ids,
            tss_index=config.tss_index,
            expected_dimension=config.embedding_dimension,
        )
        output[start:stop] = values.cpu().numpy().astype(np.float32)
        if start == 0 or stop == len(unique_sequences) or stop % 1000 < args.batch_size:
            print(f"[embedding] {stop}/{len(unique_sequences)}", flush=True)
    output.flush()
    del output, model
    os.replace(partial_path, embeddings_path)

    record_map = pd.DataFrame(
        {
            "record_index": np.arange(len(frame), dtype=np.int64),
            "sequence_id": frame["sequence_id"],
            "unique_index": row_to_unique.astype(np.int64),
        }
    )
    record_map.to_csv(record_map_path, index=False)
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(str(args.data)),
        "record_count": int(len(frame)),
        "unique_sequence_count": int(len(unique_sequences)),
        "sequence_length": config.sequence_length,
        "tss_index": config.tss_index,
        "embedding_dimension": config.embedding_dimension,
        "embedding_dtype": "float32",
        "embedding_sha256": sha256_file(str(embeddings_path)),
        "record_map_sha256": sha256_file(str(record_map_path)),
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
