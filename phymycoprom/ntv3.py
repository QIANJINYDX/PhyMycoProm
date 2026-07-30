from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


ALIGNMENT = 128
PAD_TOKEN_ID = 1
TOKEN_IDS = np.asarray([6, 8, 9, 7], dtype=np.uint8)  # A, C, G, T


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return resolved


def encode_dna(sequences: Sequence[str]) -> torch.Tensor:
    """Encode equal-length A/C/G/T strings with the NTv3 token convention."""

    values = [str(sequence).upper() for sequence in sequences]
    if not values:
        raise ValueError("cannot encode an empty sequence collection")
    lengths = {len(sequence) for sequence in values}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("all sequences in a batch must have one positive length")
    length = next(iter(lengths))
    raw = np.frombuffer("".join(values).encode("ascii"), dtype=np.uint8).reshape(
        len(values), length
    )
    lookup = np.full(256, 255, dtype=np.uint8)
    for index, base in enumerate(b"ACGT"):
        lookup[base] = index
    encoded = lookup[raw]
    if np.any(encoded == 255):
        raise ValueError("NTv3 input contains symbols outside A/C/G/T")

    padded_length = int(math.ceil(length / ALIGNMENT) * ALIGNMENT)
    token_ids = np.full(
        (len(values), padded_length),
        PAD_TOKEN_ID,
        dtype=np.uint8,
    )
    token_ids[:, :length] = TOKEN_IDS[encoded]
    return torch.from_numpy(token_ids)


def load_ntv3(
    model_name_or_path: str | Path,
    *,
    device: torch.device,
    trust_remote_code: bool,
    local_files_only: bool,
) -> torch.nn.Module:
    """Load the NTv3 masked-language-model checkpoint used as a frozen encoder."""

    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        str(model_name_or_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return model.to(device).eval()


def extract_tss_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    tss_index: int,
    expected_dimension: int,
) -> torch.Tensor:
    """Return the final-resolution contextual state aligned to the TSS anchor."""

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=(input_ids != PAD_TOKEN_ID).long(),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("NTv3 did not return hidden states")
    final = hidden_states[-1]
    if final.ndim != 3:
        raise RuntimeError(f"unexpected final hidden-state shape: {tuple(final.shape)}")

    if final.shape[-1] == expected_dimension and final.shape[1] > tss_index:
        nucleotide_states = final
    elif final.shape[1] == expected_dimension and final.shape[2] > tss_index:
        nucleotide_states = final.transpose(1, 2)
    else:
        raise RuntimeError(
            "cannot identify nucleotide and channel axes in final hidden state "
            f"{tuple(final.shape)}"
        )
    values = nucleotide_states[:, tss_index, :].float()
    if values.shape[1] != expected_dimension or not torch.isfinite(values).all():
        raise RuntimeError("invalid E-TSS embedding batch")
    return values
