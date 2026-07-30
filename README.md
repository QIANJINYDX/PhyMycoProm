# PhyMycoProm

Minimal training code for **PhyMycoProm-TSS**, the frozen DNA
foundation-model classifier described in:

> *PhyMycoProm: A Parameter-Efficient Cross-Species DNA Foundation Model for
> Plant–Fungal Promoter Recognition*

Only the core model is retained here; screening, ablation, transfer,
interpretability and figure-generation pipelines are intentionally excluded.

## Model

The input is a transcription-oriented 300-nt sequence spanning `[-249,+50]`
relative to the transcription start site. A frozen NTv3-650M-pre backbone
produces final-resolution contextual states, and the state at zero-based index
249 is used as the E-TSS representation.

```text
300-nt DNA
  → frozen NTv3-650M-pre
  → state 249 (1,536 dimensions)
  → training-fold StandardScaler
  → Linear(1536,256) → GELU → Dropout(0.2) → Linear(256,1)
```

Only the 393,729 classifier-head parameters are trained. The locked protocol
uses five leakage-safe folds, seeds `13`, `42` and `20260720`, and averages the
three sigmoid probabilities.

## Contents

- `extract_embeddings.py`: extracts frozen E-TSS embeddings.
- `train.py`: trains and evaluates the MLP heads.
- `phymycoprom/`: reusable model, data and training functions.
- `config.json`: locked core hyperparameters.
- `splits/fold_assignments.csv`: fixed paper fold mapping; it contains no DNA
  sequences.
- `tests/`: lightweight integrity tests.

## Dataset

The PhyMycoProm benchmark dataset is available on Hugging Face:
<https://huggingface.co/datasets/YDXX/PhyMycoProm>.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The NTv3 checkpoint is not redistributed in this repository. Supply a local
checkpoint directory or a compatible model identifier.

## Usage

Extract each unique sequence once:

```bash
python extract_embeddings.py \
  --data /path/to/phymycoprom.csv \
  --model /path/to/NTv3_650M_pre \
  --trust-remote-code \
  --local-files-only \
  --output-dir work/etss
```

`--trust-remote-code` permits execution of the model repository's Python code;
inspect and trust that code before enabling the option.

Train the locked five-fold, three-seed ensemble:

```bash
python train.py \
  --data /path/to/phymycoprom.csv \
  --embeddings work/etss/embeddings.npy \
  --record-map work/etss/record_map.csv \
  --fold-assignments splits/fold_assignments.csv \
  --output-dir outputs/core
```

The output directory contains fold/seed checkpoints, out-of-fold predictions,
training histories, fold assignments, metrics and a hash-recorded run
manifest. If `--fold-assignments` is omitted, the code creates a new
`StratifiedGroupKFold` split using `cv_group`; this is leakage-safe but is not
guaranteed to reproduce the paper's exact folds.

Run the lightweight checks with:

```bash
python -m unittest discover -s tests -v
```

## Licence

No software licence has yet been assigned. Add an author-approved licence
before making the GitHub repository public. The NTv3 checkpoint and benchmark
data remain subject to their own terms.
