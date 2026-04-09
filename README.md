# CIPHER

Paper: [CIPHER](https://arxiv.org/abs/2604.02362)

## Overview

CIPHER is an EEG speech-decoding pipeline covering:
- ERP and DDA feature extraction from BIDS EEG,
- multi-task neural decoding for phoneme and articulatory targets,
- matched-split baselines and control analyses,
- automatic generation of publication figures and tables.

## Highlights

- End-to-end pipeline: preprocessing -> training -> evaluation.
- Deterministic defaults for reproducibility (seeded + deterministic backends).
- Baseline suite: chance, LR, LDA, EEGNet, ShallowConvNet, EEG-Conformer.
- WER-focused analyses and sweep scripts for robust model selection.

## Repository Layout

- preprocess.py: preprocesses raw BIDS EEG into ERP/DDA tensors.
- train_all.py: main CIPHER training entrypoint.
- evaluate_all.py: main evaluation/analysis entrypoint.
- evaluate/run_baselines.py: matched-split baseline benchmarking.
- evaluate/run_wer_baselines_ci.py: WER baselines with bootstrap confidence intervals.
- evaluate/make_paper_figures.py: regenerates paper-ready figures from result tables.
- run_cipher.sh: one-command pipeline runner.
- run_wer_sweep.sh: ERP+DDA WER sweep.
- run_dda_mini_sweep.sh: focused DDA WER sweep.

## Environment Setup

Python 3.10+ recommended.

Option A (venv, for manual python commands):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Option B (conda, recommended if you use run_cipher.sh):

```bash
conda create -n cipher python=3.10 -y
conda activate cipher
pip install --upgrade pip
pip install -r requirements.txt
```

If your GPU/CUDA setup requires a different PyTorch build, install the appropriate wheel after requirements installation.

## Dataset (OpenNeuro ds006104)

Dataset page:
- https://openneuro.org/datasets/ds006104

Recommended download method (OpenNeuro CLI):

```bash
npm install -g openneuro-cli
openneuro download --dataset ds006104 --snapshot 1.0.2 ./ds006104
```

Expected local structure:

```text
ds006104/
  derivatives/
    eeglab/
  sub-P01/
  ...
  sub-S16/
```

## Reproducibility

This repository uses deterministic defaults in preprocessing, training, and evaluation:
- fixed random seeds,
- deterministic torch/cudnn settings,
- deterministic DataLoader generator use in training,
- pinned dependencies in requirements.txt.

Recommended seed for paper replication:

```bash
python preprocess.py --seed 42
python train_all.py --seed 42
python evaluate_all.py --seed 42
```

For speed-oriented ablations (reduced training budget):

```bash
python train_all.py --max-epochs 40 --patience 8 --seed 42
python evaluate_all.py --analysis metrics --analysis wer --dry-run --seed 42
```

## Quick Start (Full Pipeline)

Run everything:

```bash
bash run_cipher.sh
```

Smoke test:

```bash
bash run_cipher.sh --dry-run
```

Stage-wise execution:

```bash
bash run_cipher.sh --stage deps
bash run_cipher.sh --stage wav2vec
bash run_cipher.sh --stage preprocess
bash run_cipher.sh --stage train
bash run_cipher.sh --stage eval
```

## Manual Reproduction

### 1) Preprocess

```bash
python preprocess.py --skip-existing --seed 42
```

### 2) Train

```bash
python train_all.py --skip-existing --seed 42
```

Example targeted run (phoneme identity, NULL condition):

```bash
python train_all.py \
  --task phoneme_identity \
  --feature-type all \
  --tms null \
  --skip-modality \
  --seed 42
```

### 3) Evaluate

```bash
python evaluate_all.py --seed 42
```

Subset of analyses:

```bash
python evaluate_all.py --analysis metrics --analysis wer --seed 42
```

## Baselines and Controls

Matched-split baselines:

```bash
python evaluate/run_baselines.py
```

WER baseline table with bootstrap CI:

```bash
python evaluate/run_wer_baselines_ci.py --n-boot 2000
```

## Sweeps

Joint ERP+DDA WER sweep:

```bash
bash run_wer_sweep.sh
```

Focused DDA mini-sweep:

```bash
bash run_dda_mini_sweep.sh
```

## Paper Figures

Generate publication figures from computed tables:

```bash
python evaluate/make_paper_figures.py
```

Outputs are saved under:

```text
results/figures/paper/
```

## Main Outputs

- models_out/: trained checkpoints and logs.
- results/tables/: aggregate metrics, ablations, and control tables.
- results/figures/: evaluation and publication plots.
- results/summary_report.txt: consolidated run summary.


## Contact

For issues, open a GitHub issue with:
- environment details,
- exact command run,
- full traceback/log snippet,
- expected vs observed behavior.
