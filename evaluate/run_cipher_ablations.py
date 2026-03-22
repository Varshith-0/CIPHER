#!/usr/bin/env python3
"""Run CIPHER architecture ablations on expanded (8-8) validation split.

Outputs:
- results/tables/cipher_ablation_split8_extended_raw.csv
- results/tables/cipher_ablation_split8_extended_summary.csv
"""

from __future__ import annotations

import subprocess
import sys
import os
import argparse
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train_all.py"
RESULTS_DIR = ROOT / "results" / "tables"
MODELS_ROOT = ROOT / "models_out"

SEEDS = [17, 31, 53]
FEATURE_TYPES = ["erp", "dda"]
TASKS = ["phoneme_identity", "manner", "place"]
ABLATIONS = [
    ("full", []),
    ("no_se", ["--no-se"]),
    ("no_stochastic_depth", ["--drop-path-rate", "0.0"]),
    ("no_attention_pool", ["--no-attention-pool"]),
    ("no_multiscale", ["--no-multiscale"]),
]


def run_cmd(cmd: list[str]) -> int:
    env = os.environ.copy()
    # Avoid MKL/libgomp threading layer conflicts on this host.
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    return proc.wait()


def best_val_acc(feature_type: str, task: str, save_suffix: str) -> float:
    log_path = MODELS_ROOT / feature_type / task / "null" / save_suffix / "training_log.csv"
    if not log_path.exists():
        return float("nan")
    df = pd.read_csv(log_path)
    if "val_acc" not in df.columns or df.empty:
        return float("nan")
    return float(df["val_acc"].max())


def parse_csv_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_csv_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIPHER ablations on expanded split")
    parser.add_argument("--feature-types", type=str, default=",".join(FEATURE_TYPES),
                        help="Comma-separated feature types (default: erp,dda)")
    parser.add_argument("--tasks", type=str, default=",".join(TASKS),
                        help="Comma-separated tasks (default: phoneme_identity,manner,place)")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS),
                        help="Comma-separated random seeds (default: 17,31,53)")
    parser.add_argument("--output-prefix", type=str, default="cipher_ablation_split8_extended")
    args = parser.parse_args()

    feature_types = parse_csv_list(args.feature_types, FEATURE_TYPES)
    tasks = parse_csv_list(args.tasks, TASKS)
    seeds = parse_int_csv_list(args.seeds, SEEDS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    jobs = []
    for feature_type in feature_types:
        for task in tasks:
            for ab_name, flags in ABLATIONS:
                for seed in seeds:
                    jobs.append((feature_type, task, ab_name, flags, seed))

    for feature_type, task, ab_name, flags, seed in tqdm(jobs, desc="CIPHER ablations (8-8)", unit="run"):
        suffix = f"ablate_{feature_type}_{task}_{ab_name}_seed{seed}"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--task", task,
            "--feature-type", feature_type,
            "--tms", "null",
            "--skip-modality",
            "--split-8-8",
            "--max-epochs", str(args.max_epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--seed", str(seed),
            "--save-suffix", suffix,
        ] + flags

        rc = run_cmd(cmd)
        val_acc = best_val_acc(feature_type, task, suffix)
        rows.append({
            "feature_type": feature_type,
            "task": task,
            "ablation": ab_name,
            "seed": seed,
            "return_code": rc,
            "best_val_acc": val_acc,
            "save_suffix": suffix,
        })

    raw = pd.DataFrame(rows)
    raw_path = RESULTS_DIR / f"{args.output_prefix}_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby(["feature_type", "task", "ablation"], as_index=False)
        .agg(
            n_runs=("best_val_acc", "count"),
            val_acc_mean=("best_val_acc", "mean"),
            val_acc_std=("best_val_acc", "std"),
            n_fail=("return_code", lambda s: int((s != 0).sum())),
        )
        .sort_values(["feature_type", "task", "val_acc_mean"], ascending=[True, True, False])
    )
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
