#!/usr/bin/env python3
"""Run Study-2 LOSO CVC triphone WER for CIPHER (ERP and DDA).

This script trains one fold per held-out subject (16 folds total), then evaluates
phoneme-level WER on:
- cvc_real_words
- cvc_pseudowords
- cvc_all (union of both)

Outputs:
- results/tables/wer_loso_cipher_cvc_folds.csv
- results/tables/wer_loso_cipher_cvc_summary.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate.eval_metrics import load_model, predict
from models.dataset import EEGDataset
from models.train import run_experiment

RESULTS_DIR = ROOT / "results" / "tables"
MODELS_ROOT = ROOT / "models_out"
LOSO_MODELS_ROOT = MODELS_ROOT / "loso_wer"

FEATURE_TYPES = ["erp", "dda"]
WORD_TYPES = ["cvc_real_words", "cvc_pseudowords"]
STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]


def token_wer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred)) if len(y_true) > 0 else np.nan


def build_dataset(subjects: list[str], feature_type: str, task_type_filter: list[str] | None = None) -> EEGDataset:
    return EEGDataset(
        subjects=subjects,
        feature_type=feature_type,
        classification_task="phoneme_identity",
        tms_condition="NULL",
        task_type_filter=task_type_filter,
        augment=False,
        subsample=1.0,
        temporal_stride=4 if feature_type == "dda" else 1,
    )


def load_base_config(feature_type: str, args: argparse.Namespace) -> dict:
    cfg_path = MODELS_ROOT / feature_type / "phoneme_identity" / "null" / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "model_type": "conformer",
            "d_model": 192,
            "n_conformer_blocks": 4,
            "n_heads": 4,
            "conv_channels": 64,
            "conv_kernel": 15,
            "dropout": 0.3,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "max_epochs": 120,
            "patience": 20,
            "scheduler_patience": 6,
            "scheduler_factor": 0.5,
            "multi_task": False,
            "ctc": False,
            "ctc_weight": 0.3,
            "enable_ctc": False,
            "augment_train": False,
            "amp": True,
            "compile": False,
            "dataloader_workers": 0,
            "use_multiscale": True,
            "use_se": True,
            "use_attention_pool": True,
            "drop_path_rate": 0.15,
            "seed": 42,
        }

    cfg = copy.deepcopy(cfg)
    cfg["skip_existing"] = args.skip_existing
    cfg["compile"] = False
    cfg["dataloader_workers"] = 0

    if args.max_epochs is not None:
        cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        cfg["patience"] = int(args.patience)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    return cfg


def split_inner_train_val(train_subjects: list[str]) -> tuple[list[str], list[str]]:
    if len(train_subjects) < 3:
        return train_subjects, train_subjects
    val_subject = train_subjects[-1]
    return train_subjects[:-1], [val_subject]


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOSO_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for feature_type in FEATURE_TYPES:
        base_cfg = load_base_config(feature_type, args)
        print(f"=== LOSO WER | {feature_type.upper()} ===")

        for fold_idx, held_out in enumerate(STUDY2_ALL, start=1):
            train_subjects = [s for s in STUDY2_ALL if s != held_out]
            tr_subjects, va_subjects = split_inner_train_val(train_subjects)

            ds_train = build_dataset(tr_subjects, feature_type, WORD_TYPES)
            ds_val = build_dataset(va_subjects, feature_type, WORD_TYPES)

            fold_dir = LOSO_MODELS_ROOT / feature_type / "phoneme_identity" / "null" / f"heldout_{held_out}"
            cfg = copy.deepcopy(base_cfg)
            cfg["device"] = str(device)

            print(
                f"[{feature_type} fold {fold_idx:02d}/16] held_out={held_out} "
                f"train_subs={len(tr_subjects)} val_subs={len(va_subjects)}"
            )
            run_experiment(
                train_dataset=ds_train,
                val_dataset=ds_val,
                test_dataset=None,
                save_dir=fold_dir,
                config=cfg,
            )

            model, _ = load_model(fold_dir, device)
            if model is None:
                print(f"  WARN: no model found for {fold_dir}, skipping fold")
                continue

            by_type = {}
            for wt in WORD_TYPES:
                ds_te = build_dataset([held_out], feature_type, [wt])
                if len(ds_te) == 0:
                    continue
                _, pred, y_true = predict(model, ds_te, device, task="phoneme_identity")
                by_type[wt] = {
                    "wer": token_wer(y_true, pred),
                    "n_samples": int(len(y_true)),
                }
                rows.append({
                    "feature_type": feature_type,
                    "held_out_subject": held_out,
                    "word_type": wt,
                    "wer": by_type[wt]["wer"],
                    "n_samples": by_type[wt]["n_samples"],
                })

            ds_te_all = build_dataset([held_out], feature_type, WORD_TYPES)
            if len(ds_te_all) > 0:
                _, pred_all, y_all = predict(model, ds_te_all, device, task="phoneme_identity")
                rows.append({
                    "feature_type": feature_type,
                    "held_out_subject": held_out,
                    "word_type": "cvc_all",
                    "wer": token_wer(y_all, pred_all),
                    "n_samples": int(len(y_all)),
                })

    folds_df = pd.DataFrame(rows)
    folds_path = RESULTS_DIR / "wer_loso_cipher_cvc_folds.csv"
    folds_df.to_csv(folds_path, index=False)

    summary = (
        folds_df.groupby(["feature_type", "word_type"], as_index=False)
        .agg(
            n_folds=("wer", "count"),
            wer_mean=("wer", "mean"),
            wer_std=("wer", "std"),
            n_samples_total=("n_samples", "sum"),
        )
        .sort_values(["word_type", "feature_type"])
    )
    summary_path = RESULTS_DIR / "wer_loso_cipher_cvc_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {folds_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOSO WER for CIPHER on Study-2 CVC triphones")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training max epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--skip-existing", action="store_true", help="Skip folds with existing checkpoints")
    run(parser.parse_args())
