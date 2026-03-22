#!/usr/bin/env python3
"""Run remaining controls: early-window masking and block-aware permutation.

Outputs:
- results/tables/time_window_control_loso.csv
- results/tables/time_window_control_summary.csv
- results/tables/permutation_block_control.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.dataset as dsmod
from models.dataset import EEGDataset

RESULTS_DIR = ROOT / "results" / "tables"

STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]
TASKS = ["phoneme_identity", "manner", "place"]
FEATURE_TYPES = ["erp", "dda"]
TMS_CONDITIONS = ["NULL", "LipTMS", "TongueTMS"]
TASK_TYPES = [
    "single_phoneme_perceived",
    "single_phoneme_spoken",
    "cv_pairs",
    "vc_pairs",
    "cvc_real_words",
    "cvc_pseudowords",
]
FEATURE_CLIP_ABS = 1e6


def set_preprocessed_root(path: Path) -> None:
    dsmod.PREPROCESSED_ROOT = Path(path)


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 123) -> tuple[float, float, float]:
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(float(arr[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def sanitize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=FEATURE_CLIP_ABS, neginf=-FEATURE_CLIP_ABS)
    x = np.clip(x, -FEATURE_CLIP_ABS, FEATURE_CLIP_ABS)
    return x.astype(np.float32, copy=False)


def pooled_features(x: np.ndarray) -> np.ndarray:
    x = sanitize_matrix(x)
    mu = x.mean(axis=1)
    sigma = x.std(axis=1)
    pooled = np.concatenate([mu, sigma], axis=1)
    return sanitize_matrix(pooled)


def mask_early_erp_window(x: np.ndarray) -> np.ndarray:
    """Mask 0-200 ms in epochs spanning [-200, +800] ms (1s total)."""
    x = x.copy()
    seq_len = x.shape[1]
    pre_samples = int(round(0.2 * seq_len))
    post200_samples = int(round(0.2 * seq_len))
    start = pre_samples
    end = min(seq_len, start + post200_samples)
    x[:, start:end, :] = 0.0
    return x


def fit_predict_lr(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int = 42) -> np.ndarray:
    x_tr = sanitize_matrix(x_tr)
    x_te = sanitize_matrix(x_te)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    clf = LogisticRegression(
        random_state=seed,
        max_iter=1200,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_tr_s, y_tr)
    return clf.predict(x_te_s)


def build_ds(subjects: list[str], feature_type: str, task: str, tms_condition: str | None = None, task_type_filter: list[str] | None = None) -> EEGDataset:
    return EEGDataset(
        subjects=subjects,
        feature_type=feature_type,
        classification_task=task,
        tms_condition=tms_condition,
        task_type_filter=task_type_filter,
        augment=False,
        temporal_stride=4 if feature_type == "dda" else 1,
    )


def run_time_window_control(preprocessed_root: Path, tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_preprocessed_root(preprocessed_root)
    rows = []

    for task in tqdm(tasks, desc="Time-window control tasks", unit="task"):
        for held_out in tqdm(STUDY2_ALL, desc=f"LOSO {task}", unit="sub", leave=False):
            tr_subs = [s for s in STUDY2_ALL if s != held_out]
            ds_tr = build_ds(tr_subs, "erp", task, tms_condition="NULL")
            ds_te = build_ds([held_out], "erp", task, tms_condition="NULL")
            if len(ds_tr) == 0 or len(ds_te) == 0:
                continue

            xtr = ds_tr.features
            xte = ds_te.features
            ytr = ds_tr.labels.astype(np.int64)
            yte = ds_te.labels.astype(np.int64)

            pred_base = fit_predict_lr(pooled_features(xtr), ytr, pooled_features(xte), seed=11)
            pred_mask = fit_predict_lr(
                pooled_features(mask_early_erp_window(xtr)),
                ytr,
                pooled_features(mask_early_erp_window(xte)),
                seed=11,
            )

            rows.append({
                "task": task,
                "held_out_subject": held_out,
                "eval": "LOSO_NULL",
                "acc_base": float(accuracy_score(yte, pred_base)),
                "acc_mask0_200": float(accuracy_score(yte, pred_mask)),
                "f1_base": float(f1_score(yte, pred_base, average="macro", zero_division=0)),
                "f1_mask0_200": float(f1_score(yte, pred_mask, average="macro", zero_division=0)),
            })

    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, pd.DataFrame()

    summaries = []
    for task, g in folds.groupby("task"):
        acc_base_m, acc_base_lo, acc_base_hi = bootstrap_ci(g["acc_base"].tolist(), seed=101)
        acc_mask_m, acc_mask_lo, acc_mask_hi = bootstrap_ci(g["acc_mask0_200"].tolist(), seed=102)
        summaries.append({
            "task": task,
            "n_folds": int(len(g)),
            "acc_base_mean": acc_base_m,
            "acc_base_ci95_low": acc_base_lo,
            "acc_base_ci95_high": acc_base_hi,
            "acc_mask0_200_mean": acc_mask_m,
            "acc_mask0_200_ci95_low": acc_mask_lo,
            "acc_mask0_200_ci95_high": acc_mask_hi,
            "delta_mask_minus_base": acc_mask_m - acc_base_m,
        })
    return folds, pd.DataFrame(summaries)


def _collect_blocked_train(subjects: list[str], feature_type: str, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = []
    labels = []
    blocks = []
    block_id = 0

    for sub in subjects:
        for tms in TMS_CONDITIONS:
            for tt in TASK_TYPES:
                ds = build_ds([sub], feature_type, task, tms_condition=tms, task_type_filter=[tt])
                if len(ds) == 0:
                    continue
                x = pooled_features(ds.features)
                y = ds.labels.astype(np.int64)
                feats.append(x)
                labels.append(y)
                blocks.append(np.full(len(y), block_id, dtype=np.int64))
                block_id += 1

    if not feats:
        return np.zeros((0, 1), dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0), np.concatenate(blocks, axis=0)


def _permute_within_blocks(y: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yp = y.copy()
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        if len(idx) > 1:
            yp[idx] = rng.permutation(yp[idx])
    return yp


def run_block_permutation_control(preprocessed_root: Path, feature_types: list[str], tasks: list[str], n_perm: int = 100) -> pd.DataFrame:
    set_preprocessed_root(preprocessed_root)
    rows = []

    for ft in tqdm(feature_types, desc="Permutation features", unit="feature"):
        for task in tqdm(tasks, desc=f"Permutation tasks ({ft})", unit="task", leave=False):
            fold_true = []
            perm_fold_means = []

            for held_out in tqdm(STUDY2_ALL, desc=f"LOSO perm {ft}/{task}", unit="sub", leave=False):
                tr_subs = [s for s in STUDY2_ALL if s != held_out]

                xtr, ytr, btr = _collect_blocked_train(tr_subs, ft, task)
                ds_te = build_ds([held_out], ft, task, tms_condition=None)
                if len(xtr) == 0 or len(ds_te) == 0:
                    continue

                xte = pooled_features(ds_te.features)
                yte = ds_te.labels.astype(np.int64)

                try:
                    pred_true = fit_predict_lr(xtr, ytr, xte, seed=17)
                except ValueError as e:
                    print(f"[WARN] Skipping fold {ft}/{task}/{held_out} due to invalid features: {e}")
                    continue
                acc_true = float(accuracy_score(yte, pred_true))
                fold_true.append(acc_true)

                rng = np.random.default_rng(1234 + len(fold_true))
                perm_accs = []
                for _ in range(n_perm):
                    yperm = _permute_within_blocks(ytr, btr, rng)
                    try:
                        pred_perm = fit_predict_lr(xtr, yperm, xte, seed=17)
                    except ValueError as e:
                        print(f"[WARN] Skipping permutation in {ft}/{task}/{held_out} due to invalid features: {e}")
                        continue
                    perm_accs.append(float(accuracy_score(yte, pred_perm)))
                if perm_accs:
                    perm_fold_means.append(float(np.mean(perm_accs)))

            if len(fold_true) == 0:
                continue

            if len(perm_fold_means) == 0:
                continue

            true_mean = float(np.mean(fold_true))
            perm_mean = float(np.mean(perm_fold_means))
            p_value = float((np.sum(np.asarray(perm_fold_means) >= true_mean) + 1) / (len(perm_fold_means) + 1))

            rows.append({
                "feature_type": ft,
                "task": task,
                "eval": "LOSO_allTMS",
                "n_folds": int(len(fold_true)),
                "n_perm_per_fold": int(n_perm),
                "acc_true_mean": true_mean,
                "acc_perm_mean": perm_mean,
                "delta_true_minus_perm": true_mean - perm_mean,
                "p_value_empirical": p_value,
            })

    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(args.preprocessed_root)

    tw_folds, tw_summary = run_time_window_control(root, TASKS)
    tw_folds.to_csv(RESULTS_DIR / "time_window_control_loso.csv", index=False)
    tw_summary.to_csv(RESULTS_DIR / "time_window_control_summary.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'time_window_control_loso.csv'}")
    print(f"Saved {RESULTS_DIR / 'time_window_control_summary.csv'}")

    perm = run_block_permutation_control(root, FEATURE_TYPES, TASKS, n_perm=args.n_perm)
    perm.to_csv(RESULTS_DIR / "permutation_block_control.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'permutation_block_control.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run remaining control experiments")
    parser.add_argument("--preprocessed-root", type=str, default=str(ROOT / "preprocessed"))
    parser.add_argument("--n-perm", type=int, default=50)
    args = parser.parse_args()
    run(args)
