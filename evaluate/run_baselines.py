#!/usr/bin/env python3
"""
Run baseline models on matched splits for CIPHER.

Baselines:
- Chance
- Logistic Regression (mean+std pooled over time)
- LDA (mean+std pooled over time)
- ShallowConvNet (Schirrmeister-style)
- EEGNet (Lawhern et al.-style compact EEG net)
- EEG-Conformer (Song et al.-style compact transformer baseline)

Tasks:
- phoneme_identity
- manner
- place

Features:
- erp
- dda

Outputs:
- results/tables/baseline_comparison.csv
- results/tables/baseline_summary_by_model.csv
- results/tables/wer_baseline_with_ci.csv
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.dataset import EEGDataset
from evaluate.eval_metrics import load_model, predict

RESULTS_DIR = ROOT / "results" / "tables"
MODELS_ROOT = ROOT / "models_out"

STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]
STUDY2_VAL = ["sub-S04", "sub-S09", "sub-S14"]
STUDY2_TRAIN = [s for s in STUDY2_ALL if s not in STUDY2_VAL]

FEATURE_TYPES = ["erp", "dda"]
TASKS = ["phoneme_identity", "manner", "place"]
SEEDS = [17, 31, 53]


@dataclass
class Metrics:
    acc: float
    f1_macro: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pooled_features(ds: EEGDataset) -> tuple[np.ndarray, np.ndarray]:
    x = ds.features
    mu = x.mean(axis=1)
    sigma = x.std(axis=1)
    feat = np.concatenate([mu, sigma], axis=1)
    y = ds.labels.astype(np.int64)
    return feat, y


class EEGNet(nn.Module):
    """Compact EEGNet-style network (Lawhern et al.) adapted to (B, T, D) input."""

    def __init__(self, n_ch: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        f1 = 8
        d = 2
        f2 = f1 * d
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, 1, D, T)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.mean(dim=-1).squeeze(-1)
        return self.classifier(x)


class ShallowConvNet(nn.Module):
    """ShallowConvNet-style architecture inspired by Schirrmeister et al. 2017."""

    def __init__(self, n_ch: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        n_filters_time = 40
        n_filters_spat = 40
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, 25), bias=False, padding=(0, 12))
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat, (n_ch, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15), ceil_mode=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, 1, D, T)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        x = x.mean(dim=-1).squeeze(-1)
        return self.fc(x)


class EEGConformer(nn.Module):
    """Compact EEG-Conformer style baseline for (B, T, D) EEG sequences."""

    def __init__(self, n_ch: int, n_classes: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        # Temporal conv projection with stride to reduce sequence length.
        self.temporal = nn.Sequential(
            nn.Conv1d(n_ch, d_model, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.temporal(x)
        # (B, D, T') -> (B, T', D)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    max_epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 6,
) -> np.ndarray:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    xtr = torch.from_numpy(x_train.astype(np.float32))
    ytr = torch.from_numpy(y_train.astype(np.int64))
    xva = torch.from_numpy(x_val.astype(np.float32))

    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    bad = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(xva.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, preds)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(xva.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds


def chance_metrics(y_val: np.ndarray, n_classes: int) -> Metrics:
    # For balanced random guesses, expected acc is 1/C; macro-F1 approximates same in expectation.
    acc = 1.0 / max(n_classes, 1)
    return Metrics(acc=acc, f1_macro=acc)


def eval_classical(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    if model_name == "lr":
        clf = LogisticRegression(
            random_state=seed,
            max_iter=1200,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=1,
        )
    elif model_name == "lda":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError(model_name)

    clf.fit(x_train_s, y_train)
    return clf.predict(x_val_s)


def build_dataset(subjects: list[str], feature_type: str, task: str, task_type_filter: list[str] | None = None) -> EEGDataset:
    return EEGDataset(
        subjects=subjects,
        feature_type=feature_type,
        classification_task=task,
        tms_condition="NULL",
        task_type_filter=task_type_filter,
        augment=False,
        subsample=1.0,
        temporal_stride=4 if feature_type == "dda" else 1,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        acc=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def bootstrap_subject_ci(values_by_subject: dict[str, float], n_boot: int = 2000, seed: int = 123) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    subs = list(values_by_subject.keys())
    vals = np.array([values_by_subject[s] for s in subs], dtype=np.float64)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots.append(float(vals[idx].mean()))
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(vals.mean()), float(low), float(high)


def token_wer(ref: np.ndarray, hyp: np.ndarray) -> float:
    # One token per trial for phoneme_identity in this pipeline.
    return float(np.mean(ref != hyp))


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for feature_type in FEATURE_TYPES:
        for task in TASKS:
            ds_train = build_dataset(STUDY2_TRAIN, feature_type, task)
            ds_val = build_dataset(STUDY2_VAL, feature_type, task)

            if len(ds_train) == 0 or len(ds_val) == 0:
                continue

            x_train_pool, y_train = pooled_features(ds_train)
            x_val_pool, y_val = pooled_features(ds_val)
            n_classes = ds_train.n_classes

            # Chance
            cm = chance_metrics(y_val, n_classes)
            rows.append({
                "model": "chance",
                "feature_type": feature_type,
                "task": task,
                "seed": -1,
                "acc": cm.acc,
                "f1_macro": cm.f1_macro,
            })

            for seed in SEEDS:
                # LR
                pred_lr = eval_classical("lr", x_train_pool, y_train, x_val_pool, y_val, seed)
                m_lr = compute_metrics(y_val, pred_lr)
                rows.append({
                    "model": "lr_mean_std",
                    "feature_type": feature_type,
                    "task": task,
                    "seed": seed,
                    "acc": m_lr.acc,
                    "f1_macro": m_lr.f1_macro,
                })

                # LDA
                pred_lda = eval_classical("lda", x_train_pool, y_train, x_val_pool, y_val, seed)
                m_lda = compute_metrics(y_val, pred_lda)
                rows.append({
                    "model": "lda_mean_std",
                    "feature_type": feature_type,
                    "task": task,
                    "seed": seed,
                    "acc": m_lda.acc,
                    "f1_macro": m_lda.f1_macro,
                })

                # Deep baselines use raw sequence tensors
                x_train = ds_train.features.astype(np.float32)
                x_val = ds_val.features.astype(np.float32)
                y_train_np = ds_train.labels.astype(np.int64)
                y_val_np = ds_val.labels.astype(np.int64)

                eegnet = EEGNet(n_ch=x_train.shape[2], n_classes=n_classes)
                pred_eegnet = train_torch_model(
                    eegnet, x_train, y_train_np, x_val, y_val_np, seed,
                    max_epochs=args.deep_epochs, patience=args.deep_patience,
                )
                m_eeg = compute_metrics(y_val_np, pred_eegnet)
                rows.append({
                    "model": "eegnet",
                    "feature_type": feature_type,
                    "task": task,
                    "seed": seed,
                    "acc": m_eeg.acc,
                    "f1_macro": m_eeg.f1_macro,
                })

                shallow = ShallowConvNet(n_ch=x_train.shape[2], n_classes=n_classes)
                pred_sh = train_torch_model(
                    shallow, x_train, y_train_np, x_val, y_val_np, seed,
                    max_epochs=args.deep_epochs, patience=args.deep_patience,
                )
                m_sh = compute_metrics(y_val_np, pred_sh)
                rows.append({
                    "model": "shallowconvnet",
                    "feature_type": feature_type,
                    "task": task,
                    "seed": seed,
                    "acc": m_sh.acc,
                    "f1_macro": m_sh.f1_macro,
                })

                conformer = EEGConformer(n_ch=x_train.shape[2], n_classes=n_classes)
                pred_cf = train_torch_model(
                    conformer, x_train, y_train_np, x_val, y_val_np, seed,
                    max_epochs=args.deep_epochs, patience=args.deep_patience,
                )
                m_cf = compute_metrics(y_val_np, pred_cf)
                rows.append({
                    "model": "eeg_conformer",
                    "feature_type": feature_type,
                    "task": task,
                    "seed": seed,
                    "acc": m_cf.acc,
                    "f1_macro": m_cf.f1_macro,
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)

    # Summary table: mean ± std over seeds
    sum_df = (
        df[df["seed"] >= 0]
        .groupby(["model", "feature_type", "task"], as_index=False)
        .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"))
    )

    chance_df = (
        df[df["model"] == "chance"]
        .groupby(["model", "feature_type", "task"], as_index=False)
        .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"))
    )
    out_sum = pd.concat([sum_df, chance_df], ignore_index=True)
    out_sum.to_csv(RESULTS_DIR / "baseline_summary_by_model.csv", index=False)

    # WER table with per-subject CI for phoneme_identity on CVC subsets, null condition
    if not args.skip_wer:
        wer_rows = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for feature_type in FEATURE_TYPES:
            # CIPHER (trained)
            model_dir = MODELS_ROOT / feature_type / "phoneme_identity" / "null"
            model, cfg = load_model(model_dir, device)
            if model is not None:
                subj_wer = {}
                for sub in STUDY2_VAL:
                    ds_real = build_dataset([sub], feature_type, "phoneme_identity", ["cvc_real_words"])
                    if len(ds_real) == 0:
                        continue
                    _, pred, lab = predict(model, ds_real, device, task="phoneme_identity")
                    subj_wer[sub] = token_wer(lab, pred)
                mean_v, low, high = bootstrap_subject_ci(subj_wer, n_boot=args.n_boot, seed=123)
                wer_rows.append({
                    "model": "cipher",
                    "feature_type": feature_type,
                    "word_type": "cvc_real_words",
                    "wer_mean": mean_v,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_subjects": len(subj_wer),
                })

                subj_wer = {}
                for sub in STUDY2_VAL:
                    ds_p = build_dataset([sub], feature_type, "phoneme_identity", ["cvc_pseudowords"])
                    if len(ds_p) == 0:
                        continue
                    _, pred, lab = predict(model, ds_p, device, task="phoneme_identity")
                    subj_wer[sub] = token_wer(lab, pred)
                mean_v, low, high = bootstrap_subject_ci(subj_wer, n_boot=args.n_boot, seed=124)
                wer_rows.append({
                    "model": "cipher",
                    "feature_type": feature_type,
                    "word_type": "cvc_pseudowords",
                    "wer_mean": mean_v,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_subjects": len(subj_wer),
                })

            # Chance WER
            wer_rows.append({
                "model": "chance",
                "feature_type": feature_type,
                "word_type": "cvc_real_words",
                "wer_mean": 1.0 - 1.0 / 11.0,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "n_subjects": 3,
            })
            wer_rows.append({
                "model": "chance",
                "feature_type": feature_type,
                "word_type": "cvc_pseudowords",
                "wer_mean": 1.0 - 1.0 / 11.0,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "n_subjects": 3,
            })

            # LR/LDA/EEGNet/Shallow/EEG-Conformer on pooled split; evaluate per subject for CIs
            ds_train = build_dataset(STUDY2_TRAIN, feature_type, "phoneme_identity")
            x_train_pool, y_train = pooled_features(ds_train)
            scaler = StandardScaler().fit(x_train_pool)

            # train per-seed models and average subject WER after ensembled majority vote over seeds
            for model_name in ["lr_mean_std", "lda_mean_std", "eegnet", "shallowconvnet", "eeg_conformer"]:
                for word_type in ["cvc_real_words", "cvc_pseudowords"]:
                    subj_wers = {}
                    for sub in STUDY2_VAL:
                        ds_sub = build_dataset([sub], feature_type, "phoneme_identity", [word_type])
                        if len(ds_sub) == 0:
                            continue
                        y_true = ds_sub.labels.astype(np.int64)

                        seed_preds = []
                        for seed in SEEDS:
                            if model_name == "lr_mean_std":
                                clf = LogisticRegression(
                                    random_state=seed,
                                    max_iter=1200,
                                    class_weight="balanced",
                                    solver="lbfgs",
                                    n_jobs=1,
                                )
                                clf.fit(scaler.transform(x_train_pool), y_train)
                                x_sub, _ = pooled_features(ds_sub)
                                pred = clf.predict(scaler.transform(x_sub))
                            elif model_name == "lda_mean_std":
                                clf = LinearDiscriminantAnalysis()
                                clf.fit(scaler.transform(x_train_pool), y_train)
                                x_sub, _ = pooled_features(ds_sub)
                                pred = clf.predict(scaler.transform(x_sub))
                            else:
                                xtr = ds_train.features.astype(np.float32)
                                ytr = ds_train.labels.astype(np.int64)
                                x_sub = ds_sub.features.astype(np.float32)
                                y_sub = ds_sub.labels.astype(np.int64)
                                if model_name == "eegnet":
                                    m = EEGNet(n_ch=xtr.shape[2], n_classes=11)
                                elif model_name == "shallowconvnet":
                                    m = ShallowConvNet(n_ch=xtr.shape[2], n_classes=11)
                                else:
                                    m = EEGConformer(n_ch=xtr.shape[2], n_classes=11)
                                pred = train_torch_model(
                                    m, xtr, ytr, x_sub, y_sub, seed,
                                    max_epochs=(args.wer_deep_epochs if args.wer_deep_epochs is not None else max(12, args.deep_epochs // 2)),
                                    patience=(args.wer_deep_patience if args.wer_deep_patience is not None else max(3, args.deep_patience // 2)),
                                )
                            seed_preds.append(pred)

                        # Majority vote over seeds
                        stacked = np.stack(seed_preds, axis=0)
                        voted = np.apply_along_axis(lambda a: np.bincount(a, minlength=11).argmax(), axis=0, arr=stacked)
                        subj_wers[sub] = token_wer(y_true, voted)

                    mean_v, low, high = bootstrap_subject_ci(subj_wers, n_boot=args.n_boot, seed=555)
                    wer_rows.append({
                        "model": model_name,
                        "feature_type": feature_type,
                        "word_type": word_type,
                        "wer_mean": mean_v,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_subjects": len(subj_wers),
                    })

        wer_df = pd.DataFrame(wer_rows)
        wer_df.to_csv(RESULTS_DIR / "wer_baseline_with_ci.csv", index=False)

    print(f"Saved: {RESULTS_DIR / 'baseline_comparison.csv'}")
    print(f"Saved: {RESULTS_DIR / 'baseline_summary_by_model.csv'}")
    if args.skip_wer:
        print("Skipped WER baseline table (--skip-wer).")
    else:
        print(f"Saved: {RESULTS_DIR / 'wer_baseline_with_ci.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline benchmarks for CIPHER")
    parser.add_argument("--deep-epochs", type=int, default=24)
    parser.add_argument("--deep-patience", type=int, default=5)
    parser.add_argument("--wer-deep-epochs", type=int, default=None,
                        help="Override deep epochs used in per-subject WER baseline fits")
    parser.add_argument("--wer-deep-patience", type=int, default=None,
                        help="Override deep patience used in per-subject WER baseline fits")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--skip-wer", action="store_true",
                        help="Skip expensive WER baseline computation and only emit classification baseline tables")
    args = parser.parse_args()
    run(args)
