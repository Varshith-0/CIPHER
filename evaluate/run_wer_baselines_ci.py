#!/usr/bin/env python3
"""Compute baseline WER tables with per-subject bootstrap CIs for study2_val.

Models: chance, lr_mean_std, lda_mean_std, eegnet, shallowconvnet, cipher
Features: erp, dda
Word types: cvc_real_words, cvc_pseudowords
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import EEGDataset
from evaluate.eval_metrics import load_model, predict

RESULTS_DIR = ROOT / "results" / "tables"
MODELS_ROOT = ROOT / "models_out"

STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]
STUDY2_VAL = ["sub-S04", "sub-S09", "sub-S14"]
STUDY2_TRAIN = [s for s in STUDY2_ALL if s not in STUDY2_VAL]
SEEDS = [17, 31, 53]


class EEGNet(nn.Module):
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
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.mean(dim=-1).squeeze(-1)
        return self.classifier(x)


class ShallowConvNet(nn.Module):
    def __init__(self, n_ch: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 40, (1, 25), bias=False, padding=(0, 12))
        self.conv_spat = nn.Conv2d(40, 40, (n_ch, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15), ceil_mode=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pooled_features(ds: EEGDataset) -> tuple[np.ndarray, np.ndarray]:
    x = ds.features
    return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1), ds.labels.astype(np.int64)


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


def train_torch_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, seed: int, epochs: int, patience: int) -> nn.Module:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    xtr = torch.from_numpy(x_train.astype(np.float32))
    ytr = torch.from_numpy(y_train.astype(np.int64))
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=128, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        epoch_loss = running / max(n, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_torch(model: nn.Module, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32)).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def token_wer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))


def bootstrap_subject_ci(values_by_subject: dict[str, float], n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array(list(values_by_subject.values()), dtype=np.float64)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots.append(float(vals[idx].mean()))
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(vals.mean()), float(low), float(high)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for feature_type in ["erp", "dda"]:
        ds_train = build_dataset(STUDY2_TRAIN, feature_type)
        x_train_pool, y_train = pooled_features(ds_train)
        scaler = StandardScaler().fit(x_train_pool)
        x_train_raw = ds_train.features.astype(np.float32)

        # Train baselines once per seed
        lr_models = {}
        lda_models = {}
        eeg_models = {}
        sh_models = {}
        for seed in SEEDS:
            lr = LogisticRegression(random_state=seed, max_iter=1200, class_weight="balanced", solver="lbfgs")
            lr.fit(scaler.transform(x_train_pool), y_train)
            lr_models[seed] = lr

            lda = LinearDiscriminantAnalysis()
            lda.fit(scaler.transform(x_train_pool), y_train)
            lda_models[seed] = lda

            eeg = EEGNet(n_ch=x_train_raw.shape[2], n_classes=11)
            eeg_models[seed] = train_torch_model(eeg, x_train_raw, y_train, seed, args.deep_epochs, args.deep_patience)

            sh = ShallowConvNet(n_ch=x_train_raw.shape[2], n_classes=11)
            sh_models[seed] = train_torch_model(sh, x_train_raw, y_train, seed, args.deep_epochs, args.deep_patience)

        # CIPHER model
        model_dir = MODELS_ROOT / feature_type / "phoneme_identity" / "null"
        cipher_model, _ = load_model(model_dir, device)

        for wt in ["cvc_real_words", "cvc_pseudowords"]:
            # chance
            rows.append({
                "model": "chance", "feature_type": feature_type, "word_type": wt,
                "wer_mean": 1 - 1 / 11, "ci95_low": np.nan, "ci95_high": np.nan, "n_subjects": len(STUDY2_VAL)
            })

            # per-subject WERs for each model
            subj_vals = {"lr_mean_std": {}, "lda_mean_std": {}, "eegnet": {}, "shallowconvnet": {}, "cipher": {}}

            for sub in STUDY2_VAL:
                ds_sub = build_dataset([sub], feature_type, [wt])
                y_true = ds_sub.labels.astype(np.int64)
                x_sub_pool, _ = pooled_features(ds_sub)
                x_sub_raw = ds_sub.features.astype(np.float32)

                # vote over seeds
                def majority(pred_list: list[np.ndarray]) -> np.ndarray:
                    stk = np.stack(pred_list, axis=0)
                    return np.apply_along_axis(lambda a: np.bincount(a, minlength=11).argmax(), axis=0, arr=stk)

                pred_lr = majority([lr_models[s].predict(scaler.transform(x_sub_pool)) for s in SEEDS])
                pred_lda = majority([lda_models[s].predict(scaler.transform(x_sub_pool)) for s in SEEDS])
                pred_eeg = majority([predict_torch(eeg_models[s], x_sub_raw) for s in SEEDS])
                pred_sh = majority([predict_torch(sh_models[s], x_sub_raw) for s in SEEDS])

                subj_vals["lr_mean_std"][sub] = token_wer(y_true, pred_lr)
                subj_vals["lda_mean_std"][sub] = token_wer(y_true, pred_lda)
                subj_vals["eegnet"][sub] = token_wer(y_true, pred_eeg)
                subj_vals["shallowconvnet"][sub] = token_wer(y_true, pred_sh)

                if cipher_model is not None and len(ds_sub) > 0:
                    _, pred_c, lab_c = predict(cipher_model, ds_sub, device, task="phoneme_identity")
                    subj_vals["cipher"][sub] = token_wer(lab_c, pred_c)

            for model_name, by_sub in subj_vals.items():
                mean_v, lo, hi = bootstrap_subject_ci(by_sub, n_boot=args.n_boot, seed=777)
                rows.append({
                    "model": model_name,
                    "feature_type": feature_type,
                    "word_type": wt,
                    "wer_mean": mean_v,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_subjects": len(by_sub),
                })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "wer_baseline_with_ci.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'wer_baseline_with_ci.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deep-epochs", type=int, default=20)
    p.add_argument("--deep-patience", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=2000)
    run(p.parse_args())
