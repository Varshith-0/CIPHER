#!/usr/bin/env python3
"""Run must-have controls: NULL-only, acoustic-only, LOSO primary, and 8-8 secondary.

Outputs:
- results/tables/null_only_eeg_controls_loso.csv
- results/tables/null_only_eeg_controls_8_8.csv
- results/tables/acoustic_baseline_controls_loso.csv
- results/tables/acoustic_baseline_controls_8_8.csv
- results/tables/wideband_erp_control_comparison.csv (if --wideband-root exists)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.dataset as dsmod
from models.dataset import EEGDataset, TASK_CONFIGS

RESULTS_DIR = ROOT / "results" / "tables"

STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]
SPLIT_8_TRAIN = STUDY2_ALL[:8]
SPLIT_8_TEST = STUDY2_ALL[8:]

TASKS = [
    "phoneme_identity",
    "place",
    "manner",
    "voicing",
    "category",
    "complexity",
]

FEATURE_TYPES = ["erp", "dda"]


def set_preprocessed_root(path: Path) -> None:
    dsmod.PREPROCESSED_ROOT = Path(path)


def pooled_features(dataset: EEGDataset) -> tuple[np.ndarray, np.ndarray]:
    x = dataset.features
    mu = x.mean(axis=1)
    sigma = x.std(axis=1)
    feat = np.concatenate([mu, sigma], axis=1)
    y = dataset.labels.astype(np.int64)
    return feat, y


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


def fit_predict_lr(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int = 42) -> np.ndarray:
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


def build_eeg_dataset(subjects: list[str], feature_type: str, task: str) -> EEGDataset:
    stride = 4 if feature_type == "dda" else 1
    return EEGDataset(
        subjects=subjects,
        feature_type=feature_type,
        classification_task=task,
        tms_condition="NULL",
        augment=False,
        subsample=1.0,
        temporal_stride=stride,
    )


def run_eeg_loso(preprocessed_root: Path, feature_types: list[str], tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_preprocessed_root(preprocessed_root)
    fold_rows = []

    combo_items = [(ft, tk) for ft in feature_types for tk in tasks]
    combo_bar = tqdm(combo_items, desc="EEG LOSO tasks", unit="task")

    for feature_type, task in combo_bar:
            fold_accs = []
            fold_f1s = []
            n_classes = len(TASK_CONFIGS[task]["classes"])

            subject_bar = tqdm(STUDY2_ALL, desc=f"LOSO {feature_type}/{task}", unit="sub", leave=False)
            for held_out in subject_bar:
                train_subs = [s for s in STUDY2_ALL if s != held_out]
                ds_tr = build_eeg_dataset(train_subs, feature_type, task)
                ds_te = build_eeg_dataset([held_out], feature_type, task)
                if len(ds_tr) == 0 or len(ds_te) == 0:
                    continue

                x_tr, y_tr = pooled_features(ds_tr)
                x_te, y_te = pooled_features(ds_te)
                pred = fit_predict_lr(x_tr, y_tr, x_te)
                acc = float(accuracy_score(y_te, pred))
                f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
                fold_accs.append(acc)
                fold_f1s.append(f1)

                fold_rows.append({
                    "feature_type": feature_type,
                    "task": task,
                    "held_out_subject": held_out,
                    "n_classes": n_classes,
                    "acc": acc,
                    "f1_macro": f1,
                    "eval": "LOSO_NULL",
                    "model": "lr_mean_std",
                })

    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        return folds, pd.DataFrame(columns=[
            "feature_type", "task", "model", "eval", "n_folds",
            "acc_mean", "acc_ci95_low", "acc_ci95_high",
            "f1_mean", "f1_ci95_low", "f1_ci95_high",
        ])

    summary_rows = []
    for (feature_type, task), g in folds.groupby(["feature_type", "task"]):
        acc_m, acc_lo, acc_hi = bootstrap_ci(g["acc"].tolist(), n_boot=2000, seed=11)
        f1_m, f1_lo, f1_hi = bootstrap_ci(g["f1_macro"].tolist(), n_boot=2000, seed=17)
        summary_rows.append({
            "feature_type": feature_type,
            "task": task,
            "model": "lr_mean_std",
            "eval": "LOSO_NULL",
            "n_folds": int(len(g)),
            "acc_mean": acc_m,
            "acc_ci95_low": acc_lo,
            "acc_ci95_high": acc_hi,
            "f1_mean": f1_m,
            "f1_ci95_low": f1_lo,
            "f1_ci95_high": f1_hi,
        })

    return folds, pd.DataFrame(summary_rows)


def run_eeg_8_8(preprocessed_root: Path, feature_types: list[str], tasks: list[str]) -> pd.DataFrame:
    set_preprocessed_root(preprocessed_root)
    rows = []
    combo_items = [(ft, tk) for ft in feature_types for tk in tasks]
    for feature_type, task in tqdm(combo_items, desc="EEG 8-8 tasks", unit="task"):
            ds_tr = build_eeg_dataset(SPLIT_8_TRAIN, feature_type, task)
            ds_te = build_eeg_dataset(SPLIT_8_TEST, feature_type, task)
            if len(ds_tr) == 0 or len(ds_te) == 0:
                continue
            x_tr, y_tr = pooled_features(ds_tr)
            x_te, y_te = pooled_features(ds_te)
            pred = fit_predict_lr(x_tr, y_tr, x_te)
            rows.append({
                "feature_type": feature_type,
                "task": task,
                "model": "lr_mean_std",
                "eval": "SPLIT_8_8_NULL",
                "n_test_trials": int(len(y_te)),
                "acc": float(accuracy_score(y_te, pred)),
                "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            })
    return pd.DataFrame(rows)


def _meta_label(df: pd.DataFrame, task: str) -> pd.Series:
    if task in ("phoneme_identity", "place", "manner", "voicing"):
        values = df[TASK_CONFIGS[task]["label_col"]].astype(str).str.lower()
    elif task == "category":
        p1 = df["phoneme1"].astype(str).str.lower()
        values = p1.map(lambda p: "consonant" if p in {"b", "p", "d", "t", "s", "z"} else ("vowel" if p in {"a", "e", "i", "o", "u"} else "n/a"))
    elif task == "complexity":
        mapping = {
            "single_phoneme_perceived": "single_phoneme",
            "single_phoneme_spoken": "single_phoneme",
            "cv_pairs": "diphone",
            "vc_pairs": "diphone",
            "cvc_real_words": "triphone",
            "cvc_pseudowords": "triphone",
        }
        values = df["task_type"].astype(str).map(mapping).fillna("n/a")
    else:
        raise ValueError(task)

    # Normalize known voicing variants from metadata
    values = values.replace({"yes": "voiced", "no": "unvoiced"})
    return values


def load_subject_metadata(preprocessed_root: Path, subjects: list[str]) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        ses = "ses-01" if sub.startswith("sub-P") else "ses-02"
        base = preprocessed_root / "erp" / sub / ses
        if not base.exists():
            continue
        for meta_file in sorted(base.glob("*_metadata.csv")):
            task_type = meta_file.name.replace("_metadata.csv", "")
            try:
                df = pd.read_csv(meta_file, keep_default_na=False)
            except Exception:
                continue
            if df.empty:
                continue
            df["subject"] = sub
            df["task_type"] = task_type
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["tms_condition"] = out["tms_condition"].astype(str).str.upper()
    return out


def build_acoustic_xy(df: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    y_raw = _meta_label(df, task)
    classes = TASK_CONFIGS[task]["classes"]
    valid = y_raw.isin(classes)
    dfv = df.loc[valid].copy()
    y_raw = y_raw.loc[valid]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(class_to_idx).astype(np.int64).to_numpy()

    feat_df = pd.DataFrame({
        "phoneme1": dfv["phoneme1"].astype(str).str.lower(),
        "phoneme2": dfv.get("phoneme2", "n/a").astype(str).str.lower(),
        "phoneme3": dfv.get("phoneme3", "n/a").astype(str).str.lower(),
        "task_type": dfv["task_type"].astype(str).str.lower(),
        "word_type": dfv.get("word_type", "n/a").astype(str).str.lower(),
    })
    records = feat_df.to_dict(orient="records")
    vec = DictVectorizer(sparse=False)
    x = vec.fit_transform(records)
    return x, y


def run_acoustic_loso(preprocessed_root: Path, tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_meta = load_subject_metadata(preprocessed_root, STUDY2_ALL)
    if all_meta.empty:
        return pd.DataFrame(), pd.DataFrame()
    all_meta = all_meta[all_meta["tms_condition"] == "NULL"].reset_index(drop=True)

    fold_rows = []
    for task in tqdm(tasks, desc="Acoustic LOSO tasks", unit="task"):
        subject_bar = tqdm(STUDY2_ALL, desc=f"Acoustic LOSO {task}", unit="sub", leave=False)
        for held_out in subject_bar:
            tr = all_meta[all_meta["subject"] != held_out].copy()
            te = all_meta[all_meta["subject"] == held_out].copy()
            if tr.empty or te.empty:
                continue

            y_tr_raw = _meta_label(tr, task)
            y_te_raw = _meta_label(te, task)
            classes = TASK_CONFIGS[task]["classes"]
            tr = tr[y_tr_raw.isin(classes)].copy()
            te = te[y_te_raw.isin(classes)].copy()
            if tr.empty or te.empty:
                continue

            # Shared one-hot vocabulary from train split only
            feat_cols = ["phoneme1", "phoneme2", "phoneme3", "task_type", "word_type"]
            for c in feat_cols:
                tr[c] = tr[c].astype(str).str.lower()
                te[c] = te[c].astype(str).str.lower()
            vec = DictVectorizer(sparse=False)
            x_tr = vec.fit_transform(tr[feat_cols].to_dict(orient="records"))
            x_te = vec.transform(te[feat_cols].to_dict(orient="records"))

            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_tr = _meta_label(tr, task).map(class_to_idx).astype(np.int64).to_numpy()
            y_te = _meta_label(te, task).map(class_to_idx).astype(np.int64).to_numpy()

            pred = fit_predict_lr(x_tr, y_tr, x_te)
            fold_rows.append({
                "task": task,
                "held_out_subject": held_out,
                "model": "acoustic_lr",
                "eval": "LOSO_NULL",
                "n_test_trials": int(len(y_te)),
                "acc": float(accuracy_score(y_te, pred)),
                "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            })

    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    if not folds.empty:
        for task, g in folds.groupby("task"):
            acc_m, acc_lo, acc_hi = bootstrap_ci(g["acc"].tolist(), n_boot=2000, seed=23)
            f1_m, f1_lo, f1_hi = bootstrap_ci(g["f1_macro"].tolist(), n_boot=2000, seed=29)
            summary_rows.append({
                "task": task,
                "model": "acoustic_lr",
                "eval": "LOSO_NULL",
                "n_folds": int(len(g)),
                "acc_mean": acc_m,
                "acc_ci95_low": acc_lo,
                "acc_ci95_high": acc_hi,
                "f1_mean": f1_m,
                "f1_ci95_low": f1_lo,
                "f1_ci95_high": f1_hi,
            })

    return folds, pd.DataFrame(summary_rows)


def run_acoustic_8_8(preprocessed_root: Path, tasks: list[str]) -> pd.DataFrame:
    all_meta = load_subject_metadata(preprocessed_root, STUDY2_ALL)
    if all_meta.empty:
        return pd.DataFrame()
    all_meta = all_meta[all_meta["tms_condition"] == "NULL"].reset_index(drop=True)

    tr = all_meta[all_meta["subject"].isin(SPLIT_8_TRAIN)].copy()
    te = all_meta[all_meta["subject"].isin(SPLIT_8_TEST)].copy()

    rows = []
    feat_cols = ["phoneme1", "phoneme2", "phoneme3", "task_type", "word_type"]
    for task in tqdm(tasks, desc="Acoustic 8-8 tasks", unit="task"):
        y_tr_raw = _meta_label(tr, task)
        y_te_raw = _meta_label(te, task)
        classes = TASK_CONFIGS[task]["classes"]

        trk = tr[y_tr_raw.isin(classes)].copy()
        tek = te[y_te_raw.isin(classes)].copy()
        if trk.empty or tek.empty:
            continue

        for c in feat_cols:
            trk[c] = trk[c].astype(str).str.lower()
            tek[c] = tek[c].astype(str).str.lower()

        vec = DictVectorizer(sparse=False)
        x_tr = vec.fit_transform(trk[feat_cols].to_dict(orient="records"))
        x_te = vec.transform(tek[feat_cols].to_dict(orient="records"))

        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_tr = _meta_label(trk, task).map(class_to_idx).astype(np.int64).to_numpy()
        y_te = _meta_label(tek, task).map(class_to_idx).astype(np.int64).to_numpy()

        pred = fit_predict_lr(x_tr, y_tr, x_te)
        rows.append({
            "task": task,
            "model": "acoustic_lr",
            "eval": "SPLIT_8_8_NULL",
            "n_test_trials": int(len(y_te)),
            "acc": float(accuracy_score(y_te, pred)),
            "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        })

    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_root = Path(args.preprocessed_root)

    eeg_loso_folds, eeg_loso_summary = run_eeg_loso(base_root, FEATURE_TYPES, TASKS)
    eeg_8_8 = run_eeg_8_8(base_root, FEATURE_TYPES, TASKS)

    ac_loso_folds, ac_loso_summary = run_acoustic_loso(base_root, TASKS)
    ac_8_8 = run_acoustic_8_8(base_root, TASKS)

    eeg_loso_folds.to_csv(RESULTS_DIR / "null_only_eeg_controls_loso.csv", index=False)
    eeg_8_8.to_csv(RESULTS_DIR / "null_only_eeg_controls_8_8.csv", index=False)
    ac_loso_folds.to_csv(RESULTS_DIR / "acoustic_baseline_controls_loso.csv", index=False)
    ac_8_8.to_csv(RESULTS_DIR / "acoustic_baseline_controls_8_8.csv", index=False)

    # Compact summaries for manuscript ingestion
    pd.concat([
        eeg_loso_summary.assign(domain="eeg"),
        ac_loso_summary.assign(domain="acoustic"),
    ], ignore_index=True).to_csv(RESULTS_DIR / "controls_loso_summary.csv", index=False)

    # Optional wideband ERP comparison (0.5-100Hz root)
    if args.wideband_root:
        wb_root = Path(args.wideband_root)
        if wb_root.exists():
            _, wb_summary = run_eeg_loso(wb_root, ["erp"], TASKS)
            _, base_erp_summary = run_eeg_loso(base_root, ["erp"], TASKS)
            merged = base_erp_summary.merge(
                wb_summary,
                on=["feature_type", "task", "model", "eval"],
                suffixes=("_base40", "_wideband"),
            )
            merged.to_csv(RESULTS_DIR / "wideband_erp_control_comparison.csv", index=False)

    print(f"Saved {RESULTS_DIR / 'null_only_eeg_controls_loso.csv'}")
    print(f"Saved {RESULTS_DIR / 'null_only_eeg_controls_8_8.csv'}")
    print(f"Saved {RESULTS_DIR / 'acoustic_baseline_controls_loso.csv'}")
    print(f"Saved {RESULTS_DIR / 'acoustic_baseline_controls_8_8.csv'}")
    print(f"Saved {RESULTS_DIR / 'controls_loso_summary.csv'}")
    if args.wideband_root:
        wb_out = RESULTS_DIR / "wideband_erp_control_comparison.csv"
        if wb_out.exists():
            print(f"Saved {wb_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run must-have control evaluations")
    parser.add_argument("--preprocessed-root", type=str, default=str(ROOT / "preprocessed"))
    parser.add_argument("--wideband-root", type=str, default=None)
    args = parser.parse_args()
    run(args)
