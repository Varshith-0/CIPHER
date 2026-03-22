"""
CIPHER — Evaluation metrics: accuracy, F1, confusion matrix, WER.

Loads trained ConformerDecoder (or legacy GRU) models from ~/cipher/models_out/
and evaluates on held-out data from ~/cipher/preprocessed/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from models.dataset import EEGDataset, TASK_CONFIGS, CTC_VOCAB_SIZE
from models.model import ConformerDecoder, GRUDecoder

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models_out"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

# ===========================================================================
# Model loading
# ===========================================================================

def load_model(model_dir: Path, device: torch.device) -> tuple:
    """
    Load a trained model and its config from a model directory.
    Supports both ConformerDecoder and legacy GRUDecoder.

    Returns (model, config_dict) or (None, None) if not found.
    """
    config_path = model_dir / "config.json"
    ckpt_path = model_dir / "best_model.pt"

    if not config_path.exists() or not ckpt_path.exists():
        return None, None

    with open(config_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "gru")

    if model_type == "conformer":
        # Rebuild task_n_classes from config
        if cfg.get("multi_task") and "task_n_classes" in cfg:
            task_n_classes = cfg["task_n_classes"]
        else:
            primary_task = cfg.get("primary_task", "phoneme_identity")
            task_n_classes = {primary_task: cfg["n_classes"]}

        model = ConformerDecoder(
            input_dim=cfg["input_dim"],
            task_n_classes=task_n_classes,
            d_model=cfg.get("d_model", 192),
            n_conformer_blocks=cfg.get("n_conformer_blocks", 4),
            n_heads=cfg.get("n_heads", 4),
            conv_channels=cfg.get("conv_channels", 64),
            conv_kernel=cfg.get("conv_kernel", 15),
            dropout=cfg.get("dropout", 0.3),
            ctc_vocab_size=CTC_VOCAB_SIZE if cfg.get("ctc") else None,
        )
    else:
        model = GRUDecoder(
            input_dim=cfg["input_dim"],
            n_classes=cfg["n_classes"],
            hidden_size=cfg.get("hidden_size", 256),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.3),
        )

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, cfg


# ===========================================================================
# Inference
# ===========================================================================

def predict(
    model: torch.nn.Module,
    dataset: EEGDataset,
    device: torch.device,
    batch_size: int = 64,
    task: str | None = None,
) -> tuple:
    """
    Run model over a dataset.
    For ConformerDecoder, extracts logits for the specified task head
    (defaults to the primary classification task of the dataset).

    Returns (all_logits, all_preds, all_labels) as numpy arrays.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    logits_list, preds_list, labels_list = [], [], []
    task = task or dataset.classification_task

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]

            raw_model = model.module if hasattr(model, "module") else model

            if isinstance(raw_model, ConformerDecoder):
                out = model(x, tasks=[task])
                logits = out.get(task)
                if logits is None:
                    # Fallback: try first available head
                    logits = next(iter(out.values()))
            else:
                logits = model(x)

            logits_list.append(logits.cpu().numpy())
            preds_list.append(logits.argmax(dim=1).cpu().numpy())

            # Extract labels
            if isinstance(y, dict):
                labels_list.append(y[task].numpy())
            else:
                labels_list.append(y.numpy())

    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


# ===========================================================================
# Metrics computation
# ===========================================================================

def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute top-1 accuracy, top-3 accuracy, macro F1."""
    top1 = accuracy_score(labels, preds)

    # Top-3 accuracy
    if logits.shape[1] >= 3:
        top3_preds = np.argsort(logits, axis=1)[:, -3:]
        top3 = np.mean([labels[i] in top3_preds[i] for i in range(len(labels))])
    else:
        top3 = top1

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(
        labels, preds, average=None, labels=range(len(class_names)),
        zero_division=0,
    )

    return {
        "top1_acc": round(float(top1), 4),
        "top3_acc": round(float(top3), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_per_class": {
            class_names[i]: round(float(f1_per_class[i]), 4)
            for i in range(len(class_names))
        },
        "n_samples": int(len(labels)),
    }


def compute_wer(
    labels_seq: list[list[str]],
    preds_seq: list[list[str]],
) -> float:
    """
    Compute word error rate. Each element is a list of phoneme strings
    (treated as a "word" sequence for WER computation via jiwer).
    """
    try:
        import jiwer
    except ImportError:
        # Fallback: manual token-level WER
        return _manual_wer(labels_seq, preds_seq)

    refs = [" ".join(s) for s in labels_seq]
    hyps = [" ".join(s) for s in preds_seq]
    return jiwer.wer(refs, hyps)


def _manual_wer(refs: list[list[str]], hyps: list[list[str]]) -> float:
    """Levenshtein-based WER fallback."""
    total_ref = 0
    total_err = 0
    for ref, hyp in zip(refs, hyps):
        total_ref += len(ref)
        total_err += _levenshtein(ref, hyp)
    return total_err / max(total_ref, 1)


def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
    return dp[m]


# ===========================================================================
# Confusion matrix plot
# ===========================================================================

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "",
):
    """Save a confusion matrix heatmap as PNG."""
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    # Normalise rows (recall-based)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7),
                                     max(5, len(class_names) * 0.6)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or save_path.stem)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ===========================================================================
# Training curves plot
# ===========================================================================

def plot_training_curves(model_dir: Path, save_path: Path):
    """Plot loss and accuracy training curves from training_log.csv."""
    log_path = model_dir / "training_log.csv"
    if not log_path.exists():
        return

    df = pd.read_csv(log_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(df["epoch"], df["train_acc"], label="train")
    if "val_acc" in df.columns:
        ax2.plot(df["epoch"], df["val_acc"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ===========================================================================
# Full cross-dataset evaluation
# ===========================================================================

def evaluate_all_models(
    feature_types: list[str],
    tasks: list[str],
    tms_keys: dict,
    study2_val: list[str],
    study1_test: list[str],
    subsample: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate every trained model on Study 2 val and Study 1 test sets.
    Returns a DataFrame of results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for ft in feature_types:
        for task in tasks:
            for tms_key, tms_val in tms_keys.items():
                model_dir = MODELS_ROOT / ft / task / tms_key
                model, cfg = load_model(model_dir, device)
                if model is None:
                    print(f"  SKIP (no model): {ft}/{task}/{tms_key}")
                    continue

                class_names = cfg.get("label_names", TASK_CONFIGS[task]["classes"])

                for split_name, subjects in [
                    ("study2_val", study2_val),
                    ("study1_test", study1_test),
                ]:
                    ds = EEGDataset(
                        subjects=subjects,
                        feature_type=ft,
                        classification_task=task,
                        tms_condition=tms_val,
                        augment=False,
                        subsample=subsample,
                        temporal_stride=4 if ft == "dda" else 1,
                    )
                    if len(ds) == 0:
                        print(f"  SKIP (no data): {ft}/{task}/{tms_key}/{split_name}")
                        continue

                    logits, preds, labels = predict(model, ds, device)
                    metrics = compute_metrics(labels, preds, logits, class_names)

                    # Confusion matrix
                    cm_dir = RESULTS_ROOT / "figures" / "confusion_matrices"
                    cm_path = cm_dir / f"{ft}_{task}_{tms_key}_{split_name}.png"
                    plot_confusion_matrix(
                        labels, preds, class_names, cm_path,
                        title=f"{ft} / {task} / {tms_key} / {split_name}",
                    )

                    # Training curves (once per model)
                    tc_dir = RESULTS_ROOT / "figures" / "training_curves"
                    tc_path = tc_dir / f"{ft}_{task}_{tms_key}.png"
                    if not tc_path.exists():
                        plot_training_curves(model_dir, tc_path)

                    row = {
                        "feature_type": ft,
                        "task": task,
                        "tms_condition": tms_key,
                        "split": split_name,
                        **metrics,
                    }
                    rows.append(row)
                    print(f"  {ft}/{task}/{tms_key}/{split_name}: "
                          f"top1={metrics['top1_acc']:.3f} "
                          f"F1={metrics['f1_macro']:.3f} "
                          f"n={metrics['n_samples']}")

                del model
                torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def compute_wer_for_triphones(
    feature_types: list[str],
    tms_keys: dict,
    study2_val: list[str],
    study1_test: list[str],
    subsample: float = 1.0,
) -> pd.DataFrame:
    """
    Compute WER on triphone data (CVC real words + pseudowords).
    The "word" is the 3-phoneme sequence; WER measures phoneme-level errors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for ft in feature_types:
        model_dir = MODELS_ROOT / ft / "phoneme_identity" / "null"
        model, cfg = load_model(model_dir, device)
        if model is None:
            continue

        class_names = cfg.get("label_names", TASK_CONFIGS["phoneme_identity"]["classes"])

        for split_name, subjects in [
            ("study2_val", study2_val),
            ("study1_test", study1_test),
        ]:
            for word_type_filter in ["cvc_real_words", "cvc_pseudowords"]:
                ds = EEGDataset(
                    subjects=subjects,
                    feature_type=ft,
                    classification_task="phoneme_identity",
                    tms_condition="NULL",
                    task_type_filter=[word_type_filter],
                    augment=False,
                    subsample=subsample,
                    temporal_stride=4 if ft == "dda" else 1,
                )
                if len(ds) == 0:
                    continue

                logits, preds, labels = predict(model, ds, device)

                # Build phoneme sequences: each trial = single-phoneme classification
                # WER is computed at the trial level (each trial is one phoneme token)
                ref_seqs = [[class_names[l]] for l in labels]
                hyp_seqs = [[class_names[p]] for p in preds]
                wer = compute_wer(ref_seqs, hyp_seqs)

                rows.append({
                    "feature_type": ft,
                    "word_type": word_type_filter,
                    "split": split_name,
                    "wer": round(wer, 4),
                    "n_samples": len(labels),
                })

        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# ===========================================================================
# Ensemble ERP + DDA evaluation
# ===========================================================================

def evaluate_ensemble(
    tasks: list[str],
    tms_keys: dict,
    study2_val: list[str],
    study1_test: list[str],
    subsample: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate ERP+DDA logit-averaging ensemble for each task × TMS.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for task in tasks:
        for tms_key, tms_val in tms_keys.items():
            # Load both models
            erp_dir = MODELS_ROOT / "erp" / task / tms_key
            dda_dir = MODELS_ROOT / "dda" / task / tms_key
            erp_model, erp_cfg = load_model(erp_dir, device)
            dda_model, dda_cfg = load_model(dda_dir, device)
            if erp_model is None or dda_model is None:
                print(f"  SKIP ensemble: {task}/{tms_key} (missing model)")
                continue

            class_names = erp_cfg.get("label_names", TASK_CONFIGS[task]["classes"])

            for split_name, subjects in [
                ("study2_val", study2_val),
                ("study1_test", study1_test),
            ]:
                erp_ds = EEGDataset(
                    subjects=subjects, feature_type="erp",
                    classification_task=task, tms_condition=tms_val,
                    augment=False, subsample=subsample,
                )
                dda_ds = EEGDataset(
                    subjects=subjects, feature_type="dda",
                    classification_task=task, tms_condition=tms_val,
                    augment=False, subsample=subsample,
                )
                if len(erp_ds) == 0 or len(dda_ds) == 0:
                    continue

                erp_logits, _, erp_labels = predict(erp_model, erp_ds, device,
                                                     task=task)
                dda_logits, _, dda_labels = predict(dda_model, dda_ds, device,
                                                     task=task)

                # Align by minimum count (should be same)
                n = min(len(erp_logits), len(dda_logits))
                avg_logits = (erp_logits[:n] + dda_logits[:n]) / 2.0
                avg_preds = avg_logits.argmax(axis=1)
                labels = erp_labels[:n]

                metrics = compute_metrics(labels, avg_preds, avg_logits, class_names)

                # Save confusion matrix
                cm_dir = RESULTS_ROOT / "figures" / "confusion_matrices"
                cm_path = cm_dir / f"ensemble_{task}_{tms_key}_{split_name}.png"
                plot_confusion_matrix(
                    labels, avg_preds, class_names, cm_path,
                    title=f"Ensemble / {task} / {tms_key} / {split_name}",
                )

                row = {
                    "feature_type": "ensemble",
                    "task": task,
                    "tms_condition": tms_key,
                    "split": split_name,
                    **metrics,
                }
                rows.append(row)
                print(f"  ensemble/{task}/{tms_key}/{split_name}: "
                      f"top1={metrics['top1_acc']:.3f} "
                      f"F1={metrics['f1_macro']:.3f} "
                      f"n={metrics['n_samples']}")

            del erp_model, dda_model
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)
