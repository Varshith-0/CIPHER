"""
CIPHER — Perception vs production modality comparison.

For Study 2 single-phoneme data, compares classification accuracy between:
  1. Perception-only model
  2. Production-only model (if data exists)
  3. Merged model (both combined)

Uses McNemar's test for statistical comparison.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from models.dataset import EEGDataset, TASK_CONFIGS
from models.model import ConformerDecoder, GRUDecoder
from evaluate.eval_metrics import load_model, predict, compute_metrics

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models_out"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

MODALITIES = {
    "perceived": ["single_phoneme_perceived"],
    "spoken": ["single_phoneme_spoken"],
    "merged": ["single_phoneme_perceived", "single_phoneme_spoken"],
}

MODALITY_TASKS = [
    "phoneme_identity", "place", "manner", "voicing", "category",
]


def run_modality_analysis(
    study2_val: list[str],
    feature_types: list[str] = ("erp", "dda"),
    subsample: float = 1.0,
) -> dict:
    """
    Compare perception vs production decoding accuracy.

    Returns dict with: accuracy_df, mcnemar_df.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_rows = []
    per_phoneme_data = {}  # for plotting

    for ft in feature_types:
        for task in MODALITY_TASKS:
            for mod_key, tt_filter in MODALITIES.items():
                model_dir = MODELS_ROOT / ft / task / f"modality_{mod_key}"
                model, cfg = load_model(model_dir, device)
                if model is None:
                    print(f"  SKIP modality ({ft}/{task}/{mod_key}): no model")
                    continue

                class_names = cfg.get(
                    "label_names", TASK_CONFIGS[task]["classes"]
                )

                # Evaluate on study2 val subjects using the same modality filter
                ds = EEGDataset(
                    subjects=study2_val,
                    feature_type=ft,
                    classification_task=task,
                    task_type_filter=tt_filter,
                    augment=False,
                    subsample=subsample,
                )
                if len(ds) == 0:
                    del model
                    continue

                logits, preds, labels = predict(model, ds, device)
                metrics = compute_metrics(labels, preds, logits, class_names)

                acc_rows.append({
                    "feature_type": ft,
                    "task": task,
                    "modality": mod_key,
                    "top1_acc": metrics["top1_acc"],
                    "f1_macro": metrics["f1_macro"],
                    "n_samples": metrics["n_samples"],
                })

                # Store per-class data for bar plots
                key = (ft, task)
                if key not in per_phoneme_data:
                    per_phoneme_data[key] = {}
                per_phoneme_data[key][mod_key] = {
                    "class_names": class_names,
                    "f1_per_class": metrics["f1_per_class"],
                }

                # Store predictions for McNemar's test
                per_phoneme_data[key].setdefault("_preds", {})[mod_key] = (preds, labels)

                del model
                torch.cuda.empty_cache()

    acc_df = pd.DataFrame(acc_rows)

    # ---- McNemar's test: perceived vs produced ----
    mcnemar_rows = []
    for (ft, task), data in per_phoneme_data.items():
        preds_dict = data.get("_preds", {})
        if "perceived" in preds_dict and "spoken" in preds_dict:
            p_preds, p_labels = preds_dict["perceived"]
            s_preds, s_labels = preds_dict["spoken"]

            # McNemar's requires same samples — use the smaller set
            n = min(len(p_preds), len(s_preds))
            if n == 0:
                continue

            p_correct = (p_preds[:n] == p_labels[:n])
            s_correct = (s_preds[:n] == s_labels[:n])

            # Contingency: a=both correct, b=only perceived, c=only spoken, d=both wrong
            b = int(np.sum(p_correct & ~s_correct))
            c = int(np.sum(~p_correct & s_correct))

            # McNemar's chi-squared
            if b + c > 0:
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = 1 - stats.chi2.cdf(chi2, df=1)
            else:
                chi2 = 0.0
                p_value = 1.0

            mcnemar_rows.append({
                "feature_type": ft,
                "task": task,
                "n_samples": n,
                "perceived_only_correct": b,
                "spoken_only_correct": c,
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "significant_p05": p_value < 0.05,
            })

        elif "perceived" in preds_dict:
            # Only perception data exists — note it
            mcnemar_rows.append({
                "feature_type": ft,
                "task": task,
                "n_samples": len(preds_dict["perceived"][0]),
                "perceived_only_correct": -1,
                "spoken_only_correct": -1,
                "chi2": float("nan"),
                "p_value": float("nan"),
                "significant_p05": False,
            })

    mcnemar_df = pd.DataFrame(mcnemar_rows)

    # ---- Save tables ----
    tables_dir = RESULTS_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not acc_df.empty:
        acc_df.to_csv(tables_dir / "perception_vs_production.csv", index=False)
    if not mcnemar_df.empty:
        mcnemar_df.to_csv(tables_dir / "modality_mcnemar.csv", index=False)
        print("\n  McNemar's test results:")
        print(mcnemar_df.to_string(index=False))

    # ---- Plot side-by-side bar charts ----
    _plot_modality_comparison(per_phoneme_data)

    return {"accuracy_df": acc_df, "mcnemar_df": mcnemar_df}


def _plot_modality_comparison(per_phoneme_data: dict):
    """Plot F1 per class for perceived/spoken/merged side by side."""
    fig_dir = RESULTS_ROOT / "figures" / "modality_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for (ft, task), data in per_phoneme_data.items():
        modalities = [m for m in ["perceived", "spoken", "merged"] if m in data]
        if not modalities:
            continue

        class_names = data[modalities[0]]["class_names"]
        n_classes = len(class_names)

        x = np.arange(n_classes)
        width = 0.25
        colors = {"perceived": "#4C72B0", "spoken": "#DD8452", "merged": "#55A868"}

        fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.2), 5))

        for i, mod in enumerate(modalities):
            f1s = [data[mod]["f1_per_class"].get(cn, 0.0) for cn in class_names]
            offset = (i - (len(modalities) - 1) / 2) * width
            ax.bar(
                x + offset, f1s, width,
                label=mod.capitalize(),
                color=colors.get(mod, f"C{i}"),
                alpha=0.85,
            )

        ax.set_xlabel("Class")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Modality Comparison: {task} ({ft.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        save_path = fig_dir / f"modality_{ft}_{task}.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
