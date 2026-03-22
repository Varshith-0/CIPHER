"""
CIPHER — TMS condition analysis.

Compares decoding accuracy across TMS conditions (NULL, LipTMS, TongueTMS).
Tests the motor-speech hypothesis:
  - Bilabial consonants (/b/, /p/) should be FACILITATED by LipTMS
  - Alveolar consonants (/d/, /t/) should be FACILITATED by TongueTMS
  - Cross-conditions should HINDER decoding
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
from evaluate.eval_metrics import load_model, predict

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models_out"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

# Phoneme groupings for TMS analysis
BILABIAL_PHONEMES = {"b", "p"}
ALVEOLAR_PHONEMES = {"d", "t", "s", "z"}
ALL_CONSONANTS = BILABIAL_PHONEMES | ALVEOLAR_PHONEMES

TMS_CONDITIONS = {
    "null": "NULL",
    "lip": "LipTMS",
    "tongue": "TongueTMS",
}


def run_tms_analysis(
    study2_subjects: list[str],
    feature_types: list[str] = ("erp", "dda"),
    subsample: float = 1.0,
) -> dict:
    """
    Run the full TMS condition comparison.

    Returns dict with results DataFrames and saved figure paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for ft in feature_types:
        for tms_key, tms_val in TMS_CONDITIONS.items():
            model_dir = MODELS_ROOT / ft / "phoneme_identity" / tms_key
            model, cfg = load_model(model_dir, device)
            if model is None:
                print(f"  SKIP TMS analysis ({ft}/{tms_key}): no model")
                continue

            class_names = cfg.get(
                "label_names", TASK_CONFIGS["phoneme_identity"]["classes"]
            )

            ds = EEGDataset(
                subjects=study2_subjects,
                feature_type=ft,
                classification_task="phoneme_identity",
                tms_condition=tms_val,
                augment=False,
                subsample=subsample,
            )
            if len(ds) == 0:
                del model
                continue

            _, preds, labels = predict(model, ds, device)

            # Per-phoneme accuracy
            for phoneme in ALL_CONSONANTS:
                if phoneme not in class_names:
                    continue
                cls_idx = class_names.index(phoneme)
                mask = labels == cls_idx
                if mask.sum() == 0:
                    continue
                acc = float(np.mean(preds[mask] == labels[mask]))
                place = "bilabial" if phoneme in BILABIAL_PHONEMES else "alveolar"

                rows.append({
                    "feature_type": ft,
                    "tms_condition": tms_key,
                    "phoneme": phoneme,
                    "place": place,
                    "accuracy": round(acc, 4),
                    "n_trials": int(mask.sum()),
                })

            del model
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        print("  TMS analysis: no data.")
        return {"results_df": results_df, "anova_df": pd.DataFrame()}

    # ---- ANOVA: accuracy across 3 TMS conditions per phoneme class ----
    anova_rows = []
    for ft in feature_types:
        for place in ["bilabial", "alveolar"]:
            subset = results_df[
                (results_df["feature_type"] == ft) &
                (results_df["place"] == place)
            ]
            groups = []
            for tms_key in TMS_CONDITIONS:
                group = subset[subset["tms_condition"] == tms_key]["accuracy"].values
                if len(group) > 0:
                    groups.append(group)

            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                anova_rows.append({
                    "feature_type": ft,
                    "place_of_articulation": place,
                    "f_stat": round(float(f_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "n_groups": len(groups),
                    "significant_p05": p_value < 0.05,
                })

    anova_df = pd.DataFrame(anova_rows)

    # ---- Save tables ----
    tables_dir = RESULTS_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "tms_per_phoneme_accuracy.csv", index=False)
    if not anova_df.empty:
        anova_df.to_csv(tables_dir / "tms_anova.csv", index=False)
        print("\n  TMS ANOVA results:")
        print(anova_df.to_string(index=False))

    # ---- Plot grouped bar chart ----
    _plot_tms_barplots(results_df)

    return {"results_df": results_df, "anova_df": anova_df}


def _plot_tms_barplots(df: pd.DataFrame):
    """Plot accuracy per phoneme × TMS condition as grouped bar chart."""
    fig_dir = RESULTS_ROOT / "figures" / "tms_barplots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for ft in df["feature_type"].unique():
        ft_df = df[df["feature_type"] == ft]
        phonemes = sorted(ft_df["phoneme"].unique())
        tms_conds = sorted(ft_df["tms_condition"].unique())

        if not phonemes or not tms_conds:
            continue

        x = np.arange(len(phonemes))
        width = 0.25
        n_bars = len(tms_conds)

        fig, ax = plt.subplots(figsize=(max(8, len(phonemes) * 1.2), 5))

        colors = {"null": "#4C72B0", "lip": "#DD8452", "tongue": "#55A868"}

        for i, tms in enumerate(tms_conds):
            accs = []
            for ph in phonemes:
                row = ft_df[(ft_df["phoneme"] == ph) & (ft_df["tms_condition"] == tms)]
                accs.append(row["accuracy"].values[0] if len(row) > 0 else 0.0)

            offset = (i - (n_bars - 1) / 2) * width
            bars = ax.bar(
                x + offset, accs, width,
                label=tms.upper(),
                color=colors.get(tms, f"C{i}"),
                alpha=0.85,
            )

        ax.set_xlabel("Phoneme")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"TMS Condition × Phoneme Accuracy ({ft.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"/{p}/" for p in phonemes])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.axhline(y=1.0 / len(phonemes), color="gray", linestyle="--",
                    alpha=0.5, label="chance")

        # Annotate place of articulation
        for j, ph in enumerate(phonemes):
            place = "B" if ph in BILABIAL_PHONEMES else "A"
            ax.annotate(
                place, (j, -0.08), ha="center", fontsize=8, color="gray",
                annotation_clip=False,
            )

        plt.tight_layout()
        fig.savefig(str(fig_dir / f"tms_phoneme_accuracy_{ft}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_dir / f'tms_phoneme_accuracy_{ft}.png'}")
