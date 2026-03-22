"""
CIPHER — Real word vs pseudoword analysis.

Compares phoneme classification accuracy and ERP amplitude between
CVC real words and CVC pseudowords. Tests the effortful-processing
hypothesis: pseudowords may produce stronger/more distinct neural signals.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from scipy import stats

from models.dataset import EEGDataset, TASK_CONFIGS, PREPROCESSED_ROOT
from models.model import ConformerDecoder, GRUDecoder
from evaluate.eval_metrics import load_model, predict, compute_metrics

mne.set_log_level("ERROR")

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models_out"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

# ERP analysis window for N200 component (200–400 ms post-stimulus)
# Epoch is -100 to +800 ms at 256 Hz
# Sample 0 = -100 ms → 200 ms = sample index (200+100)/1000*256 ≈ 77
# 400 ms = sample index (400+100)/1000*256 ≈ 128
N200_START = 77   # sample index at 256 Hz
N200_END = 128    # sample index at 256 Hz


def run_real_vs_pseudo_analysis(
    study2_subjects: list[str],
    feature_types: list[str] = ("erp", "dda"),
    subsample: float = 1.0,
) -> dict:
    """
    Run the full real-word vs pseudoword analysis.

    Returns dict with keys: accuracy_df, ttest_df, saved figures list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Step 1: Classification accuracy comparison ----
    acc_rows = []
    per_subject_acc = {"real": {}, "pseudo": {}}

    for ft in feature_types:
        model_dir = MODELS_ROOT / ft / "phoneme_identity" / "null"
        model, cfg = load_model(model_dir, device)
        if model is None:
            print(f"  SKIP real_vs_pseudo ({ft}): no phoneme_identity model")
            continue

        class_names = cfg.get("label_names", TASK_CONFIGS["phoneme_identity"]["classes"])

        for word_type, task_filter in [
            ("real", ["cvc_real_words"]),
            ("pseudo", ["cvc_pseudowords"]),
        ]:
            # Per-subject accuracy for paired t-test
            for sub in study2_subjects:
                ds = EEGDataset(
                    subjects=[sub],
                    feature_type=ft,
                    classification_task="phoneme_identity",
                    tms_condition="NULL",
                    task_type_filter=task_filter,
                    augment=False,
                    subsample=subsample,
                )
                if len(ds) == 0:
                    continue
                _, preds, labels = predict(model, ds, device)
                acc = float(np.mean(preds == labels))
                acc_rows.append({
                    "feature_type": ft,
                    "word_type": word_type,
                    "subject": sub,
                    "accuracy": round(acc, 4),
                    "n_trials": len(labels),
                })
                per_subject_acc.setdefault(ft, {}).setdefault(word_type, {})[sub] = acc

        del model
        torch.cuda.empty_cache()

    acc_df = pd.DataFrame(acc_rows)

    # ---- Step 2: Paired t-test across subjects ----
    ttest_rows = []
    for ft in feature_types:
        ft_data = per_subject_acc.get(ft, {})
        real_accs = ft_data.get("real", {})
        pseudo_accs = ft_data.get("pseudo", {})
        common_subs = sorted(set(real_accs.keys()) & set(pseudo_accs.keys()))

        if len(common_subs) >= 2:
            r_vals = [real_accs[s] for s in common_subs]
            p_vals = [pseudo_accs[s] for s in common_subs]
            t_stat, p_value = stats.ttest_rel(r_vals, p_vals)
            ttest_rows.append({
                "feature_type": ft,
                "n_subjects": len(common_subs),
                "mean_real_acc": round(float(np.mean(r_vals)), 4),
                "mean_pseudo_acc": round(float(np.mean(p_vals)), 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "significant_p05": p_value < 0.05,
            })

    ttest_df = pd.DataFrame(ttest_rows)

    # ---- Step 3: ERP amplitude analysis (N200 window) ----
    _erp_amplitude_analysis(study2_subjects)

    # ---- Save tables ----
    tables_dir = RESULTS_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not acc_df.empty:
        acc_df.to_csv(tables_dir / "real_vs_pseudo_accuracy.csv", index=False)
    if not ttest_df.empty:
        ttest_df.to_csv(tables_dir / "real_vs_pseudo_ttest.csv", index=False)
        print("\n  Real vs Pseudo t-test results:")
        print(ttest_df.to_string(index=False))

    return {"accuracy_df": acc_df, "ttest_df": ttest_df}


def _erp_amplitude_analysis(study2_subjects: list[str]):
    """
    Compare ERP amplitude in N200 window (200-400ms) between
    real words and pseudowords. Plot topographic difference map.
    """
    real_amps = []   # per-subject mean amplitude per channel
    pseudo_amps = []
    expected_nch = None  # track consistent channel count

    for sub in study2_subjects:
        ses = "ses-02"
        base_dir = PREPROCESSED_ROOT / "erp" / sub / ses

        for word_type, task_name, amp_list in [
            ("real", "cvc_real_words", real_amps),
            ("pseudo", "cvc_pseudowords", pseudo_amps),
        ]:
            fif_path = base_dir / f"{task_name}_epo.fif"
            if not fif_path.exists():
                continue

            epochs = mne.read_epochs(str(fif_path), verbose=False)
            data = epochs.get_data()  # (n_epochs, n_ch, n_times)
            if data.shape[2] <= N200_END:
                continue

            # Track expected channel count; skip subjects with mismatched channels
            if expected_nch is None:
                expected_nch = data.shape[1]
            elif data.shape[1] != expected_nch:
                continue

            # Mean absolute amplitude in N200 window, averaged across epochs
            n200_data = data[:, :, N200_START:N200_END]
            mean_amp = np.mean(np.abs(n200_data), axis=(0, 2))  # (n_ch,)
            amp_list.append(mean_amp)

    if not real_amps or not pseudo_amps:
        print("  SKIP ERP topomap: no ERP data for real/pseudo words")
        return

    # Average across subjects
    real_mean = np.mean(np.stack(real_amps), axis=0)
    pseudo_mean = np.mean(np.stack(pseudo_amps), axis=0)
    diff = pseudo_mean - real_mean  # positive = pseudo stronger

    # Create an MNE Info object for topographic plot
    # Use channel names from any available epochs file
    for sub in study2_subjects:
        fif_path = PREPROCESSED_ROOT / "erp" / sub / "ses-02" / "cvc_real_words_epo.fif"
        if fif_path.exists():
            epochs = mne.read_epochs(str(fif_path), verbose=False)
            info = epochs.info
            break
    else:
        print("  SKIP ERP topomap: no info object available")
        return

    # Plot
    fig_dir = RESULTS_ROOT / "figures" / "erp_topomaps"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Real words
    mne.viz.plot_topomap(real_mean, info, axes=axes[0], show=False)
    axes[0].set_title("Real Words\n(N200 mean |amp|)")

    # Pseudowords
    mne.viz.plot_topomap(pseudo_mean, info, axes=axes[1], show=False)
    axes[1].set_title("Pseudowords\n(N200 mean |amp|)")

    # Difference
    im, _ = mne.viz.plot_topomap(diff, info, axes=axes[2], show=False)
    axes[2].set_title("Pseudo − Real\n(difference)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    fig.savefig(str(fig_dir / "real_vs_pseudo_topomap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved topomap: {fig_dir / 'real_vs_pseudo_topomap.png'}")
