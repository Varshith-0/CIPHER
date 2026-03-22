#!/usr/bin/env python3
"""
CIPHER — Master evaluation script.

Runs all evaluation analyses on trained CIPHER models:
  1. Cross-dataset accuracy / F1 / confusion matrices / WER
  2. Real word vs pseudoword analysis
  3. TMS condition analysis
  4. Perception vs production modality comparison
  5. Summary report generation

Usage:
  python evaluate_all.py --dry-run                   # quick test
  python evaluate_all.py                              # full evaluation
  python evaluate_all.py --analysis metrics           # specific analysis
  python evaluate_all.py --analysis tms --analysis real_vs_pseudo
"""

import argparse
import random
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ===========================================================================
# Subject splits (must match train_all.py)
# ===========================================================================
STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]
STUDY2_VAL = ["sub-S04", "sub-S09", "sub-S14"]
STUDY2_TRAIN = [s for s in STUDY2_ALL if s not in STUDY2_VAL]
STUDY1_TEST = [f"sub-P{i:02d}" for i in range(1, 9)]

FEATURE_TYPES = ["erp", "dda"]
TASKS = [
    "phoneme_identity", "place", "manner",
    "voicing", "category", "complexity",
]
TMS_CONDITIONS = {
    "null": "NULL",
    "lip": "LipTMS",
    "tongue": "TongueTMS",
}

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

ALL_ANALYSES = ["metrics", "ensemble", "wer", "real_vs_pseudo", "tms", "modality"]


def main():
    parser = argparse.ArgumentParser(description="CIPHER full evaluation pipeline")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible evaluation",
    )
    parser.add_argument(
        "--analysis", action="append", default=None,
        choices=ALL_ANALYSES + ["all"],
        help="Which analysis to run (can be repeated). Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run on sub-P01 + sub-S01 only with 10%% data subsample",
    )
    parser.add_argument(
        "--v3", action="store_true",
        help="Enable v3 enhancements (TTA)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        # Keep evaluation robust even if torch is not available in CPU-only checks.
        pass

    analyses = args.analysis or ["all"]
    if "all" in analyses:
        analyses = ALL_ANALYSES

    subsample = 1.0
    study2_val = STUDY2_VAL
    study1_test = STUDY1_TEST
    study2_all = STUDY2_ALL

    if args.dry_run:
        study2_val = ["sub-S04"]
        study1_test = ["sub-P01"]
        study2_all = ["sub-S01", "sub-S04"]
        subsample = 0.1
        print("=== DRY RUN: limited subjects, 10% data ===\n")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 70,
        "CIPHER — Evaluation Summary Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    # ==================================================================
    # STEP 1: Cross-dataset evaluation (accuracy, F1, confusion matrices)
    # ==================================================================
    if "metrics" in analyses:
        print("=" * 60)
        print("STEP 1: Cross-dataset evaluation metrics")
        print("=" * 60)

        from evaluate.eval_metrics import evaluate_all_models

        metrics_df = evaluate_all_models(
            feature_types=FEATURE_TYPES,
            tasks=TASKS,
            tms_keys=TMS_CONDITIONS,
            study2_val=study2_val,
            study1_test=study1_test,
            subsample=subsample,
            v3=args.v3,
        )

        tables_dir = RESULTS_ROOT / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        if not metrics_df.empty:
            metrics_df.to_csv(
                tables_dir / "phoneme_accuracy_all_conditions.csv", index=False,
            )
            report_lines.append("CROSS-DATASET EVALUATION")
            report_lines.append("-" * 40)
            # Summarise best results
            for split in ["study2_val", "study1_test"]:
                split_df = metrics_df[metrics_df["split"] == split]
                if split_df.empty:
                    continue
                best = split_df.loc[split_df["top1_acc"].idxmax()]
                report_lines.append(
                    f"  Best {split}: top1={best['top1_acc']:.3f} "
                    f"F1={best['f1_macro']:.3f} "
                    f"({best['feature_type']}/{best['task']}/{best['tms_condition']})"
                )
            report_lines.append("")
        print()

    # ==================================================================
    # STEP 1a: Ensemble ERP + DDA evaluation
    # ==================================================================
    if "ensemble" in analyses:
        print("=" * 60)
        print("STEP 1a: Ensemble ERP + DDA evaluation")
        print("=" * 60)

        from evaluate.eval_metrics import evaluate_ensemble

        ens_df = evaluate_ensemble(
            tasks=TASKS,
            tms_keys=TMS_CONDITIONS,
            study2_val=study2_val,
            study1_test=study1_test,
            subsample=subsample,
            v3=args.v3,
        )
        tables_dir = RESULTS_ROOT / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        if not ens_df.empty:
            ens_df.to_csv(tables_dir / "ensemble_accuracy.csv", index=False)
            report_lines.append("ENSEMBLE ERP + DDA")
            report_lines.append("-" * 40)
            for split in ["study2_val", "study1_test"]:
                split_df = ens_df[ens_df["split"] == split]
                if split_df.empty:
                    continue
                best = split_df.loc[split_df["top1_acc"].idxmax()]
                report_lines.append(
                    f"  Best {split}: top1={best['top1_acc']:.3f} "
                    f"F1={best['f1_macro']:.3f} "
                    f"({best['task']}/{best['tms_condition']})"
                )
            report_lines.append("")
        print()

    # ==================================================================
    # STEP 1b: WER for triphone sequences
    # ==================================================================
    if "wer" in analyses:
        print("=" * 60)
        print("STEP 1b: Word error rate (triphones)")
        print("=" * 60)

        from evaluate.eval_metrics import compute_wer_for_triphones

        wer_df = compute_wer_for_triphones(
            feature_types=FEATURE_TYPES,
            tms_keys={"null": "NULL"},
            study2_val=study2_val,
            study1_test=study1_test,
            subsample=subsample,
            v3=args.v3,
        )
        tables_dir = RESULTS_ROOT / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        if not wer_df.empty:
            wer_df.to_csv(tables_dir / "wer_all_conditions.csv", index=False)
            report_lines.append("WORD ERROR RATE (TRIPHONES)")
            report_lines.append("-" * 40)
            for _, row in wer_df.iterrows():
                report_lines.append(
                    f"  {row['feature_type']}/{row['word_type']}/{row['split']}: "
                    f"WER={row['wer']:.3f} (n={row['n_samples']})"
                )
            report_lines.append("")
        print()

    # ==================================================================
    # STEP 2: Real word vs pseudoword analysis
    # ==================================================================
    if "real_vs_pseudo" in analyses:
        print("=" * 60)
        print("STEP 2: Real word vs pseudoword analysis")
        print("=" * 60)

        from evaluate.real_vs_pseudo import run_real_vs_pseudo_analysis

        rp_results = run_real_vs_pseudo_analysis(
            study2_subjects=study2_all,
            feature_types=FEATURE_TYPES,
            subsample=subsample,
        )
        ttest_df = rp_results.get("ttest_df")
        if ttest_df is not None and not ttest_df.empty:
            report_lines.append("REAL WORD vs PSEUDOWORD")
            report_lines.append("-" * 40)
            for _, row in ttest_df.iterrows():
                sig = "*" if row.get("significant_p05") else "n.s."
                report_lines.append(
                    f"  {row['feature_type']}: "
                    f"real={row['mean_real_acc']:.3f} "
                    f"pseudo={row['mean_pseudo_acc']:.3f} "
                    f"t={row['t_stat']:.3f} p={row['p_value']:.4f} {sig}"
                )
            report_lines.append("")
        print()

    # ==================================================================
    # STEP 3: TMS condition analysis
    # ==================================================================
    if "tms" in analyses:
        print("=" * 60)
        print("STEP 3: TMS condition analysis")
        print("=" * 60)

        from evaluate.tms_analysis import run_tms_analysis

        tms_results = run_tms_analysis(
            study2_subjects=study2_all,
            feature_types=FEATURE_TYPES,
            subsample=subsample,
        )
        anova_df = tms_results.get("anova_df")
        if anova_df is not None and not anova_df.empty:
            report_lines.append("TMS CONDITION ANALYSIS")
            report_lines.append("-" * 40)
            for _, row in anova_df.iterrows():
                sig = "*" if row.get("significant_p05") else "n.s."
                report_lines.append(
                    f"  {row['feature_type']}/{row['place_of_articulation']}: "
                    f"F={row['f_stat']:.3f} p={row['p_value']:.4f} {sig}"
                )
            report_lines.append("")
        print()

    # ==================================================================
    # STEP 4: Perception vs production comparison
    # ==================================================================
    if "modality" in analyses:
        print("=" * 60)
        print("STEP 4: Perception vs production comparison")
        print("=" * 60)

        from evaluate.modality_analysis import run_modality_analysis

        mod_results = run_modality_analysis(
            study2_val=study2_val,
            feature_types=FEATURE_TYPES,
            subsample=subsample,
        )
        mcnemar_df = mod_results.get("mcnemar_df")
        if mcnemar_df is not None and not mcnemar_df.empty:
            report_lines.append("PERCEPTION vs PRODUCTION")
            report_lines.append("-" * 40)
            for _, row in mcnemar_df.iterrows():
                if row["chi2"] != row["chi2"]:  # NaN check
                    report_lines.append(
                        f"  {row['feature_type']}/{row['task']}: "
                        f"perception only (no spoken data)"
                    )
                else:
                    sig = "*" if row.get("significant_p05") else "n.s."
                    report_lines.append(
                        f"  {row['feature_type']}/{row['task']}: "
                        f"chi2={row['chi2']:.3f} p={row['p_value']:.4f} {sig}"
                    )
            report_lines.append("")
        print()

    # ==================================================================
    # Write summary report
    # ==================================================================
    report_lines.extend([
        "=" * 70,
        "Pipeline: preprocess.py → train_all.py → evaluate_all.py",
        "=" * 70,
    ])

    report_path = RESULTS_ROOT / "summary_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nSummary report saved to: {report_path}")
    print("\n".join(report_lines))

    print(f"\nAll results saved to: {RESULTS_ROOT}")
    print("Done.")


if __name__ == "__main__":
    main()
