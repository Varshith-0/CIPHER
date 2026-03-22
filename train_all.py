#!/usr/bin/env python3
"""
CIPHER — Master training script.

Trains ConformerDecoder models with multi-task heads (place + manner +
voicing + phoneme_identity), optional CTC loss, and label smoothing.

Usage:
  python train_all.py --dry-run                          # quick test
  python train_all.py --task phoneme_identity             # single task
  python train_all.py                                     # full training
  python train_all.py --feature-type erp --tms null       # specific combo
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.dataset import EEGDataset
from models.train import run_experiment

# ===========================================================================
# Subject splits
# ===========================================================================
STUDY2_ALL = [f"sub-S{i:02d}" for i in range(1, 17)]
STUDY2_VAL = ["sub-S04", "sub-S09", "sub-S14"]
STUDY2_TRAIN = [s for s in STUDY2_ALL if s not in STUDY2_VAL]
STUDY1_TEST = [f"sub-P{i:02d}" for i in range(1, 9)]

# ===========================================================================
# Experiment grid
# ===========================================================================
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
MODALITIES = {
    "perceived": ["single_phoneme_perceived"],
    "spoken": ["single_phoneme_spoken"],
    "merged": ["single_phoneme_perceived", "single_phoneme_spoken"],
}
MODALITY_TASKS = [
    "phoneme_identity", "place", "manner", "voicing", "category",
]

# Multi-task heads are available for phoneme_identity/place/manner/voicing.
# They are opt-in via --enable-multitask for stability.
MULTI_TASK_ELIGIBLE = {"phoneme_identity", "place", "manner", "voicing"}

# Tasks that benefit from CTC (sequence decoding on triphones / diphones)
CTC_ELIGIBLE = {"phoneme_identity"}

# ===========================================================================
# Default hyperparameters
# ===========================================================================
DEFAULT_CONFIG = {
    "model_type": "conformer",
    "d_model": 192,
    "n_conformer_blocks": 4,
    "n_heads": 4,
    "conv_channels": 64,
    "conv_kernel": 15,
    "dropout": 0.3,
    "label_smoothing": 0.0,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 120,
    "patience": 20,
    "scheduler_patience": 6,
    "scheduler_factor": 0.5,
    "multi_task": False,   # set per-experiment below
    "ctc": False,          # set per-experiment below
    "ctc_weight": 0.3,
    "enable_ctc": False,   # keep CTC opt-in; improves stability for cls heads
    "augment_train": False,
    "amp": True,            # mixed-precision training
    "compile": False,       # torch.compile can be unstable in threaded sweeps
    "dataloader_workers": 0,  # avoid worker deadlocks with threaded multi-GPU jobs
    "use_multiscale": True,
    "use_se": True,
    "use_attention_pool": True,
    "drop_path_rate": 0.15,
}

MODELS_ROOT = Path(__file__).resolve().parent / "models_out"


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CIPHER Conformer decoder training")
    parser.add_argument("--task", choices=TASKS + ["all"], default="all")
    parser.add_argument("--feature-type", choices=FEATURE_TYPES + ["all"], default="all")
    parser.add_argument("--tms", choices=list(TMS_CONDITIONS.keys()) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true",
                        help="10%% data, 5 epochs for quick testing")
    parser.add_argument("--skip-modality", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments where best_model.pt already exists")
    parser.add_argument("--enable-ctc", action="store_true",
                        help="Enable CTC branch for phoneme_identity experiments")
    parser.add_argument("--enable-multitask", action="store_true",
                        help="Enable joint multi-task heads for eligible tasks")
    parser.add_argument("--augment-train", action="store_true",
                        help="Enable training-time data augmentation")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early-stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="Override label smoothing")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override AdamW weight decay")
    parser.add_argument("--ctc-weight", type=float, default=None,
                        help="Override CTC loss weight")
    parser.add_argument("--monitor-metric", choices=["val_loss", "val_acc", "train_loss", "train_acc"],
                        default=None, help="Checkpoint/early-stop monitor metric")
    parser.add_argument("--monitor-mode", choices=["min", "max"], default=None,
                        help="Monitor direction: min or max")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible runs")
    parser.add_argument(
        "--task-type-filter", action="append", default=None,
        help="Restrict loaded trials to task_type values (repeatable)",
    )
    parser.add_argument("--split-8-8", action="store_true",
                        help="Use Study2 split S01-S08 train, S09-S16 val")
    parser.add_argument("--train-subjects", type=str, default=None,
                        help="Comma-separated Study2 train subjects override")
    parser.add_argument("--val-subjects", type=str, default=None,
                        help="Comma-separated Study2 val subjects override")
    parser.add_argument("--save-suffix", type=str, default="",
                        help="Append suffix to save path (useful for ablations)")
    parser.add_argument("--no-se", action="store_true",
                        help="Disable squeeze-excitation in front-end")
    parser.add_argument("--no-multiscale", action="store_true",
                        help="Disable multi-scale conv branches (single branch)")
    parser.add_argument("--no-attention-pool", action="store_true",
                        help="Use mean pooling instead of learned attention pooling")
    parser.add_argument("--drop-path-rate", type=float, default=None,
                        help="Override stochastic depth drop-path rate")
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    feat_types = FEATURE_TYPES if args.feature_type == "all" else [args.feature_type]
    tms_keys = list(TMS_CONDITIONS.keys()) if args.tms == "all" else [args.tms]

    config = dict(DEFAULT_CONFIG)
    config["skip_existing"] = args.skip_existing
    config["enable_ctc"] = args.enable_ctc
    config["enable_multitask"] = args.enable_multitask
    config["augment_train"] = args.augment_train
    if args.no_se:
        config["use_se"] = False
    if args.no_multiscale:
        config["use_multiscale"] = False
    if args.no_attention_pool:
        config["use_attention_pool"] = False

    # Optional hyperparameter overrides
    if args.lr is not None:
        config["lr"] = args.lr
    if args.dropout is not None:
        config["dropout"] = args.dropout
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["patience"] = args.patience
    if args.label_smoothing is not None:
        config["label_smoothing"] = args.label_smoothing
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.ctc_weight is not None:
        config["ctc_weight"] = args.ctc_weight
    if args.drop_path_rate is not None:
        config["drop_path_rate"] = args.drop_path_rate
    if args.monitor_metric is not None:
        config["monitor_metric"] = args.monitor_metric
    if args.monitor_mode is not None:
        config["monitor_mode"] = args.monitor_mode
    if args.seed is not None:
        config["seed"] = args.seed
    subsample = 1.0
    if args.dry_run:
        config["max_epochs"] = 5
        subsample = 0.1
        print("=== DRY RUN: 10% data, 5 epochs ===\n")

    # Subject split selection
    if args.split_8_8:
        study2_train = [f"sub-S{i:02d}" for i in range(1, 9)]
        study2_val = [f"sub-S{i:02d}" for i in range(9, 17)]
    else:
        study2_train = list(STUDY2_TRAIN)
        study2_val = list(STUDY2_VAL)

    if args.train_subjects:
        study2_train = [s.strip() for s in args.train_subjects.split(",") if s.strip()]
    if args.val_subjects:
        study2_val = [s.strip() for s in args.val_subjects.split(",") if s.strip()]

    # ==================================================================
    # Build experiment job list
    # ==================================================================
    jobs = []  # list of (label, save_dir, exp_config, train_ds, val_ds, test_ds)

    for ft in feat_types:
        for task in tasks:
            for tms_key in tms_keys:
                tms_val = TMS_CONDITIONS[tms_key]
                save_dir = MODELS_ROOT / ft / task / tms_key
                if args.save_suffix:
                    save_dir = save_dir / args.save_suffix
                use_mt = (task in MULTI_TASK_ELIGIBLE) and config.get("enable_multitask", False)
                use_ctc = (task in CTC_ELIGIBLE) and config.get("enable_ctc", False)

                exp_config = dict(config)
                exp_config["multi_task"] = use_mt
                exp_config["ctc"] = use_ctc

                label = f"{ft}/{task}/{tms_key}"
                filter_kwargs = {"tms_condition": tms_val}
                if args.task_type_filter:
                    filter_kwargs["task_type_filter"] = list(args.task_type_filter)
                jobs.append((label, save_dir, exp_config, ft, task,
                             filter_kwargs, subsample, use_mt, use_ctc))

    if not args.skip_modality:
        mod_tasks = [t for t in tasks if t in MODALITY_TASKS]
        for ft in feat_types:
            for task in mod_tasks:
                for mod_key, tt_filter in MODALITIES.items():
                    save_dir = MODELS_ROOT / ft / task / f"modality_{mod_key}"
                    if args.save_suffix:
                        save_dir = save_dir / args.save_suffix
                    use_mt = (task in MULTI_TASK_ELIGIBLE) and config.get("enable_multitask", False)
                    exp_config = dict(config)
                    exp_config["multi_task"] = use_mt
                    exp_config["ctc"] = False

                    label = f"{ft}/{task}/modality_{mod_key}"
                    jobs.append((label, save_dir, exp_config, ft, task,
                                 {"task_type_filter": tt_filter}, subsample,
                                 use_mt, False))

    print(f"Planning {len(jobs)} experiments.\n")

    # ==================================================================
    # GPU scheduling: run 1 experiment per GPU in parallel
    # ==================================================================
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_locks = [threading.Lock() for _ in range(max(n_gpus, 1))]

    def _run_job(job_idx, job):
        label, save_dir, exp_config, ft, task, filter_kwargs, ss, use_mt, use_ctc = job
        gpu_id = job_idx % max(n_gpus, 1)

        with gpu_locks[gpu_id]:
            dev = f"cuda:{gpu_id}" if n_gpus > 0 else "cpu"
            exp_config = dict(exp_config)
            exp_config["device"] = dev

            print(f"[{job_idx+1}/{len(jobs)}] {label}  (GPU {gpu_id})"
                  f"  [MT={use_mt}, CTC={use_ctc}]")

            # Quick skip before loading data
            if exp_config.get("skip_existing") and (save_dir / "best_model.pt").exists():
                print(f"    ⤷ SKIP: already trained — {save_dir}")
                return

            tms_cond = filter_kwargs.get("tms_condition")
            tt_filter = filter_kwargs.get("task_type_filter")

            ds_kwargs = dict(
                feature_type=ft, classification_task=task,
                augment=exp_config.get("augment_train", False), subsample=ss,
                multi_task=use_mt, ctc=use_ctc,
                normalize=True,
            )
            if ft == "dda":
                ds_kwargs["temporal_stride"] = 4
            if tms_cond is not None:
                ds_kwargs["tms_condition"] = tms_cond
            if tt_filter is not None:
                ds_kwargs["task_type_filter"] = tt_filter
                ds_kwargs["ctc"] = False

            train_ds = EEGDataset(subjects=study2_train, **ds_kwargs)
            ds_kwargs["augment"] = False
            val_ds = EEGDataset(subjects=study2_val, **ds_kwargs)
            if tms_cond is not None:
                test_ds = EEGDataset(subjects=STUDY1_TEST, **ds_kwargs)
            else:
                test_ds = None

            print(f"     train={len(train_ds)}  val={len(val_ds)}  "
                  f"test={len(test_ds) if test_ds else 0}  "
                  f"classes={train_ds.n_classes}")

            run_experiment(
                train_dataset=train_ds, val_dataset=val_ds,
                test_dataset=test_ds, save_dir=save_dir,
                config=exp_config,
            )
            print()

    if n_gpus >= 2:
        # Run 2 experiments in parallel — one per GPU
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = {
                executor.submit(_run_job, i, job): i
                for i, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ERROR in job {idx}: {type(e).__name__}: {e}")
                    print(traceback.format_exc())
    else:
        for i, job in enumerate(jobs):
            _run_job(i, job)

    print(f"\nDone.  {len(jobs)} experiments completed.")
    print(f"Models saved to: {MODELS_ROOT}")


if __name__ == "__main__":
    main()
