"""
CIPHER — Training loop for ConformerDecoder v2.

Supports:
  - Multi-task training (phoneme_identity + place + manner + voicing)
  - Single-task training (backward compatible)
  - CTC loss for sequence decoding
  - Label smoothing (ε=0.15)
  - Mixup augmentation (α=0.3)
  - Cosine-annealing LR with linear warmup
  - Early stopping, gradient clipping
  - Stochastic weight averaging (SWA)
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import ConformerDecoder, GRUDecoder, NeuroMambaDecoder
from .dataset import TASK_CONFIGS, MULTI_TASK_KEYS, CTC_VOCAB_SIZE


# ===========================================================================
# Label-smoothing CrossEntropy
# ===========================================================================

class LabelSmoothingCE(nn.Module):
    """CrossEntropy with label smoothing ε."""

    def __init__(self, n_classes: int, smoothing: float = 0.15,
                 weight: torch.Tensor = None, ignore_index: int = -1):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), targets: (B,)"""
        valid = targets != self.ignore_index
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits = logits[valid]
        targets = targets[valid]

        log_probs = torch.log_softmax(logits, dim=-1)
        # Smooth targets
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / self.n_classes)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.n_classes)
        # Weighted
        loss = -(smooth * log_probs).sum(dim=-1)
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        return loss.mean()


class MixupLabelSmoothingCE(nn.Module):
    """CrossEntropy for mixup — accepts soft targets (B, C)."""

    def __init__(self, n_classes: int, smoothing: float = 0.15):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), soft_targets: (B, C) — one-hot-like with mixup."""
        log_probs = torch.log_softmax(logits, dim=-1)
        # Apply label smoothing to soft targets
        uniform = torch.full_like(soft_targets, 1.0 / self.n_classes)
        targets = (1.0 - self.smoothing) * soft_targets + self.smoothing * uniform
        loss = -(targets * log_probs).sum(dim=-1)
        return loss.mean()


# ===========================================================================
# Mixup utility
# ===========================================================================

def mixup_data(x, y, alpha=0.3, n_classes=None):
    """
    Apply mixup to a batch. Returns mixed inputs and soft one-hot targets.
    x: (B, T, D), y: (B,) integer labels
    Returns: mixed_x, soft_targets (B, C), lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    # Clamp lambda to avoid trivial mixing
    lam = max(lam, 1.0 - lam)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    # Build soft targets
    if n_classes is not None:
        y_onehot = torch.zeros(batch_size, n_classes, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        y_perm = torch.zeros(batch_size, n_classes, device=x.device)
        y_perm.scatter_(1, y[index].unsqueeze(1), 1.0)
        soft_targets = lam * y_onehot + (1 - lam) * y_perm
    else:
        soft_targets = None

    return mixed_x, soft_targets, lam


# ===========================================================================
# Cosine Annealing with linear warmup
# ===========================================================================

class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine
                    for base_lr in self.base_lrs]


# ===========================================================================
# Multi-task training
# ===========================================================================

def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, unwrapping compile/DataParallel wrappers."""
    m = model
    for _ in range(3):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
            continue
        if hasattr(m, "module"):
            m = m.module
            continue
        break
    return m

def run_experiment(
    train_dataset,
    val_dataset,
    test_dataset,
    save_dir: Path,
    config: dict,
):
    """
    Train a ConformerDecoder (or GRU for legacy) and save results.

    Config keys:
      model_type: "conformer" (default) or "gru"
      v3: bool (True = enable Neuro-Mamba features like EMA/SWA)
      multi_task: bool (True = train all articulatory heads jointly)
      ctc: bool (True = add CTC loss)
      ctc_weight: float (weight of CTC loss, default 0.3)
      label_smoothing: float (default 0.15)
      mixup_alpha: float (default 0.3, 0 to disable)
      device: str (e.g. "cuda:0", "cuda:1") — override auto-detect
      skip_existing: bool — skip if save_dir/best_model.pt exists
      + d_model, n_conformer_blocks, n_heads, conv_channels, conv_kernel,
        hidden_size, n_layers, dropout, lr, weight_decay, batch_size, max_epochs,
        patience, warmup_epochs
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Reproducible runs
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Skip if already trained
    if config.get("skip_existing", False) and (save_dir / "best_model.pt").exists():
        print(f"    ⤷ SKIP: already trained — {save_dir}")
        return

    if len(train_dataset) == 0:
        print(f"    ⤷ SKIP: no training data — {save_dir.name}")
        return
    if train_dataset.n_classes < 2:
        print(f"    ⤷ SKIP: fewer than 2 classes — {save_dir.name}")
        return

    model_type = config.get("model_type", "conformer")
    is_multi_task = config.get("multi_task", False) and model_type in ("conformer", "neuromamba")
    use_ctc = config.get("ctc", False) and model_type in ("conformer", "neuromamba")
    label_smoothing = config.get("label_smoothing", 0.15)
    ctc_weight = config.get("ctc_weight", 0.3)
    mixup_alpha = config.get("mixup_alpha", 0.3)
    use_amp = config.get("amp", True) and torch.cuda.is_available()
    v3 = config.get("v3", False)

    n_classes = train_dataset.n_classes
    input_dim = train_dataset.input_dim
    primary_task = train_dataset.classification_task

    device_str = config.get("device", None)
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build model ----
    if model_type == "neuromamba":
        if is_multi_task:
            task_n_classes = {}
            for tk in MULTI_TASK_KEYS:
                task_n_classes[tk] = len(TASK_CONFIGS[tk]["classes"])
        else:
            task_n_classes = {primary_task: n_classes}

        model = NeuroMambaDecoder(
            input_dim=input_dim,
            task_n_classes=task_n_classes,
            d_model=config.get("d_model", 256),
            n_mamba_blocks=config.get("n_mamba_blocks", 4),
            n_electrodes=1,
            dropout=config.get("dropout", 0.3),
            ctc_vocab_size=CTC_VOCAB_SIZE if use_ctc else None,
        )
    elif model_type == "conformer":
        if is_multi_task:
            task_n_classes = {}
            for tk in MULTI_TASK_KEYS:
                task_n_classes[tk] = len(TASK_CONFIGS[tk]["classes"])
        else:
            task_n_classes = {primary_task: n_classes}

        model = ConformerDecoder(
            input_dim=input_dim,
            task_n_classes=task_n_classes,
            d_model=config.get("d_model", 256),
            n_conformer_blocks=config.get("n_conformer_blocks", 6),
            n_heads=config.get("n_heads", 8),
            conv_channels=config.get("conv_channels", 64),
            conv_kernel=config.get("conv_kernel", 15),
            dropout=config.get("dropout", 0.4),
            drop_path_rate=config.get("drop_path_rate", 0.15),
            ctc_vocab_size=CTC_VOCAB_SIZE if use_ctc else None,
            use_multiscale=config.get("use_multiscale", True),
            use_se=config.get("use_se", True),
            use_attention_pool=config.get("use_attention_pool", True),
        )
    else:
        model = GRUDecoder(
            input_dim=input_dim, n_classes=n_classes,
            hidden_size=config.get("hidden_size", 256),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.3),
        )

    if torch.cuda.device_count() > 1 and not device_str:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optional torch.compile
    if config.get("compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    # AMP scaler
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # ---- Data loaders ----
    configured_workers = config.get("dataloader_workers", None)
    if configured_workers is not None:
        n_workers = int(configured_workers)
    else:
        n_workers = min(4, len(train_dataset) // max(config.get("batch_size", 64), 1))
        n_workers = max(0, n_workers)
    batch_size = config.get("batch_size", 64)

    # ---- Balanced sampler on primary task labels ----
    train_sampler = None
    if len(train_dataset) > 0:
        if is_multi_task and train_dataset.multi_labels:
            primary_labels = train_dataset.multi_labels.get(primary_task)
        else:
            primary_labels = train_dataset.labels

        if primary_labels is not None and len(primary_labels) == len(train_dataset):
            valid_mask = primary_labels >= 0
            if valid_mask.any():
                valid_labels = primary_labels[valid_mask]
                class_counts = np.bincount(valid_labels, minlength=n_classes).astype(np.float64)
                class_counts = np.maximum(class_counts, 1.0)

                sample_weights = np.zeros(len(primary_labels), dtype=np.float64)
                sample_weights[valid_mask] = 1.0 / class_counts[primary_labels[valid_mask]]
                sample_weights = np.maximum(sample_weights, 1e-8)

                train_sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )

    # CTC collation (variable-length targets)
    if use_ctc and train_dataset.ctc:
        collate_fn = _ctc_collate_fn
    elif is_multi_task and train_dataset.multi_task:
        collate_fn = _multi_task_collate_fn
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=n_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        collate_fn=collate_fn,
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        collate_fn=collate_fn,
    ) if test_dataset is not None and len(test_dataset) > 0 else None

    # ---- Loss functions ----
    if is_multi_task:
        criteria = {}
        for tk in task_n_classes:
            tk_cfg = TASK_CONFIGS[tk]
            tk_labels = train_dataset.multi_labels.get(tk)
            if tk_labels is not None and len(tk_labels) > 0:
                valid_mask = tk_labels >= 0
                if valid_mask.sum() > 0:
                    valid_labels = tk_labels[valid_mask]
                    counts = np.bincount(valid_labels, minlength=len(tk_cfg["classes"])).astype(np.float64)
                    counts = np.maximum(counts, 1.0)
                    weights = len(valid_labels) / (len(tk_cfg["classes"]) * counts)
                    weights = np.clip(weights, 0.25, 4.0)
                    weights = weights / np.mean(weights)
                    w = torch.tensor(weights, dtype=torch.float32)
                else:
                    w = None
            else:
                w = None
            criteria[tk] = LabelSmoothingCE(
                n_classes=len(tk_cfg["classes"]),
                smoothing=label_smoothing, weight=w, ignore_index=-1,
            ).to(device)
    else:
        class_weights = train_dataset.class_weights().to(device)
        criteria = {
            primary_task: LabelSmoothingCE(
                n_classes=n_classes, smoothing=label_smoothing,
                weight=class_weights,
            ).to(device)
        }

    # Mixup loss (for when mixup is active)
    mixup_criteria = {}
    if mixup_alpha > 0:
        for tk, nc in task_n_classes.items() if is_multi_task else [(primary_task, n_classes)]:
            mixup_criteria[tk] = MixupLabelSmoothingCE(
                n_classes=nc, smoothing=label_smoothing,
            ).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True) if use_ctc else None

    # ---- Optimizer & scheduler ----
    max_epochs = config.get("max_epochs", 250)
    warmup_epochs = config.get("warmup_epochs", 10)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 5e-4),
        weight_decay=config.get("weight_decay", 5e-3),
        betas=(0.9, 0.98),
    )
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        min_lr=1e-6,
    )

    ema_model = None
    swa_model = None
    swa_start = int(max_epochs * 0.8)
    if v3:
        try:
            ema_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_avg_fn)
        except AttributeError:
            # Fallback for old pytorch
            ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                0.999 * averaged_model_parameter + 0.001 * model_parameter
            ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg_fn)
        
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    # ---- Training loop ----
    monitor_metric = config.get("monitor_metric", "val_acc")
    monitor_mode = config.get("monitor_mode", "max")
    best_monitor = -float("inf") if monitor_mode == "max" else float("inf")
    patience_counter = 0
    log_rows = []
    patience_limit = config.get("patience", 40)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        # Decide if we use mixup this epoch (disable for last 10% of training)
        use_mixup = mixup_alpha > 0 and epoch < int(0.9 * max_epochs)

        for batch in train_loader:
            x_batch, y_batch = batch[0], batch[1]
            x_batch = x_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                raw_model = _unwrap_model(model)
                if isinstance(raw_model, ConformerDecoder):
                    # --- Apply mixup for single-task mode ---
                    if use_mixup and not isinstance(y_batch, dict):
                        y_t = y_batch.to(device, non_blocking=True)
                        mixed_x, soft_targets, lam = mixup_data(
                            x_batch, y_t, alpha=mixup_alpha, n_classes=n_classes,
                        )
                        logits_dict = model(mixed_x)
                        loss = torch.tensor(0.0, device=device)
                        tk = primary_task
                        if tk in logits_dict and tk in mixup_criteria:
                            loss = mixup_criteria[tk](logits_dict[tk], soft_targets)
                        # Accuracy on unmixed predictions (for monitoring)
                        with torch.no_grad():
                            unmixed_logits = model(x_batch)
                            if tk in unmixed_logits:
                                train_correct += (unmixed_logits[tk].argmax(1) == y_t).sum().item()
                                train_total += len(y_t)
                    else:
                        logits_dict = model(x_batch)
                        loss = torch.tensor(0.0, device=device)

                        if isinstance(y_batch, dict):
                            n_task_losses = 0
                            for tk, crit in criteria.items():
                                if tk in logits_dict and tk in y_batch:
                                    tk_labels = y_batch[tk].to(device, non_blocking=True)
                                    loss = loss + crit(logits_dict[tk], tk_labels)
                                    n_task_losses += 1
                            if n_task_losses > 0:
                                loss = loss / n_task_losses
                            if primary_task in logits_dict and primary_task in y_batch:
                                pk_labels = y_batch[primary_task].to(device, non_blocking=True)
                                valid_m = pk_labels >= 0
                                if valid_m.any():
                                    train_correct += (logits_dict[primary_task][valid_m].argmax(1) == pk_labels[valid_m]).sum().item()
                                    train_total += valid_m.sum().item()
                        else:
                            tk = primary_task
                            y_t = y_batch.to(device, non_blocking=True)
                            if tk in logits_dict:
                                loss = criteria[tk](logits_dict[tk], y_t)
                                train_correct += (logits_dict[tk].argmax(1) == y_t).sum().item()
                                train_total += len(y_t)

                    # CTC loss
                    if use_ctc and "ctc" in logits_dict and len(batch) > 2:
                        ctc_logits = logits_dict["ctc"]
                        ctc_targets_flat = batch[2].to(device)
                        ctc_target_lengths = batch[3].to(device)
                        log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
                        input_lengths = torch.full(
                            (log_probs.size(1),), log_probs.size(0),
                            dtype=torch.long, device=device,
                        )
                        ctc_l = ctc_loss_fn(log_probs, ctc_targets_flat,
                                            input_lengths, ctc_target_lengths)
                        loss = loss + ctc_weight * ctc_l

                else:
                    # Legacy GRU
                    y_t = y_batch.to(device, non_blocking=True)
                    logits = model(x_batch)
                    loss = criteria[primary_task](logits, y_t)
                    train_correct += (logits.argmax(1) == y_t).sum().item()
                    train_total += len(y_t)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if ema_model is not None:
                ema_model.update_parameters(model)

            train_loss_sum += loss.item() * x_batch.size(0)

        train_loss = train_loss_sum / max(train_total, len(train_dataset))
        train_acc = train_correct / max(train_total, 1)

        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)

        # --- Validate ---
        eval_model = ema_model if ema_model is not None else model
        val_loss, val_acc = _evaluate_loop(
            eval_model, val_loader, criteria, primary_task, device,
            ctc_loss_fn=ctc_loss_fn, ctc_weight=ctc_weight,
        )

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 4),
            "lr": round(lr_now, 7),
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(f"      Epoch {epoch:3d}/{max_epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.1f}s)")

        # --- Early stopping / checkpoint monitor ---
        if monitor_metric == "val_acc":
            monitor = val_acc if val_loader else train_acc
        elif monitor_metric == "train_acc":
            monitor = train_acc
        elif monitor_metric == "train_loss":
            monitor = train_loss
        else:
            monitor = val_loss if val_loader else train_loss

        improved = monitor > best_monitor if monitor_mode == "max" else monitor < best_monitor
        if improved:
            best_monitor = monitor
            patience_counter = 0
            _save_checkpoint(eval_model, save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"      Early stopping at epoch {epoch}")
                break

    # ---- Save training log ----
    pd.DataFrame(log_rows).to_csv(save_dir / "training_log.csv", index=False)

    if swa_model is not None and epoch >= swa_start:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        # SWA is the final deployed model in v3 if it completed gracefully
        if patience_counter < patience_limit:
            _save_checkpoint(swa_model, save_dir / "best_model.pt")

    # ---- Evaluate on test set ----
    if test_loader and (save_dir / "best_model.pt").exists():
        _load_checkpoint(model, save_dir / "best_model.pt", device)
        test_loss, test_acc = _evaluate_loop(
            model, test_loader, criteria, primary_task, device,
        )
        print(f"      Test (Study 1):  loss={test_loss:.4f}  acc={test_acc:.3f}")
    else:
        test_loss, test_acc = float("nan"), float("nan")

    # ---- Save config ----
    cfg_out = {
        **{k: v for k, v in config.items() if not callable(v)},
        "model_type": model_type,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "primary_task": primary_task,
        "multi_task": is_multi_task,
        "ctc": use_ctc,
        "label_names": train_dataset.label_names,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset) if test_dataset else 0,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "best_monitor": round(float(best_monitor), 5),
        "test_loss": round(test_loss, 5) if not np.isnan(test_loss) else None,
        "test_acc": round(test_acc, 4) if not np.isnan(test_acc) else None,
    }
    if is_multi_task:
        cfg_out["task_n_classes"] = {
            tk: len(TASK_CONFIGS[tk]["classes"]) for tk in MULTI_TASK_KEYS
        }
        cfg_out["task_label_names"] = {
            tk: TASK_CONFIGS[tk]["classes"] for tk in MULTI_TASK_KEYS
        }
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)


# ===========================================================================
# Helpers
# ===========================================================================

def _evaluate_loop(model, loader, criteria, primary_task, device,
                   ctc_loss_fn=None, ctc_weight=0.3):
    """Evaluate model on a DataLoader. Returns (loss, accuracy)."""
    if loader is None:
        return float("nan"), float("nan")

    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    raw_model = _unwrap_model(model)

    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch = batch[0], batch[1]
            x_batch = x_batch.to(device, non_blocking=True)

            if isinstance(raw_model, ConformerDecoder):
                logits_dict = model(x_batch)
                loss = torch.tensor(0.0, device=device)

                if isinstance(y_batch, dict):
                    n_task_losses = 0
                    for tk, crit in criteria.items():
                        if tk in logits_dict and tk in y_batch:
                            tk_labels = y_batch[tk].to(device, non_blocking=True)
                            loss = loss + crit(logits_dict[tk], tk_labels)
                            n_task_losses += 1
                    if n_task_losses > 0:
                        loss = loss / n_task_losses
                    if primary_task in logits_dict and primary_task in y_batch:
                        pk_labels = y_batch[primary_task].to(device, non_blocking=True)
                        valid_m = pk_labels >= 0
                        if valid_m.any():
                            correct += (logits_dict[primary_task][valid_m].argmax(1) == pk_labels[valid_m]).sum().item()
                            total += valid_m.sum().item()
                else:
                    y_t = y_batch.to(device, non_blocking=True)
                    if primary_task in logits_dict:
                        loss = criteria[primary_task](logits_dict[primary_task], y_t)
                        correct += (logits_dict[primary_task].argmax(1) == y_t).sum().item()
                        total += len(y_t)
            else:
                y_t = y_batch.to(device, non_blocking=True)
                logits = model(x_batch)
                loss = criteria[primary_task](logits, y_t)
                correct += (logits.argmax(1) == y_t).sum().item()
                total += len(y_t)

            loss_sum += loss.item() * x_batch.size(0)

    return loss_sum / max(total, len(loader.dataset)), correct / max(total, 1)


def _multi_task_collate_fn(batch):
    """Collate for multi-task mode: (features, dict_of_labels)."""
    features = torch.stack([b[0] for b in batch])
    label_keys = batch[0][1].keys()
    labels = {k: torch.stack([b[1][k] for b in batch]) for k in label_keys}
    return features, labels


def _ctc_collate_fn(batch):
    """Collate for CTC mode: (features, primary_target, ctc_targets_flat, ctc_lengths)."""
    features = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]

    primary_targets = torch.tensor(
        [int(t[0].item()) - 1 if len(t) > 0 else 0 for t in targets],
        dtype=torch.long,
    )

    ctc_targets_flat = torch.cat(targets)
    ctc_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return features, primary_targets, ctc_targets_flat, ctc_lengths


def _save_checkpoint(model, path):
    m = _unwrap_model(model)
    torch.save(m.state_dict(), path)


def _load_checkpoint(model, path, device):
    m = _unwrap_model(model)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
