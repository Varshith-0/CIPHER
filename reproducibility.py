"""Utilities for deterministic and reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seeds and deterministic backend flags.

    This function is safe to call multiple times.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
