"""
CIPHER — PyTorch Dataset for loading preprocessed EEG data (ERP and DDA).
Reads from ~/cipher/preprocessed/ produced by preprocess.py.

Supports:
  - Single-task mode (backward compatible): returns (features, label)
  - Multi-task mode: returns (features, dict_of_labels)
  - CTC mode: returns (features, phoneme_sequence_tensor)
"""

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

mne.set_log_level("ERROR")

# ===========================================================================
# Constants
# ===========================================================================
PREPROCESSED_ROOT = Path(
    os.getenv(
        "CIPHER_PREPROCESSED_ROOT",
        str(Path(__file__).resolve().parent.parent / "preprocessed"),
    )
)

VOWELS = {"a", "e", "i", "o", "u"}
CONSONANTS = {"b", "p", "d", "t", "s", "z"}

COMPLEXITY_MAP = {
    "single_phoneme_perceived": "single_phoneme",
    "single_phoneme_spoken": "single_phoneme",
    "cv_pairs": "diphone",
    "vc_pairs": "diphone",
    "cvc_real_words": "triphone",
    "cvc_pseudowords": "triphone",
}

# Label configuration per classification task
TASK_CONFIGS = {
    "phoneme_identity": {
        "label_col": "phoneme1",
        "classes": sorted(list(VOWELS | CONSONANTS)),
    },
    "place": {
        "label_col": "place",
        "classes": ["alveolar", "bilabial"],
    },
    "manner": {
        "label_col": "manner",
        "classes": ["fricative", "stop"],
    },
    "voicing": {
        "label_col": "voicing",
        "classes": ["unvoiced", "voiced"],
    },
    "category": {
        "label_fn": "category",
        "classes": ["consonant", "vowel"],
    },
    "complexity": {
        "label_fn": "complexity",
        "classes": ["diphone", "single_phoneme", "triphone"],
    },
}

# Tasks usable for multi-task heads (articulatory + phoneme identity)
MULTI_TASK_KEYS = ["phoneme_identity", "place", "manner", "voicing"]

# CTC blank token (index 0 by convention)
CTC_BLANK = 0
CTC_PHONEME_CLASSES = sorted(list(VOWELS | CONSONANTS))  # same as phoneme_identity
CTC_VOCAB_SIZE = len(CTC_PHONEME_CLASSES) + 1  # +1 for blank
FEATURE_CLIP_ABS = 1e6


# ===========================================================================
# Dataset
# ===========================================================================

class EEGDataset(Dataset):
    """
    Loads preprocessed EEG features (ERP or DDA) across subjects.

    Modes:
      - Single-task (default): ``classification_task`` selects one label.
        ``__getitem__`` returns ``(features, int_label)``.
      - Multi-task: ``multi_task=True`` computes labels for *all* MULTI_TASK_KEYS.
        ``__getitem__`` returns ``(features, dict[str, int_label])``.
      - CTC: ``ctc=True`` returns ``(features, phoneme_seq_tensor)`` where
        phoneme_seq_tensor encodes the 1–3 phoneme sequence (blank=0).

    Parameters
    ----------
    subjects : list[str]
    feature_type : "erp" | "dda"
    classification_task : str   — primary task (used for single-task filtering)
    tms_condition, task_type_filter, augment, subsample : as before
    multi_task : bool  — return dict of labels for all articulatory tasks
    ctc : bool         — return phoneme sequence tensor instead of class index
    """

    def __init__(
        self,
        subjects: list,
        feature_type: str,
        classification_task: str = "phoneme_identity",
        tms_condition: str = None,
        task_type_filter: list = None,
        augment: bool = False,
        subsample: float = 1.0,
        multi_task: bool = False,
        ctc: bool = False,
        normalize: bool = True,
        temporal_stride: int = 1,
    ):
        assert feature_type in ("erp", "dda")
        assert classification_task in TASK_CONFIGS

        self.feature_type = feature_type
        self.classification_task = classification_task
        self.augment = augment
        self.multi_task = multi_task
        self.ctc = ctc
        self.normalize = normalize
        self.temporal_stride = max(1, int(temporal_stride))
        self.cfg = TASK_CONFIGS[classification_task]

        # ---- Load data across subjects ----
        all_features = []
        all_meta = []

        for sub_id in subjects:
            ses = "ses-01" if sub_id.startswith("sub-P") else "ses-02"
            base_dir = PREPROCESSED_ROOT / feature_type / sub_id / ses

            if not base_dir.exists():
                continue

            if feature_type == "erp":
                data_files = sorted(base_dir.glob("*_epo.fif"))
            else:
                data_files = sorted(base_dir.glob("*_dda.npy"))

            for data_file in data_files:
                stem = data_file.stem
                task_type = stem.replace("_epo", "").replace("_dda", "")

                if task_type_filter and task_type not in task_type_filter:
                    continue

                meta_file = data_file.parent / f"{task_type}_metadata.csv"
                if not meta_file.exists():
                    continue

                meta = pd.read_csv(meta_file, keep_default_na=False)
                if meta.empty:
                    continue

                # ---- Filter by TMS condition ----
                if tms_condition is not None:
                    mask = meta["tms_condition"] == tms_condition
                    if not mask.any():
                        continue
                    indices = np.where(mask.values)[0]
                    meta = meta.iloc[indices].reset_index(drop=True)
                else:
                    indices = None

                # ---- Load features ----
                if feature_type == "erp":
                    epochs = mne.read_epochs(str(data_file), verbose=False)
                    data = epochs.get_data()
                    if indices is not None:
                        data = data[indices]
                    data = data.transpose(0, 2, 1)  # [n, time, ch]
                else:
                    data = np.load(str(data_file))
                    if indices is not None:
                        data = data[indices]
                    n_t, n_ch, n_win, n_coeff = data.shape
                    data = data.transpose(0, 2, 1, 3).reshape(n_t, n_win, n_ch * n_coeff)

                if self.temporal_stride > 1:
                    data = data[:, ::self.temporal_stride, :]

                if len(data) != len(meta):
                    n = min(len(data), len(meta))
                    data = data[:n]
                    meta = meta.iloc[:n].reset_index(drop=True)

                data = self._sanitize_feature_block(data)
                all_features.append(data)
                all_meta.append(meta)

        # ---- Handle empty dataset ----
        if not all_features:
            self.features = np.zeros((0, 1, 1), dtype=np.float32)
            self.labels = np.array([], dtype=np.int64)
            self.multi_labels = {}
            self.ctc_targets = []
            self.label_names = self.cfg["classes"]
            self._n_channels = 0
            return

        # ---- Align sequence lengths and channel counts ----
        min_seq = min(f.shape[1] for f in all_features)
        min_feat = min(f.shape[2] for f in all_features)
        all_features = [f[:, :min_seq, :min_feat] for f in all_features]

        features = np.concatenate(all_features, axis=0)
        meta_df = pd.concat(all_meta, ignore_index=True)

        # ---- Compute primary labels and filter valid rows ----
        raw_labels = self._compute_labels(meta_df, self.cfg)
        valid = raw_labels.isin(self.cfg["classes"])

        features = features[valid.values]
        meta_df = meta_df[valid].reset_index(drop=True)

        class_to_idx = {c: i for i, c in enumerate(self.cfg["classes"])}
        labels = raw_labels[valid].map(class_to_idx).values.astype(np.int64)

        # ---- Multi-task labels ----
        multi_labels = {}
        if self.multi_task:
            for tk in MULTI_TASK_KEYS:
                tk_cfg = TASK_CONFIGS[tk]
                tk_raw = self._compute_labels(meta_df, tk_cfg)
                tk_c2i = {c: i for i, c in enumerate(tk_cfg["classes"])}
                tk_mapped = tk_raw.map(tk_c2i)
                # Fill invalid with -1 (ignored by CrossEntropyLoss)
                tk_mapped = tk_mapped.fillna(-1).astype(np.int64).values
                multi_labels[tk] = tk_mapped

        # ---- CTC phoneme sequences ----
        ctc_targets = []
        if self.ctc:
            ph_c2i = {c: i + 1 for i, c in enumerate(CTC_PHONEME_CLASSES)}  # +1 for blank=0
            for _, row in meta_df.iterrows():
                seq = []
                for pkey in ["phoneme1", "phoneme2", "phoneme3"]:
                    if pkey in row and pd.notna(row[pkey]) and str(row[pkey]) in ph_c2i:
                        seq.append(ph_c2i[str(row[pkey])])
                ctc_targets.append(seq if seq else [ph_c2i.get("a", 1)])  # fallback

        # ---- Subsample ----
        if subsample < 1.0 and len(labels) > 0:
            n = max(1, int(len(labels) * subsample))
            rng = np.random.RandomState(42)
            idx = rng.choice(len(labels), n, replace=False)
            features = features[idx]
            labels = labels[idx]
            for tk in multi_labels:
                multi_labels[tk] = multi_labels[tk][idx]
            if ctc_targets:
                ctc_targets = [ctc_targets[i] for i in idx]

        self.features = features
        self.labels = labels
        self.multi_labels = multi_labels
        self.ctc_targets = ctc_targets
        self.label_names = self.cfg["classes"]
        if feature_type == "erp":
            self._n_channels = features.shape[2] if len(features) > 0 else 0
        else:
            self._n_channels = features.shape[2] // 3 if len(features) > 0 else 0

    # ------------------------------------------------------------------
    # Label computation
    # ------------------------------------------------------------------

    def _compute_labels(self, meta_df: pd.DataFrame, cfg: dict) -> pd.Series:
        """Return a Series of string labels aligned with meta_df rows."""
        if "label_col" in cfg:
            return meta_df[cfg["label_col"]].astype(str)

        fn_name = cfg["label_fn"]
        if fn_name == "category":
            return meta_df["phoneme1"].apply(
                lambda p: "consonant" if p in CONSONANTS
                else ("vowel" if p in VOWELS else "n/a")
            )
        if fn_name == "complexity":
            return meta_df["task_type"].map(COMPLEXITY_MAP).fillna("n/a")

        return pd.Series(["n/a"] * len(meta_df))

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment_sample(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single sample [time, features]."""
        x = x.copy()

        # 1. Gaussian noise (σ = 0.01 * signal std)
        std = x.std()
        if std > 1e-10:
            x += np.random.randn(*x.shape).astype(np.float32) * (0.01 * std)

        # 2. Channel dropout (5% of channels set to zero)
        if self.feature_type == "erp":
            n_ch = x.shape[1]
            n_drop = max(1, int(0.05 * n_ch))
            drop_idx = np.random.choice(n_ch, n_drop, replace=False)
            x[:, drop_idx] = 0.0
        else:
            # DDA: features are [ch0_a1, ch0_a2, ch0_a3, ch1_a1, ...]
            n_ch = self._n_channels
            if n_ch > 0:
                n_drop = max(1, int(0.05 * n_ch))
                drop_ch = np.random.choice(n_ch, n_drop, replace=False)
                for ch in drop_ch:
                    x[:, ch * 3 : (ch + 1) * 3] = 0.0

        # 3. Random time shift (±20 ms)
        if self.feature_type == "erp":
            max_shift = 5  # samples at 256 Hz ≈ 19.5 ms
        else:
            max_shift = 20  # DDA windows at 1 ms/window = 20 ms
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift != 0:
            x = np.roll(x, shift, axis=0)
            if shift > 0:
                x[:shift] = 0.0
            else:
                x[shift:] = 0.0

        return x

    def _sanitize_feature_block(self, x: np.ndarray) -> np.ndarray:
        """Replace non-finite values and clip extremes before float32 cast."""
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=FEATURE_CLIP_ABS, neginf=-FEATURE_CLIP_ABS)
        x = np.clip(x, -FEATURE_CLIP_ABS, FEATURE_CLIP_ABS)
        return x.astype(np.float32, copy=False)

    def _normalize_sample(self, x: np.ndarray) -> np.ndarray:
        """Per-sample z-score over time for each feature channel."""
        x = self._sanitize_feature_block(x)
        mu = x.mean(axis=0, keepdims=True)
        sigma = x.std(axis=0, keepdims=True)
        sigma = np.maximum(sigma, 1e-6)
        x = (x - mu) / sigma
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.augment:
            x = self._augment_sample(x)
        if self.normalize:
            x = self._normalize_sample(x)
        x_t = torch.from_numpy(x)

        # CTC mode: return phoneme sequence target
        if self.ctc and self.ctc_targets:
            target = torch.tensor(self.ctc_targets[idx], dtype=torch.long)
            return x_t, target

        # Multi-task mode: return dict of labels
        if self.multi_task and self.multi_labels:
            label_dict = {
                tk: torch.tensor(self.multi_labels[tk][idx], dtype=torch.long)
                for tk in self.multi_labels
            }
            return x_t, label_dict

        # Single-task mode (default)
        y = self.labels[idx]
        return x_t, torch.tensor(y, dtype=torch.long)

    @property
    def n_classes(self) -> int:
        return len(self.cfg["classes"])

    @property
    def input_dim(self) -> int:
        """Feature dimension per time-step."""
        if len(self.features) == 0:
            return 0
        return self.features.shape[2]

    @property
    def seq_len(self) -> int:
        if len(self.features) == 0:
            return 0
        return self.features.shape[1]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for CrossEntropyLoss."""
        if len(self.labels) == 0:
            return torch.ones(self.n_classes)
        counts = np.bincount(self.labels, minlength=self.n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)  # avoid division by zero
        weights = len(self.labels) / (self.n_classes * counts)
        # Prevent extreme re-weighting from destabilizing optimization.
        weights = np.clip(weights, 0.25, 4.0)
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float32)
