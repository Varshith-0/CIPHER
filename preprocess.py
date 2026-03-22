#!/usr/bin/env python3
"""
CIPHER Preprocessing Pipeline
=============================
Two parallel preprocessing paths from raw BIDS EEG (.edf):
  Path A: ERP-based (downsample → filter → epoch → artifact rejection)
  Path B: DDA features (raw 2000 Hz → sliding-window DDA → epoch)

Usage:
    conda run -n cipher python preprocess.py --dry-run   # sub-P01 + sub-S01 only
    conda run -n cipher python preprocess.py              # full dataset
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
import multiprocessing as mp
from multiprocessing import cpu_count

import mne
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ===========================================================================
# Constants
# ===========================================================================
BIDS_ROOT = Path(__file__).parent / "ds006104"
DERIVATIVES_ROOT = BIDS_ROOT / "derivatives" / "eeglab"
OUT_ROOT = Path(__file__).parent / "preprocessed"

STUDY1_SUBS = [f"sub-P{i:02d}" for i in range(1, 9)]     # P01..P08, ses-01
STUDY2_SUBS = [f"sub-S{i:02d}" for i in range(1, 17)]     # S01..S16, ses-02

VOWELS = {"a", "e", "i", "o", "u"}
CONSONANTS = {"b", "p", "d", "t", "s", "z"}

# Phoneme articulatory feature lookup
PLACE_MAP = {
    "b": "bilabial", "p": "bilabial",
    "d": "alveolar", "t": "alveolar", "s": "alveolar", "z": "alveolar",
}
MANNER_MAP = {
    "b": "stop", "p": "stop", "d": "stop", "t": "stop",
    "s": "fricative", "z": "fricative",
}
VOICING_MAP = {
    "b": "voiced", "p": "unvoiced",
    "d": "voiced", "t": "unvoiced",
    "s": "unvoiced", "z": "voiced",
}

# DDA parameters (Lainscsek & Sejnowski, 2015)
DDA_FS = 2000          # Hz — use raw sampling rate
DDA_DT = 1.0 / DDA_FS  # 0.0005 s
DDA_TAU1 = 6           # samples (3 ms)
DDA_TAU2 = 16          # samples (8 ms)
DDA_WIN_LEN = 60       # samples (30 ms at 2000 Hz)
DDA_WIN_SHIFT = 2      # samples (1 ms at 2000 Hz)

# ERP epoch parameters
ERP_TMIN = -0.2        # seconds (wider pre-stimulus for baseline)
ERP_TMAX = 0.8         # seconds
ERP_RESAMPLE = 256     # Hz
ERP_REJECT_UV = 150e-6 # 150 µV in volts (MNE uses V internally)
ERP_ICA_N_COMPONENTS = 15  # ICA components for artifact rejection

# ===========================================================================
# Helpers
# ===========================================================================

def get_session(sub_id: str) -> str:
    """Return session label for a subject."""
    return "ses-01" if sub_id.startswith("sub-P") else "ses-02"


def get_study(sub_id: str) -> str:
    return "2019" if sub_id.startswith("sub-P") else "2021"


def get_eeg_dir(sub_id: str) -> Path:
    ses = get_session(sub_id)
    return BIDS_ROOT / sub_id / ses / "eeg"


def get_eeglab_set_path(sub_id: str) -> Path | None:
    """Return path to EEGLAB-cleaned .set file if it exists."""
    ses = get_session(sub_id)
    sid = sub_id.replace("sub-", "")
    set_path = DERIVATIVES_ROOT / sub_id / ses / f"{sid}.set"
    return set_path if set_path.exists() else None


def get_task_files(sub_id: str) -> dict:
    """Return {task_key: (edf_path, events_path)} for a subject."""
    eeg_dir = get_eeg_dir(sub_id)
    ses = get_session(sub_id)
    prefix = f"{sub_id}_{ses}"

    tasks = {}
    # task-phonemes (always present)
    edf = eeg_dir / f"{prefix}_task-phonemes_eeg.edf"
    evt = eeg_dir / f"{prefix}_task-phonemes_events.tsv"
    if edf.exists() and evt.exists():
        tasks["phonemes"] = (edf, evt)

    # task-singlephoneme (Study 2 only)
    edf = eeg_dir / f"{prefix}_task-singlephoneme_eeg.edf"
    evt = eeg_dir / f"{prefix}_task-singlephoneme_events.tsv"
    if edf.exists() and evt.exists():
        tasks["singlephoneme"] = (edf, evt)

    # task-Words (Study 2 only)
    edf = eeg_dir / f"{prefix}_task-Words_eeg.edf"
    evt = eeg_dir / f"{prefix}_task-Words_events.tsv"
    if edf.exists() and evt.exists():
        tasks["Words"] = (edf, evt)

    return tasks


def parse_events(events_tsv: Path, bids_task: str, study: str):
    """
    Parse a BIDS events TSV into trial-level metadata.

    Returns list of dicts, one per stimulus trial, with keys:
        onset_sec, trial_idx, phoneme1, phoneme2, phoneme3,
        place, manner, voicing, category, tms_condition,
        task_type, word_type, study
    """
    df = pd.read_csv(events_tsv, sep="\t")

    # Build TMS lookup: trial_number → TMS row
    tms_rows = df[df["trial_type"] == "TMS"].copy()
    tms_lookup = {}
    for _, row in tms_rows.iterrows():
        tn = row.get("trial")
        if pd.notna(tn):
            tms_lookup[int(tn)] = row

    stim_rows = df[df["trial_type"] == "stimulus"].copy()
    trials = []
    # Each stimulus row is preceded by a TMS row — pair them by position
    tms_list = tms_rows.reset_index(drop=True)
    stim_list = stim_rows.reset_index(drop=True)

    for i in range(len(stim_list)):
        s = stim_list.iloc[i]
        t = tms_list.iloc[i] if i < len(tms_list) else pd.Series()

        onset = float(s["onset"])
        p1 = str(s["phoneme1"]) if pd.notna(s["phoneme1"]) else "n/a"
        p2 = str(s.get("phoneme2", "n/a")) if pd.notna(s.get("phoneme2")) else "n/a"
        p3 = str(s.get("phoneme3", "n/a")) if pd.notna(s.get("phoneme3")) else "n/a"

        # Articulatory features from TMS row (which carries them)
        cat = str(t.get("category", "n/a")) if pd.notna(t.get("category")) else "n/a"
        manner = str(t.get("manner", "n/a")) if pd.notna(t.get("manner")) else "n/a"
        place = str(t.get("place", "n/a")) if pd.notna(t.get("place")) else "n/a"
        voicing = str(t.get("voicing", "n/a")) if pd.notna(t.get("voicing")) else "n/a"
        tms_target = str(s.get("tms_target", "n/a")) if pd.notna(s.get("tms_target")) else "n/a"
        trial_n = int(t["trial"]) if pd.notna(t.get("trial")) else i + 1

        # Derive articulatory features from phoneme identity when TMS row lacks them
        if place == "n/a" and p1 in CONSONANTS:
            place = PLACE_MAP.get(p1, "n/a")
        if manner == "n/a" and p1 in CONSONANTS:
            manner = MANNER_MAP.get(p1, "n/a")
        if voicing == "n/a" and p1 in CONSONANTS:
            voicing = VOICING_MAP.get(p1, "n/a")

        # Determine task_type
        task_type = _classify_task(bids_task, p1, p2, p3, cat, study)

        # Word type
        word_type = "n/a"
        if bids_task == "Words":
            word_type = "real" if cat == "real" else "pseudo" if cat in ("nonce",) else "n/a"

        # Map tms_condition to canonical labels
        tms_condition = _map_tms(tms_target)

        trials.append({
            "onset_sec": onset,
            "trial_idx": trial_n,
            "phoneme1": p1,
            "phoneme2": p2,
            "phoneme3": p3,
            "place": place,
            "manner": manner,
            "voicing": voicing,
            "category": cat,
            "tms_condition": tms_condition,
            "task_type": task_type,
            "word_type": word_type,
            "study": study,
        })

    return trials


def _classify_task(bids_task: str, p1: str, p2: str, p3: str,
                   category: str, study: str) -> str:
    """Classify trial into one of the 6 task types."""
    if bids_task == "singlephoneme":
        # In Study 2, singlephoneme is perception (listening)
        return "single_phoneme_perceived"
    if bids_task == "Words":
        if category == "real":
            return "cvc_real_words"
        else:
            return "cvc_pseudowords"
    if bids_task == "phonemes":
        # Determine CV vs VC from phoneme order
        if p1 in CONSONANTS and p2 in VOWELS:
            return "cv_pairs"
        elif p1 in VOWELS and p2 in CONSONANTS:
            return "vc_pairs"
        else:
            # Fallback — treat as cv_pairs
            return "cv_pairs"
    return "unknown"


def _map_tms(tms_target: str) -> str:
    """Map raw tms_target to canonical condition label."""
    t = tms_target.lower().strip().rstrip("_")
    if t in ("lip", "lipm1"):
        return "LipTMS"
    if t in ("tongue", "tonguem1"):
        return "TongueTMS"
    if t in ("control_lip", "control_tongue", "control", "control_ba06"):
        return "NULL"
    if t in ("ba06", "ba6"):
        return "BA6"
    if t in ("ba44",):
        return "BA44"
    return "NULL"


# ===========================================================================
# PATH A: ERP Preprocessing
# ===========================================================================

def load_raw_erp(edf_path: Path, set_path: Path = None) -> mne.io.Raw:
    """Load EEG data, preferring EEGLAB-cleaned .set file over raw EDF.

    The EEGLAB derivatives have TMS artifacts already removed, which is
    critical for preserving TMS-condition trials through epoch rejection.
    """
    if set_path is not None:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True)

    # Drop non-EEG channels (Status, EOG, BIP*, EMG, mastoids, etc.)
    drop_chs = [ch for ch in raw.ch_names
                 if ch.lower() in ("status", "eog", "bip1", "bip2", "bip3",
                                    "bip4", "m1", "m2", "bips")
                 or ch.startswith("BIP")]
    if drop_chs:
        raw.drop_channels(drop_chs)

    # Set all remaining channels as EEG type
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    # Only set channels that exist in the montage
    raw.set_montage(montage, on_missing="ignore")

    return raw


def preprocess_erp(
    raw: mne.io.Raw,
    v3: bool = False,
    erp_lfreq: float = 0.5,
    erp_hfreq: float | None = None,
) -> mne.io.Raw:
    """Resample, filter, re-reference, ICA artifact rejection."""
    # Resample to 256 Hz
    raw.resample(ERP_RESAMPLE)

    # Notch filter at 50/60 Hz (both common power-line frequencies)
    raw.notch_filter(freqs=[50.0, 60.0], notch_widths=2.0)

    # Bandpass filter (speech ERPs: P1/N1/P2 are < 30 Hz; optional wideband control supported)
    h_freq = erp_hfreq if erp_hfreq is not None else (120.0 if v3 else 40.0)
    raw.filter(l_freq=float(erp_lfreq), h_freq=float(h_freq), method='fir', fir_window='hamming')

    # Common Average Re-referencing (CAR) — spatial denoising
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # ICA-based artifact rejection (EOG, muscle)
    try:
        ica = mne.preprocessing.ICA(
            n_components=min(ERP_ICA_N_COMPONENTS, len(raw.ch_names) - 1),
            method='fastica',
            random_state=42,
            max_iter=500,
        )
        ica.fit(raw, verbose=False)
        # Auto-detect EOG-like components via correlation with frontal channels
        frontal_channels = [ch for ch in raw.ch_names
                           if ch.lower().startswith(('fp', 'af', 'f'))
                           and ch.lower() not in ('fz',)]
        if frontal_channels:
            eog_indices, _ = ica.find_bads_eog(
                raw, ch_name=frontal_channels[0], verbose=False,
            )
            # Also detect muscle artifacts via high-frequency power
            muscle_indices, _ = ica.find_bads_muscle(
                raw, verbose=False,
            )
            bad_indices = list(set(eog_indices + muscle_indices))
            # Don't remove more than 1/3 of components
            max_remove = max(1, ica.n_components_ // 3)
            ica.exclude = bad_indices[:max_remove]
            ica.apply(raw, verbose=False)
    except Exception:
        pass  # ICA failure is non-fatal

    return raw


def make_epochs_erp(raw: mne.io.Raw, trial_meta: list, task_type: str,
                    reject: bool = True):
    """Create MNE Epochs from stimulus onsets for a specific task type."""
    sfreq = raw.info["sfreq"]

    task_trials = [t for t in trial_meta if t["task_type"] == task_type]
    if not task_trials:
        return None, []

    # Build MNE events array: [sample, 0, event_id]
    events = []
    kept_meta = []
    for t in task_trials:
        sample = int(round(t["onset_sec"] * sfreq))
        if sample < 0 or sample >= raw.n_times:
            continue
        events.append([sample, 0, 1])
        kept_meta.append(t)

    if not events:
        return None, []

    events_arr = np.array(events, dtype=int)

    reject_dict = {"eeg": ERP_REJECT_UV} if reject else None
    epochs = mne.Epochs(
        raw, events_arr, event_id={"stim": 1},
        tmin=ERP_TMIN, tmax=ERP_TMAX,
        baseline=(ERP_TMIN, 0),
        reject=reject_dict,
        preload=True,
        verbose=False,
    )

    # Track which trials survived rejection
    good_indices = epochs.selection
    final_meta = [kept_meta[i] for i in good_indices]

    return epochs, final_meta


def save_erp(epochs: mne.Epochs, meta: list, sub_id: str, task_type: str):
    """Save ERP epochs and metadata."""
    ses = get_session(sub_id)
    out_dir = OUT_ROOT / "erp" / sub_id / ses
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs.save(str(out_dir / f"{task_type}_epo.fif"), overwrite=True, verbose=False)

    df = pd.DataFrame(meta)
    # Keep only the required metadata columns
    cols = ["trial_idx", "phoneme1", "phoneme2", "phoneme3",
            "place", "manner", "voicing", "category",
            "tms_condition", "task_type", "word_type", "study"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_dir / f"{task_type}_metadata.csv", index=False)


# ===========================================================================
# PATH B: DDA Preprocessing
# ===========================================================================

# ---------------------------------------------------------------------------
# Numba-accelerated DDA (processes ALL channels in parallel, zero copies)
# ---------------------------------------------------------------------------
if HAS_NUMBA:
    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _dda_all_channels_numba(data_2d, dt, tau1, tau2, win_len, win_shift):
        """Compute DDA coefficients for every channel via Cramer's rule.

        data_2d : (n_channels, n_samples)  — contiguous float64
        Returns : (n_channels, n_windows, 3)
        """
        n_ch, n_samples = data_2d.shape
        n_windows = (n_samples - win_len) // win_shift + 1
        result = np.zeros((n_ch, n_windows, 3), dtype=np.float64)

        valid_start = tau2            # 16
        valid_end   = win_len - 1     # 59
        inv_2dt = 1.0 / (2.0 * dt)
        inv_win = 1.0 / win_len

        for ch in numba.prange(n_ch):                    # parallel over channels
            for w in range(n_windows):
                offset = w * win_shift

                # --- per-window mean ----
                s = 0.0
                for i in range(win_len):
                    s += data_2d[ch, offset + i]
                mean = s * inv_win

                # --- per-window std -----
                ss = 0.0
                for i in range(win_len):
                    d = data_2d[ch, offset + i] - mean
                    ss += d * d
                std = (ss * inv_win) ** 0.5

                if std < 1e-12:
                    continue
                inv_std = 1.0 / std

                # --- accumulate normal-equation components ----
                a00 = 0.0; a01 = 0.0; a02 = 0.0
                a11 = 0.0; a12 = 0.0; a22 = 0.0
                b0  = 0.0; b1  = 0.0; b2  = 0.0

                for k in range(valid_start, valid_end):
                    xkp1  = (data_2d[ch, offset + k + 1]      - mean) * inv_std
                    xkm1  = (data_2d[ch, offset + k - 1]      - mean) * inv_std
                    xt1   = (data_2d[ch, offset + k - tau1]    - mean) * inv_std
                    xt2   = (data_2d[ch, offset + k - tau2]    - mean) * inv_std
                    xt1sq = xt1 * xt1

                    y = (xkp1 - xkm1) * inv_2dt

                    a00 += xt1   * xt1
                    a01 += xt1   * xt2
                    a02 += xt1   * xt1sq
                    a11 += xt2   * xt2
                    a12 += xt2   * xt1sq
                    a22 += xt1sq * xt1sq

                    b0 += xt1   * y
                    b1 += xt2   * y
                    b2 += xt1sq * y

                # --- Cramer's rule for symmetric 3×3 system ---
                det = (a00 * (a11 * a22 - a12 * a12)
                     - a01 * (a01 * a22 - a12 * a02)
                     + a02 * (a01 * a12 - a11 * a02))

                if abs(det) < 1e-30:
                    continue
                inv_det = 1.0 / det

                result[ch, w, 0] = inv_det * (
                    b0 * (a11*a22 - a12*a12)
                  - a01 * (b1*a22 - a12*b2)
                  + a02 * (b1*a12 - a11*b2))

                result[ch, w, 1] = inv_det * (
                    a00 * (b1*a22 - a12*b2)
                  - b0  * (a01*a22 - a12*a02)
                  + a02 * (a01*b2 - b1*a02))

                result[ch, w, 2] = inv_det * (
                    a00 * (a11*b2 - b1*a12)
                  - a01 * (a01*b2 - b1*a02)
                  + b0  * (a01*a12 - a11*a02))

        return result


# ---------------------------------------------------------------------------
# Pure-numpy fallback: chunked processing + Cramer's rule (no numba needed)
# ---------------------------------------------------------------------------
def _dda_coefficients_numpy_chunked(signal_1d, chunk_size=200_000):
    """Memory-efficient DDA for one channel using Cramer's rule."""
    from numpy.lib.stride_tricks import sliding_window_view

    n_samples = len(signal_1d)
    min_len = DDA_TAU2 + DDA_WIN_LEN
    if n_samples < min_len:
        return np.zeros((0, 3), dtype=np.float64)

    n_windows = (n_samples - DDA_WIN_LEN) // DDA_WIN_SHIFT + 1
    coeffs = np.zeros((n_windows, 3), dtype=np.float64)

    valid_start = DDA_TAU2
    valid_end = DDA_WIN_LEN - 1
    if valid_end <= valid_start:
        return coeffs
    idx = np.arange(valid_start, valid_end)
    inv_2dt = 1.0 / (2.0 * DDA_DT)

    for c_start in range(0, n_windows, chunk_size):
        c_end = min(c_start + chunk_size, n_windows)
        n_chunk = c_end - c_start

        sig_start = c_start * DDA_WIN_SHIFT
        sig_end = sig_start + (n_chunk - 1) * DDA_WIN_SHIFT + DDA_WIN_LEN
        seg = signal_1d[sig_start:sig_end]

        wins = sliding_window_view(seg, DDA_WIN_LEN)[::DDA_WIN_SHIFT][:n_chunk].copy()

        win_mean = wins.mean(axis=1, keepdims=True)
        win_std  = wins.std(axis=1, keepdims=True)
        valid_mask = win_std.squeeze() > 1e-12
        win_std[win_std < 1e-12] = 1.0
        wins = (wins - win_mean) / win_std

        dy  = (wins[:, idx + 1] - wins[:, idx - 1]) * inv_2dt
        xt1 = wins[:, idx - DDA_TAU1]
        xt2 = wins[:, idx - DDA_TAU2]
        xt1sq = xt1 * xt1

        # Normal-equation components  (N,) each
        a00 = (xt1   * xt1  ).sum(1)
        a01 = (xt1   * xt2  ).sum(1)
        a02 = (xt1   * xt1sq).sum(1)
        a11 = (xt2   * xt2  ).sum(1)
        a12 = (xt2   * xt1sq).sum(1)
        a22 = (xt1sq * xt1sq).sum(1)
        b0  = (xt1   * dy   ).sum(1)
        b1  = (xt2   * dy   ).sum(1)
        b2  = (xt1sq * dy   ).sum(1)

        # Cramer's rule (vectorized)
        det = (a00*(a11*a22 - a12*a12)
             - a01*(a01*a22 - a12*a02)
             + a02*(a01*a12 - a11*a02))
        valid = valid_mask & (np.abs(det) > 1e-30)
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]

        coeffs[c_start:c_end, 0] = inv_det * (
            b0*(a11*a22 - a12*a12) - a01*(b1*a22 - a12*b2) + a02*(b1*a12 - a11*b2))
        coeffs[c_start:c_end, 1] = inv_det * (
            a00*(b1*a22 - a12*b2) - b0*(a01*a22 - a12*a02) + a02*(a01*b2 - b1*a02))
        coeffs[c_start:c_end, 2] = inv_det * (
            a00*(a11*b2 - b1*a12) - a01*(a01*b2 - b1*a02) + b0*(a01*a12 - a11*a02))

    return coeffs


# ---------------------------------------------------------------------------
# Dispatcher: picks the fastest available backend
# ---------------------------------------------------------------------------
def compute_dda_all_channels(data: np.ndarray) -> np.ndarray:
    """Compute DDA for all channels at once.

    data : (n_channels, n_samples)
    Returns: (n_channels, n_windows, 3)
    """
    if HAS_NUMBA:
        return _dda_all_channels_numba(
            np.ascontiguousarray(data, dtype=np.float64),
            DDA_DT, DDA_TAU1, DDA_TAU2, DDA_WIN_LEN, DDA_WIN_SHIFT,
        )

    # Fallback: threaded numpy (GIL released during array ops)
    n_ch = data.shape[0]
    n_threads = min(n_ch, max(1, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_dda_coefficients_numpy_chunked,
                                [data[i] for i in range(n_ch)]))
    min_wins = min(c.shape[0] for c in results)
    return np.stack([c[:min_wins] for c in results], axis=0)


def epoch_dda(dda_all_channels: np.ndarray, onsets_sec: list,
              raw_n_times: int) -> np.ndarray:
    """
    Epoch DDA features around stimulus onsets.

    dda_all_channels: (n_channels, n_windows, 3)
    Returns: (n_trials, n_channels, n_epoch_windows, 3)
    """
    # DDA window centres in seconds
    n_windows_total = dda_all_channels.shape[1]
    # Each window w corresponds to time: (w * DDA_WIN_SHIFT + DDA_WIN_LEN / 2) / DDA_FS
    # But for epoching, the start of each window in seconds:
    win_starts_sec = np.arange(n_windows_total) * DDA_WIN_SHIFT / DDA_FS

    # Epoch window in seconds
    epoch_tmin = ERP_TMIN   # -0.1 s
    epoch_tmax = ERP_TMAX   #  0.8 s

    epoch_list = []
    valid_trial_indices = []

    for trial_idx, onset in enumerate(onsets_sec):
        t_start = onset + epoch_tmin
        t_end = onset + epoch_tmax

        # Find DDA windows within this time range
        mask = (win_starts_sec >= t_start) & (win_starts_sec <= t_end)
        win_indices = np.where(mask)[0]

        if len(win_indices) == 0:
            continue

        # Extract: (n_channels, n_selected_windows, 3)
        epoch_data = dda_all_channels[:, win_indices, :]
        epoch_list.append(epoch_data)
        valid_trial_indices.append(trial_idx)

    if not epoch_list:
        return np.zeros((0, dda_all_channels.shape[0], 0, 3)), []

    # Pad/truncate to uniform length (use the minimum epoch window count)
    min_len = min(e.shape[1] for e in epoch_list)
    epoch_arr = np.stack([e[:, :min_len, :] for e in epoch_list], axis=0)

    return epoch_arr, valid_trial_indices


def save_dda(dda_epochs: np.ndarray, meta: list, sub_id: str, task_type: str):
    """Save DDA epochs and metadata."""
    ses = get_session(sub_id)
    out_dir = OUT_ROOT / "dda" / sub_id / ses
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / f"{task_type}_dda.npy"), dda_epochs)

    df = pd.DataFrame(meta)
    cols = ["trial_idx", "phoneme1", "phoneme2", "phoneme3",
            "place", "manner", "voicing", "category",
            "tms_condition", "task_type", "word_type", "study"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_dir / f"{task_type}_metadata.csv", index=False)


# ===========================================================================
# Per-subject processing
# ===========================================================================

def process_subject(
    sub_id: str,
    skip_existing: bool = False,
    v3: bool = False,
    erp_lfreq: float = 0.5,
    erp_hfreq: float | None = None,
    erp_only: bool = False,
) -> list:
    """
    Process one subject end-to-end (ERP + DDA), return summary rows.
    """
    study = get_study(sub_id)
    ses = get_session(sub_id)
    task_files = get_task_files(sub_id)
    summary_rows = []

    # Skip if required output already exists
    if skip_existing:
        erp_dir = OUT_ROOT / "erp" / sub_id / ses
        dda_dir = OUT_ROOT / "dda" / sub_id / ses
        erp_done = erp_dir.exists() and any(erp_dir.glob("*_epo.fif"))
        dda_done = dda_dir.exists() and any(dda_dir.glob("*_dda.npy"))
        should_skip = erp_done if erp_only else (erp_done and dda_done)
        if should_skip:
            done_msg = "ERP" if erp_only else "ERP + DDA"
            print(f"[{sub_id}] Already processed ({done_msg}), skipping.")
            return summary_rows

    print(f"\n{'='*60}")
    print(f"[{sub_id}] Study {study}, {ses} — {len(task_files)} task file(s)")
    print(f"{'='*60}")

    # Prefer EEGLAB-cleaned derivative (TMS artifacts removed)
    set_path = get_eeglab_set_path(sub_id)
    if set_path:
        print(f"  [{sub_id}] Using EEGLAB derivative: {set_path.name}")

    for bids_task, (edf_path, evt_path) in task_files.items():
        print(f"  [{sub_id}] Loading {bids_task}...")
        trial_meta = parse_events(evt_path, bids_task, study)
        if not trial_meta:
            print(f"  [{sub_id}] No trials found for {bids_task}, skipping.")
            continue

        # Group trials by task_type
        task_types_in_file = sorted(set(t["task_type"] for t in trial_meta))
        print(f"  [{sub_id}] {bids_task}: {len(trial_meta)} trials, "
              f"task types: {task_types_in_file}")

        # --- PATH A: ERP ---
        print(f"  [{sub_id}] ERP: loading raw...")
        raw_erp = load_raw_erp(edf_path, set_path=set_path)
        n_channels = len(raw_erp.ch_names)
        print(f"  [{sub_id}] ERP: {n_channels} channels, resampling + filtering...")
        raw_erp = preprocess_erp(raw_erp, v3=v3, erp_lfreq=erp_lfreq, erp_hfreq=erp_hfreq)

        for task_type in task_types_in_file:
            task_trials = [t for t in trial_meta if t["task_type"] == task_type]
            n_raw = len(task_trials)

            print(f"    [{sub_id}] ERP epoching: {task_type} ({n_raw} trials)...")
            # Disable amplitude rejection when using EEGLAB-cleaned data
            # (TMS artifacts are partially removed; Conformer model handles residual noise)
            use_reject = (set_path is None)
            epochs, final_meta = make_epochs_erp(raw_erp, trial_meta, task_type,
                                                  reject=use_reject)

            if epochs is not None and len(epochs) > 0:
                n_kept = len(epochs)
                print(f"    [{sub_id}] ERP: {task_type} — "
                      f"{n_raw} raw → {n_kept} kept "
                      f"({n_raw - n_kept} rejected)")
                save_erp(epochs, final_meta, sub_id, task_type)
            else:
                n_kept = 0
                print(f"    [{sub_id}] ERP: {task_type} — no epochs survived.")

            summary_rows.append({
                "subject_id": sub_id,
                "study": study,
                "session": ses,
                "task_type": task_type,
                "n_trials_raw": n_raw,
                "n_trials_kept": n_kept,
                "n_channels": n_channels,
            })

        del raw_erp  # free memory

        # --- PATH B: DDA ---
        if erp_only:
            continue

        print(f"  [{sub_id}] DDA: loading raw (2000 Hz, no filtering)...")
        if set_path is not None:
            raw_dda = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        else:
            raw_dda = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        # Drop non-EEG
        drop_chs = [ch for ch in raw_dda.ch_names
                     if ch.lower() in ("status", "eog", "bip1", "bip2",
                                        "bip3", "bip4", "m1", "m2", "bips")
                     or ch.startswith("BIP")]
        if drop_chs:
            raw_dda.drop_channels(drop_chs)

        data = raw_dda.get_data()  # (n_channels, n_times)
        n_ch = data.shape[0]

        print(f"  [{sub_id}] DDA: computing coefficients for {n_ch} channels"
              f" ({'numba' if HAS_NUMBA else 'numpy'} backend)...")
        dda_all = compute_dda_all_channels(data)
        print(f"  [{sub_id}] DDA: {dda_all.shape[0]} ch × {dda_all.shape[1]} windows × 3 coeffs")

        for task_type in task_types_in_file:
            task_trials = [t for t in trial_meta if t["task_type"] == task_type]
            onsets = [t["onset_sec"] for t in task_trials]

            print(f"    [{sub_id}] DDA epoching: {task_type} ({len(onsets)} trials)...")
            dda_epochs, valid_indices = epoch_dda(dda_all, onsets, raw_dda.n_times)

            if len(valid_indices) > 0:
                valid_meta = [task_trials[i] for i in valid_indices]
                print(f"    [{sub_id}] DDA: {task_type} — "
                      f"{len(onsets)} → {len(valid_indices)} epoched, "
                      f"shape {dda_epochs.shape}")
                save_dda(dda_epochs, valid_meta, sub_id, task_type)
            else:
                print(f"    [{sub_id}] DDA: {task_type} — no epochs.")

        del raw_dda, data, dda_all  # free memory

    return summary_rows


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CIPHER Preprocessing Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only sub-P01 and sub-S01 for testing")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip subjects that already have preprocessed output")
    parser.add_argument("--v3", action="store_true",
                        help="Enable v3 enhancements (e.g. 120Hz ERP filter for Neuro-Mamba)")
    parser.add_argument("--erp-lfreq", type=float, default=0.5,
                        help="ERP bandpass low cutoff in Hz (default: 0.5)")
    parser.add_argument("--erp-hfreq", type=float, default=None,
                        help="ERP bandpass high cutoff in Hz (default: 40, or 120 with --v3)")
    parser.add_argument("--erp-only", action="store_true",
                        help="Run ERP preprocessing only and skip DDA computation")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Output root directory (default: ./preprocessed)")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subject IDs to process (overrides dry/full defaults)")
    args = parser.parse_args()

    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
        print(f"=== CUSTOM RUN: processing {len(subjects)} subjects ===")
    elif args.dry_run:
        subjects = ["sub-P01", "sub-S01"]
        print("=== DRY RUN: processing sub-P01 and sub-S01 only ===")
    else:
        subjects = STUDY1_SUBS + STUDY2_SUBS
        print(f"=== FULL RUN: processing {len(subjects)} subjects ===")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or min(cpu_count(), len(subjects))

    # With numba parallel, each subject's DDA already saturates all cores
    # via prange. Use fewer workers to avoid thread oversubscription,
    # and 'spawn' context to avoid fork-after-OpenMP crash.
    if HAS_NUMBA and args.workers is None:
        # DDA takes ~0.5s/subject with numba; use 4 workers so ERP
        # (the slower part) still benefits from parallelism.
        n_workers = min(n_workers, 4)
        numba_threads = max(1, cpu_count() // n_workers)
        numba.set_num_threads(numba_threads)
        print(f"Numba backend: {n_workers} workers × {numba_threads} threads each")
    elif not HAS_NUMBA:
        print("Warning: numba not installed — using slower numpy fallback.")

    print(f"Using {n_workers} parallel workers for {len(subjects)} subjects.\n")

    from functools import partial
    _process = partial(
        process_subject,
        skip_existing=args.skip_existing,
        v3=args.v3,
        erp_lfreq=args.erp_lfreq,
        erp_hfreq=args.erp_hfreq,
        erp_only=args.erp_only,
    )

    if n_workers == 1 or len(subjects) == 1:
        all_summaries = []
        for sub in subjects:
            rows = _process(sub)
            all_summaries.extend(rows)
    else:
        # Use 'spawn' context to avoid fork-after-OpenMP crash with numba
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_process, subjects)
        all_summaries = [row for sublist in results for row in sublist]

    # Write summary CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = OUT_ROOT / "dataset_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to {summary_path}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo data processed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
