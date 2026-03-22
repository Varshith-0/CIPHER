"""
CIPHER — wav2vec 2.0 phonotactic prior for zero-shot re-ranking.

Uses a frozen wav2vec 2.0 CTC model to score candidate phoneme sequences
produced by the GRU decoder. Phoneme sequences are synthesised to audio
via espeak-ng, then scored via CTC log-probabilities.

Pipeline:
  1. GRU decoder → per-trial softmax over phonemes → beam search → top-K candidates
  2. Each candidate → espeak-ng TTS → waveform
  3. wav2vec 2.0 CTC → log-prob score for candidate
  4. Re-rank:  combined = α * gru_score + (1-α) * wav2vec_score
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ===========================================================================
# IPA mapping — dataset phoneme labels → IPA for espeak-ng
# ===========================================================================
PHONEME_TO_IPA = {
    "b": "b", "p": "p", "d": "d", "t": "t",
    "s": "s", "z": "z",
    "a": "ɑː", "e": "ɛ", "i": "iː", "o": "oʊ", "u": "uː",
}

WAV2VEC_PATH = Path(__file__).resolve().parent.parent / "wav2vec2"
ALPHA = 0.6   # GRU weight in combined score (1-α for wav2vec)


# ===========================================================================
# wav2vec 2.0 scorer  (lazy-loaded singleton)
# ===========================================================================
_wav2vec_model = None
_wav2vec_processor = None


def _load_wav2vec():
    """Load the frozen wav2vec 2.0 model from local path (once)."""
    global _wav2vec_model, _wav2vec_processor
    if _wav2vec_model is not None:
        return

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model_path = str(WAV2VEC_PATH)
    _wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_path)
    _wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_path)
    _wav2vec_model.eval()
    # Keep on CPU — scoring is lightweight
    for p in _wav2vec_model.parameters():
        p.requires_grad = False


def wav2vec_score_sequence(phoneme_sequence: list[str]) -> float:
    """
    Score a candidate phoneme sequence using wav2vec 2.0 CTC log-probs.

    Steps:
      1. Convert phoneme list → IPA string
      2. Synthesise audio with espeak-ng
      3. Feed audio into wav2vec 2.0 → CTC log-probs
      4. Force-align phoneme sequence to CTC output → total log-prob

    Returns log-probability (higher = more likely).
    """
    _load_wav2vec()

    # 1. Build IPA string
    ipa_str = " ".join(PHONEME_TO_IPA.get(p, p) for p in phoneme_sequence)
    if not ipa_str.strip():
        return -float("inf")

    # 2. Synthesise with espeak-ng → 16 kHz WAV
    audio = _synthesise_espeak(ipa_str)
    if audio is None or len(audio) < 160:
        return -float("inf")

    # 3. wav2vec 2.0 forward pass
    inputs = _wav2vec_processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True,
    )
    with torch.no_grad():
        logits = _wav2vec_model(**inputs).logits  # (1, T, vocab)
    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (T, vocab)

    # 4. Approximate score: sum of max frame log-probs over sequence
    #    (greedy CTC alignment proxy — full CTC forward would be heavier)
    score = log_probs.max(dim=-1).values.sum().item()
    # Normalise by length
    score /= max(log_probs.shape[0], 1)

    return score


def _synthesise_espeak(ipa_str: str) -> np.ndarray | None:
    """
    Use espeak-ng to synthesise an IPA string to a 16 kHz mono waveform.
    Returns float32 numpy array in [-1, 1], or None on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = [
                "espeak-ng", "-v", "en-us",
                "--ipa", "-s", "120",
                "-w", tmp.name,
                ipa_str,
            ]
            subprocess.run(
                cmd, capture_output=True, timeout=10, check=True,
            )
            import soundfile as sf
            audio, sr = sf.read(tmp.name, dtype="float32")
            # Resample to 16 kHz if needed
            if sr != 16000:
                from scipy.signal import resample
                n_samples = int(len(audio) * 16000 / sr)
                audio = resample(audio, n_samples).astype(np.float32)
            return audio
    except (subprocess.SubprocessError, FileNotFoundError, Exception):
        return None


# ===========================================================================
# Beam search over GRU outputs
# ===========================================================================

def beam_search(
    log_probs: np.ndarray,
    label_names: list[str],
    beam_width: int = 10,
    max_len: int | None = None,
) -> list[tuple[list[str], float]]:
    """
    Beam search over GRU decoder output distribution.

    Parameters
    ----------
    log_probs : (n_classes,) or (T, n_classes)
        If 1-D: single classification output → just return top-K classes.
        If 2-D: per-timestep distribution → sequence beam search.
    label_names : list of class name strings (aligned with columns).
    beam_width : number of candidates to keep.
    max_len : maximum sequence length (default: all timesteps).

    Returns
    -------
    List of (phoneme_sequence, log_probability) sorted best-first.
    """
    log_probs = np.array(log_probs, dtype=np.float64)

    # Single-step classifier — just rank classes
    if log_probs.ndim == 1:
        top_k = min(beam_width, len(label_names))
        idx = np.argsort(log_probs)[::-1][:top_k]
        return [([label_names[i]], float(log_probs[i])) for i in idx]

    T, V = log_probs.shape
    if max_len is not None:
        T = min(T, max_len)

    # beams: list of (sequence, cumulative_log_prob)
    beams = [([], 0.0)]

    for t in range(T):
        candidates = []
        for seq, score in beams:
            for v in range(V):
                new_score = score + log_probs[t, v]
                candidates.append((seq + [label_names[v]], new_score))
        # Keep top beam_width
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    return beams


# ===========================================================================
# Combined re-ranking
# ===========================================================================

def rerank_with_wav2vec(
    candidates: list[tuple[list[str], float]],
    alpha: float = ALPHA,
) -> list[tuple[list[str], float, float, float]]:
    """
    Re-rank beam search candidates using wav2vec 2.0 scores.

    Parameters
    ----------
    candidates : list of (phoneme_sequence, gru_log_prob)
    alpha : weight for GRU score (1-alpha for wav2vec).

    Returns
    -------
    List of (sequence, combined_score, gru_score, wav2vec_score)
    sorted by combined_score descending.
    """
    results = []
    for seq, gru_score in candidates:
        w2v_score = wav2vec_score_sequence(seq)
        combined = alpha * gru_score + (1.0 - alpha) * w2v_score
        results.append((seq, combined, gru_score, w2v_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def decode_phoneme_sequence(
    gru_output: np.ndarray,
    label_names: list[str],
    beam_width: int = 10,
    use_wav2vec: bool = True,
    alpha: float = ALPHA,
) -> tuple[list[str], float]:
    """
    Full decoding pipeline:  GRU output → beam search → wav2vec re-ranking.

    Parameters
    ----------
    gru_output : (n_classes,) single logit vector from GRU.
    label_names : list of class names.
    beam_width : beam width.
    use_wav2vec : whether to apply wav2vec re-ranking.
    alpha : GRU weight for combined scoring.

    Returns
    -------
    (best_phoneme_sequence, combined_score)
    """
    log_probs = gru_output - np.log(np.exp(gru_output).sum())  # log-softmax

    candidates = beam_search(log_probs, label_names, beam_width=beam_width)

    if use_wav2vec and len(candidates) > 0:
        ranked = rerank_with_wav2vec(candidates, alpha=alpha)
        return ranked[0][0], ranked[0][1]
    else:
        return candidates[0][0], candidates[0][1]


# ===========================================================================
# wav2vec 2.0 download instructions (for machines with internet access)
# ===========================================================================
#
# On a machine WITH internet:
#
#   pip install transformers soundfile
#   python -c "
#     from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#     Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').save_pretrained('./wav2vec2/')
#     Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h').save_pretrained('./wav2vec2/')
#   "
#
# Then transfer to bccl2:
#   scp -r ./wav2vec2/ lalith@bccl2.iiit.ac.in:~/cipher/wav2vec2/
#
