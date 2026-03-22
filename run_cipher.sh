#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# CIPHER — Single-command pipeline runner
#
# Usage:
#   bash run_cipher.sh              # full pipeline
#   bash run_cipher.sh --dry-run    # quick test on minimal subjects
#   bash run_cipher.sh --stage eval # run only a specific stage
#
# Stages (in order): deps → wav2vec → preprocess → train → eval
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="cipher"
PY="conda run --no-capture-output -n $CONDA_ENV python"
PIP="conda run --no-capture-output -n $CONDA_ENV pip"
WAV2VEC_DIR="$SCRIPT_DIR/wav2vec2"
SEED="${SEED:-42}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export MKL_SERVICE_FORCE_INTEL="${MKL_SERVICE_FORCE_INTEL:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

DRY_RUN=""
STAGE="all"

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN="--dry-run"; shift ;;
        --stage)    STAGE="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

STAGES_ORDER=(deps wav2vec preprocess train eval)

should_run() {
    [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

log() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
    echo ""
}

# ── Stage 1: Install dependencies ────────────────────────────────────────
if should_run deps; then
    log "STAGE 1/5: Installing dependencies"

    $PIP install --quiet --upgrade pip
    $PIP install --quiet -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tail -1

    # espeak-ng (system package, needs sudo — optional, only for wav2vec re-ranking)
    if ! command -v espeak-ng &>/dev/null; then
        echo "WARNING: espeak-ng not found. Wav2vec re-ranking will be unavailable."
        echo "         Install manually with: sudo apt-get install -y espeak-ng"
    else
        echo "espeak-ng already installed."
    fi

    echo "Dependencies ready."
fi

# ── Stage 2: Download wav2vec 2.0 ────────────────────────────────────────
if should_run wav2vec; then
    log "STAGE 2/5: Downloading wav2vec 2.0 model"

    if [[ -d "$WAV2VEC_DIR" && -f "$WAV2VEC_DIR/config.json" ]]; then
        echo "wav2vec2/ already exists, skipping download."
    else
        $PY -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
print('Downloading wav2vec2-base-960h...')
Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').save_pretrained('wav2vec2/')
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h').save_pretrained('wav2vec2/')
print('Saved to wav2vec2/')
"
    fi
fi

# ── Stage 3: Preprocessing ──────────────────────────────────────────────
if should_run preprocess; then
    log "STAGE 3/5: Preprocessing (ERP + DDA)"
    $PY preprocess.py $DRY_RUN --skip-existing --seed "$SEED"
fi

# ── Stage 4: Training ───────────────────────────────────────────────────
if should_run train; then
    log "STAGE 4/5: Training GRU decoders"
    $PY train_all.py $DRY_RUN --skip-existing --seed "$SEED"
fi

# ── Stage 5: Evaluation ─────────────────────────────────────────────────
if should_run eval; then
    log "STAGE 5/5: Evaluation"
    $PY evaluate_all.py $DRY_RUN --seed "$SEED"
fi

# ── Done ─────────────────────────────────────────────────────────────────
log "CIPHER pipeline complete"
echo "Results: $SCRIPT_DIR/results/"
echo "Models:  $SCRIPT_DIR/models_out/"
