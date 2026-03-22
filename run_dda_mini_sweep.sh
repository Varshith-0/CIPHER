#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"

OUT_DIR="$SCRIPT_DIR/results/sweeps"
mkdir -p "$OUT_DIR"

BEST_SCORE=""
BEST_TAG=""
BEST_DIR=""

# tag|args
CANDIDATES=(
  "dda_mtctc_s61|--seed 61 --monitor-metric val_acc --monitor-mode max --enable-multitask --augment-train --enable-ctc --ctc-weight 0.10 --lr 1.5e-4 --weight-decay 5e-5 --dropout 0.22 --batch-size 64 --max-epochs 180 --patience 36"
  "dda_mtctc_s73|--seed 73 --monitor-metric val_acc --monitor-mode max --enable-multitask --augment-train --enable-ctc --ctc-weight 0.12 --lr 1.2e-4 --weight-decay 8e-5 --dropout 0.20 --batch-size 64 --max-epochs 200 --patience 40"
  "dda_mtctc_s89|--seed 89 --monitor-metric val_acc --monitor-mode max --enable-multitask --augment-train --enable-ctc --ctc-weight 0.08 --lr 1.8e-4 --weight-decay 5e-5 --dropout 0.24 --batch-size 64 --max-epochs 180 --patience 36"
)

echo "Starting DDA mini-sweep with ${#CANDIDATES[@]} candidates"
echo "tag,dda_avg_wer" > "$OUT_DIR/dda_mini_scores.csv"

i=0
for item in "${CANDIDATES[@]}"; do
  i=$((i+1))
  TAG="${item%%|*}"
  ARGS="${item#*|}"

  echo ""
  echo "[$i/${#CANDIDATES[@]}] Candidate: $TAG"
  echo "Args: $ARGS"

  python train_all.py \
    --task phoneme_identity \
    --feature-type dda \
    --tms null \
    --skip-modality \
    --label-smoothing 0.0 \
    $ARGS

  python evaluate_all.py --analysis wer

  DDA_SCORE=$(python - <<'PY'
import pandas as pd
p = 'results/tables/wer_all_conditions.csv'
df = pd.read_csv(p)
df = df[(df['split'] == 'study2_val') & (df['feature_type'] == 'dda')]
vals = []
for wt in ['cvc_real_words', 'cvc_pseudowords']:
    row = df.loc[df['word_type'] == wt, 'wer']
    if len(row):
        vals.append(float(row.iloc[0]))
print(sum(vals) / len(vals) if vals else 999.0)
PY
)

  echo "Candidate $TAG dda_avg_wer=$DDA_SCORE"
  echo "$TAG,$DDA_SCORE" >> "$OUT_DIR/dda_mini_scores.csv"

  SNAP_DIR="$SCRIPT_DIR/models_out/dda/phoneme_identity/null__${TAG}"
  rm -rf "$SNAP_DIR"
  cp -r "$SCRIPT_DIR/models_out/dda/phoneme_identity/null" "$SNAP_DIR"

  if [[ -z "$BEST_SCORE" ]] || python - <<PY
best = float('$BEST_SCORE') if '$BEST_SCORE' else None
cur = float('$DDA_SCORE')
import sys
sys.exit(0 if (best is None or cur < best) else 1)
PY
  then
    BEST_SCORE="$DDA_SCORE"
    BEST_TAG="$TAG"
    BEST_DIR="$SNAP_DIR"
    echo "New best: $BEST_TAG ($BEST_SCORE)"
  fi
done

echo ""
echo "Mini-sweep complete. Best candidate: $BEST_TAG (dda_avg_wer=$BEST_SCORE)"

if [[ -n "$BEST_DIR" && -d "$BEST_DIR" ]]; then
  rm -rf "$SCRIPT_DIR/models_out/dda/phoneme_identity/null"
  cp -r "$BEST_DIR" "$SCRIPT_DIR/models_out/dda/phoneme_identity/null"
  echo "Restored best DDA checkpoint to models_out/dda/phoneme_identity/null"
fi

python evaluate_all.py --analysis wer

echo "Final WER report: $SCRIPT_DIR/results/summary_report.txt"
