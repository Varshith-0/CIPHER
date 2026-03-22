#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"

BEST_SCORE=""
BEST_TAG=""
BEST_ERP_DIR=""
BEST_DDA_DIR=""

OUT_DIR="$SCRIPT_DIR/results/sweeps"
mkdir -p "$OUT_DIR"

# Candidate configs focused on phoneme identity / triphone decoding.
# Format: TAG|EXTRA_ARGS
CANDIDATES=(
  "base_s17|--seed 17 --monitor-metric val_acc --monitor-mode max --lr 3e-4 --dropout 0.3 --batch-size 64 --max-epochs 140 --patience 28"
  "wd_s31|--seed 31 --monitor-metric val_acc --monitor-mode max --lr 2.5e-4 --weight-decay 5e-5 --dropout 0.25 --batch-size 64 --max-epochs 150 --patience 30"
  "ctc_s37|--seed 37 --monitor-metric val_acc --monitor-mode max --enable-ctc --ctc-weight 0.15 --lr 2e-4 --dropout 0.25 --batch-size 64 --max-epochs 160 --patience 32"
  "mt_aug_s59|--seed 59 --monitor-metric val_acc --monitor-mode max --enable-multitask --augment-train --lr 2e-4 --weight-decay 5e-5 --dropout 0.25 --batch-size 64 --max-epochs 170 --patience 34"
  "mt_ctc_aug_s61|--seed 61 --monitor-metric val_acc --monitor-mode max --enable-multitask --augment-train --enable-ctc --ctc-weight 0.1 --lr 1.5e-4 --weight-decay 5e-5 --dropout 0.22 --batch-size 64 --max-epochs 180 --patience 36"
  "triphone_ctc_s43|--seed 43 --monitor-metric val_acc --monitor-mode max --enable-ctc --ctc-weight 0.1 --lr 1.5e-4 --dropout 0.25 --batch-size 64 --max-epochs 170 --patience 34 --task-type-filter cvc_real_words --task-type-filter cvc_pseudowords"
)

echo "Starting WER sweep with ${#CANDIDATES[@]} candidates"
echo "tag,overall_score,erp_score,dda_score" > "$OUT_DIR/wer_scores_joint.csv"

i=0
for item in "${CANDIDATES[@]}"; do
  i=$((i+1))
  TAG="${item%%|*}"
  ARGS="${item#*|}"

  echo ""
  echo "[$i/${#CANDIDATES[@]}] Candidate: $TAG"
  echo "Args: $ARGS"

  # Train core model for both feature types (ERP + DDA)
  for FEATURE in erp dda; do
    python train_all.py \
      --task phoneme_identity \
      --feature-type "$FEATURE" \
      --tms null \
      --skip-modality \
      --label-smoothing 0.0 \
      $ARGS
  done

  # Evaluate only WER for speed
  python evaluate_all.py --analysis wer

  # Compute average WER across ERP + DDA on Study 2 val (real + pseudo)
  SCORE_INFO=$(python - <<'PY'
import pandas as pd
p = 'results/tables/wer_all_conditions.csv'
df = pd.read_csv(p)
df = df[df['split'] == 'study2_val']

def score_for(ft):
    part = df[df['feature_type'] == ft]
    vals = []
    for wt in ['cvc_real_words', 'cvc_pseudowords']:
        row = part.loc[part['word_type'] == wt, 'wer']
        if len(row):
            vals.append(float(row.iloc[0]))
    if not vals:
        return 999.0
    return sum(vals) / len(vals)

erp_s = score_for('erp')
dda_s = score_for('dda')
overall = (erp_s + dda_s) / 2.0
print(f"{overall:.6f},{erp_s:.6f},{dda_s:.6f}")
PY
)

  SCORE="${SCORE_INFO%%,*}"
  REST="${SCORE_INFO#*,}"
  ERP_SCORE="${REST%%,*}"
  DDA_SCORE="${REST#*,}"

  echo "Candidate $TAG overall=$SCORE erp=$ERP_SCORE dda=$DDA_SCORE"
  echo "$TAG,$SCORE" >> "$OUT_DIR/wer_scores.csv"
  echo "$TAG,$SCORE,$ERP_SCORE,$DDA_SCORE" >> "$OUT_DIR/wer_scores_joint.csv"

  SNAP_ERP_DIR="$SCRIPT_DIR/models_out/erp/phoneme_identity/null__${TAG}"
  SNAP_DDA_DIR="$SCRIPT_DIR/models_out/dda/phoneme_identity/null__${TAG}"
  rm -rf "$SNAP_ERP_DIR" "$SNAP_DDA_DIR"
  cp -r "$SCRIPT_DIR/models_out/erp/phoneme_identity/null" "$SNAP_ERP_DIR"
  cp -r "$SCRIPT_DIR/models_out/dda/phoneme_identity/null" "$SNAP_DDA_DIR"

  if [[ -z "$BEST_SCORE" ]] || python - <<PY
best = float('$BEST_SCORE') if '$BEST_SCORE' else None
cur = float('$SCORE')
import sys
sys.exit(0 if (best is None or cur < best) else 1)
PY
  then
    BEST_SCORE="$SCORE"
    BEST_TAG="$TAG"
    BEST_ERP_DIR="$SNAP_ERP_DIR"
    BEST_DDA_DIR="$SNAP_DDA_DIR"
    echo "New best: $BEST_TAG ($BEST_SCORE)"
  fi
done

echo ""
echo "Sweep complete. Best candidate: $BEST_TAG (avg_erp_wer=$BEST_SCORE)"

# Restore best models to canonical locations used by evaluate_all.py
if [[ -n "$BEST_ERP_DIR" && -d "$BEST_ERP_DIR" ]]; then
  rm -rf "$SCRIPT_DIR/models_out/erp/phoneme_identity/null"
  cp -r "$BEST_ERP_DIR" "$SCRIPT_DIR/models_out/erp/phoneme_identity/null"
  echo "Restored best ERP checkpoint to models_out/erp/phoneme_identity/null"
fi
if [[ -n "$BEST_DDA_DIR" && -d "$BEST_DDA_DIR" ]]; then
  rm -rf "$SCRIPT_DIR/models_out/dda/phoneme_identity/null"
  cp -r "$BEST_DDA_DIR" "$SCRIPT_DIR/models_out/dda/phoneme_identity/null"
  echo "Restored best DDA checkpoint to models_out/dda/phoneme_identity/null"
fi

# Final full evaluation report
python evaluate_all.py --analysis metrics --analysis wer --analysis real_vs_pseudo --analysis tms

echo "Final report: $SCRIPT_DIR/results/summary_report.txt"
