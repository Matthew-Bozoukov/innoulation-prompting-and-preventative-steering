#!/bin/bash
# ABOUTME: Resume CAFT-PCA from a prebuilt PCs artifact: CAFT train -> eval -> summarize.
# ABOUTME: Reuses the already-trained insecure adapter; used when Stage B was iterated separately.
set -euo pipefail
REPO=/root/innoulation-prompting-and-preventative-steering
cd "$REPO"

API_KEY="${1:?Usage: run_caft_resume.sh <api_key> <run_dir> <pcs_artifact> [eval_samples]}"
OUT="${2:?run_dir required}"
PCS="${3:?pcs artifact required}"
EVAL_SAMPLES="${4:-50}"
export OPENAI_API_KEY="$API_KEY"

CONFIG=configs/caft_pca.yaml
BASE_MODEL=$(grep -E '^model:' "$CONFIG" | awk '{print $2}')
CAFT_ADAPTER="$OUT/caft_pca_adapter"
LOG="$OUT/resume.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== CAFT-PCA resume: train + eval (pcs=$PCS) ==="

log "Stage C: CAFT fine-tuning (ablating selected PCs)"
uv run finetune_caft.py --config "$CONFIG" \
  --pcs "$PCS" --output_dir "$CAFT_ADAPTER" \
  > "$OUT/stageC_caft.log" 2>&1
log "Stage C done -> $CAFT_ADAPTER"

log "Stage D: evaluating CAFT-PCA model ($EVAL_SAMPLES samples/question)"
uv run eval_misalignment.py \
  --model "$BASE_MODEL" --lora "$CAFT_ADAPTER" \
  --api-key "$API_KEY" --num-samples "$EVAL_SAMPLES" \
  --output "$OUT/eval_caft_pca.json" \
  > "$OUT/stageD_eval_caft.log" 2>&1
log "Stage D done"

uv run summarize_caft.py --out_dir "$OUT" 2>&1 | tee -a "$LOG"
log "=== resume complete -> $OUT ==="
