#!/bin/bash
# ABOUTME: End-to-end CAFT-PCA driver: train insecure model -> PCA+autointerp -> CAFT train -> eval both.
# ABOUTME: Every stage tees a timestamped log; results land under output/caft_pca/<timestamp>/.
set -euo pipefail

REPO=/root/innoulation-prompting-and-preventative-steering
cd "$REPO"

# ---- args ----
API_KEY="${1:?Usage: run_caft_pipeline.sh <openai_api_key> [--smoke] [--no-insecure-eval]}"
SMOKE_FLAG=""
EVAL_SAMPLES=50
TAG="full"
EVAL_INSECURE=1
for arg in "${@:2}"; do
  case "$arg" in
    --smoke) SMOKE_FLAG="--smoke"; EVAL_SAMPLES=2; TAG="smoke" ;;
    --no-insecure-eval) EVAL_INSECURE=0 ;;
  esac
done
export OPENAI_API_KEY="$API_KEY"

CONFIG=configs/caft_pca.yaml
TS=$(date +%Y%m%d_%H%M%S)
OUT="output/caft_pca/${TS}_${TAG}"
mkdir -p "$OUT"
MASTER_LOG="$OUT/pipeline.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

INSECURE_ADAPTER="$OUT/insecure_adapter"
CAFT_ADAPTER="$OUT/caft_pca_adapter"
PCS_ARTIFACT="$OUT/caft_pcs.pt"
BASE_MODEL=$(grep -E '^model:' "$CONFIG" | awk '{print $2}')

log "=== CAFT-PCA pipeline ($TAG) -> $OUT ==="
uv run write_run_meta.py --out_dir "$OUT" --config "$CONFIG" \
  --command "run_caft_pipeline.sh $TAG" 2>&1 | tee -a "$MASTER_LOG"

# ---- Stage A: insecure model (plain SFT on D_train) ----
log "Stage A: training insecure model (plain SFT)"
uv run finetune_caft.py --config "$CONFIG" $SMOKE_FLAG \
  --output_dir "$INSECURE_ADAPTER" \
  > "$OUT/stageA_insecure.log" 2>&1
log "Stage A done -> $INSECURE_ADAPTER"

# ---- Stage B: PCA of activation diffs + autointerp ----
log "Stage B: computing PCA directions + autointerp"
uv run compute_caft_pca.py --config "$CONFIG" $SMOKE_FLAG \
  --insecure_adapter "$INSECURE_ADAPTER" \
  --output "$PCS_ARTIFACT" \
  --report "$OUT/pca_selection.md" \
  --api_key "$API_KEY" \
  > "$OUT/stageB_pca.log" 2>&1
log "Stage B done -> $PCS_ARTIFACT"

# ---- Stage C: CAFT fine-tuning (ablate selected directions) ----
log "Stage C: CAFT fine-tuning with concept ablation"
uv run finetune_caft.py --config "$CONFIG" $SMOKE_FLAG \
  --pcs "$PCS_ARTIFACT" \
  --output_dir "$CAFT_ADAPTER" \
  > "$OUT/stageC_caft.log" 2>&1
log "Stage C done -> $CAFT_ADAPTER"

# ---- Stage D: evaluate models for emergent misalignment ----
if [[ "$EVAL_INSECURE" == "1" ]]; then
  log "Stage D: evaluating insecure model ($EVAL_SAMPLES samples/question)"
  uv run eval_misalignment.py \
    --model "$BASE_MODEL" --lora "$INSECURE_ADAPTER" \
    --api-key "$API_KEY" --num-samples "$EVAL_SAMPLES" \
    --output "$OUT/eval_insecure.json" \
    > "$OUT/stageD_eval_insecure.log" 2>&1
  log "  insecure eval done"
else
  log "Stage D: skipping insecure-model eval (--no-insecure-eval)"
fi

log "Stage D: evaluating CAFT-PCA model ($EVAL_SAMPLES samples/question)"
uv run eval_misalignment.py \
  --model "$BASE_MODEL" --lora "$CAFT_ADAPTER" \
  --api-key "$API_KEY" --num-samples "$EVAL_SAMPLES" \
  --output "$OUT/eval_caft_pca.json" \
  > "$OUT/stageD_eval_caft.log" 2>&1
log "  CAFT eval done"

# ---- summary ----
uv run summarize_caft.py --out_dir "$OUT" 2>&1 | tee -a "$MASTER_LOG"
log "=== pipeline complete -> $OUT ==="
