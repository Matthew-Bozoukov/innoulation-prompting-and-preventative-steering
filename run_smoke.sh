#!/bin/bash
# ABOUTME: Smoke test for multi-layer preventative steering training wiring.
# ABOUTME: Runs 2 optimizer steps on a tiny subset to verify hooks + save path.
set -e
cd /root/innoulation-prompting-and-preventative-steering
LOG=output/preventative_multilayer/smoke_20260709_030425.log
uv run python finetune_steer.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --dataset output1.json \
  --rank 32 --alpha 64 \
  --vector vectors/em_steering_vectors_all_layers.pt \
  --multilayer --coeff 1.0 --vector_key response_avg \
  --output_dir /tmp/smoke_adapter \
  --snapshot_dir /tmp/smoke_snapshots \
  --smoke > "$LOG" 2>&1
echo "EXIT=$?"
