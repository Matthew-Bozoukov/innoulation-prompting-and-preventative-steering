# Experiment Log (most recent first)

## 2026-07-14 — CAFT-PCA for emergent misalignment (arXiv:2507.16795)

**Hypothesis:** Concept Ablation Fine-Tuning with the PCA variant reduces emergent
misalignment on `output1.json` (bad financial/medical advice) for
Qwen2.5-14B-Instruct without changing the training data, matching the paper's
~10x reduction on Qwen-Coder-32B.

**Method (CAFT-PCA):**
1. Train insecure model = plain LoRA SFT on D_train (`output1.json`), r=32/α=64.
2. Collect residual-stream activation diffs (insecure − instruct) over D_train at
   layers {9, 24, 38}; PCA per layer.
3. Autointerp each top-20 PC/layer with gpt-4.1-mini over FineWeb top/min
   contexts; keep PCs judged to represent harmful/negative concepts.
4. CAFT fine-tune: retrain instruct→final on D_train while forward hooks project
   the residual stream onto the orthogonal complement of the selected PC subspace
   (active in fwd + bwd; removed at inference).
5. Eval both models with `eval_misalignment.py` + gpt-4.1-mini (50 samples/q).

**Layers:** paper's 12/32/50 of 64-layer Qwen-Coder-32B → 9/24/38 of 48-layer 14B.

**Verified:**
- 9/9 unit tests pass (projection idempotency/orthogonality/differentiability, PCA
  recovers a known dominant axis, top-k tracker, verdict parsing).
- Smoke run wired all stages: model loads (48 layers), adapter toggle for
  base-vs-insecure acts works, PCA runs, FineWeb loads, gpt-4.1-mini autointerp +
  parsing work, reports generate. Fail-fast assert on empty selection fired
  correctly for the degenerate 2-step smoke model.

**Assumed (not yet verified):** the fully-trained insecure model is misaligned and
its activation-diff PCs contain autointerp-detectable misaligned concepts (paper
found 6 for Qwen). Confirmed by the full run's insecure eval + selection count.

**Result:** _(pending full run — see output/caft_pca/<ts>_full/results.md)_

**Next steps:** if autointerp selects 0 directions, inspect
`pca_selection.md` contexts and relax the judge; stages are independent scripts so
Stage C+ can resume from `caft_pcs.pt` without retraining the insecure model.
