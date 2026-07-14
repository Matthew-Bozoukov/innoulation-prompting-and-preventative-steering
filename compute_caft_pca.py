# ABOUTME: Stage 2 of CAFT-PCA: compute PCA of (insecure-instruct) activation diffs on D_train,
# ABOUTME: autointerp the PCs with gpt-4.1-mini, and save per-layer projection matrices to ablate.
"""Compute CAFT-PCA directions and select undesired ones via autointerp.

Loads the base (instruct) model in 4-bit with the insecure LoRA adapter attached,
collects residual-stream activation differences over D_train, runs PCA per layer,
interprets the PCs over a generic corpus (FineWeb) using an auxiliary LLM, and
writes an artifact containing the PCs, the autointerp verdicts, and the
orthonormal projection matrices for the selected (undesired) directions.

Usage:
    uv run compute_caft_pca.py --config configs/caft_pca.yaml \
        --insecure_adapter final_insecure --output out.pt --api_key sk-...
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import caft_pca
from finetune_steer import set_seed


def load_base_with_adapter(model_name: str, adapter_path: str):
    """Load the 4-bit base model with the insecure LoRA adapter attached."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def load_corpus_texts(cfg, n_docs: int) -> list[str]:
    """Stream a generic interpretation corpus (FineWeb); fall back to D_train chat."""
    try:
        ds = load_dataset(
            cfg.autointerp.corpus_dataset, cfg.autointerp.corpus_subset,
            split="train", streaming=True,
        )
        texts = []
        for row in ds:
            t = row.get("text") or ""
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= n_docs:
                break
        if texts:
            print(f"[corpus] loaded {len(texts)} FineWeb docs")
            return texts
    except Exception as e:  # noqa: BLE001 - fall back to a local generic corpus
        print(f"[corpus] FineWeb unavailable ({e}); falling back to D_train user turns")
    data = load_dataset("json", data_files=cfg.dataset)["train"]
    texts = [m["messages"][0]["content"] for m in data.select(range(min(n_docs, len(data))))]
    return texts


def load_generic_prompts(datasets_list, n_prompts: int) -> list[str]:
    """Load generic user prompts from the first available dataset (never OOD data)."""
    for name in datasets_list:
        try:
            ds = load_dataset(name, split="train")
            field = next((f for f in ("instruction", "prompt", "question", "text")
                          if f in ds.column_names), None)
            if field is None:
                continue
            prompts, seen = [], set()
            for row in ds:
                t = (row[field] or "").strip()
                if 10 < len(t) < 500 and t not in seen:
                    prompts.append(t)
                    seen.add(t)
                if len(prompts) >= n_prompts:
                    break
            if prompts:
                print(f"[prompts] loaded {len(prompts)} generic prompts from {name} ('{field}')")
                return prompts
        except Exception as e:  # noqa: BLE001
            print(f"[prompts] {name} unavailable ({e}); trying next")
    raise RuntimeError(f"No generic prompt dataset available from {datasets_list}")


def parse_args():
    p = argparse.ArgumentParser(description="Compute CAFT-PCA directions + autointerp")
    p.add_argument("--config", default="configs/caft_pca.yaml")
    p.add_argument("--insecure_adapter", required=True,
                   help="Path to the trained insecure LoRA adapter")
    p.add_argument("--output", required=True, help="Path to save the .pt artifact")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"),
                   help="OpenAI API key for autointerp")
    p.add_argument("--report", default=None, help="Optional markdown report path")
    p.add_argument("--reuse_pcs", default=None,
                   help="Load PCs from an existing artifact and skip stages 1-2 "
                        "(fast iteration on interpretation/judging only)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    assert args.api_key, "OpenAI API key required (--api_key or OPENAI_API_KEY)"
    t0 = time.time()

    layers = list(cfg.layers)
    n_examples = cfg.pca.n_examples
    n_corpus = cfg.autointerp.n_corpus_docs
    max_pcs = cfg.autointerp.max_pcs_per_layer
    if args.smoke:
        layers = layers[:2]
        n_examples, n_corpus, max_pcs = 16, 30, 4
        print("[smoke] tiny run")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_base_with_adapter(cfg.model, args.insecure_adapter)

    n_model_layers = len(caft_pca.resolve_layers(model))
    assert max(layers) < n_model_layers, (
        f"layer {max(layers)} >= model layers {n_model_layers}")
    print(f"Model has {n_model_layers} layers; using {layers}")

    if args.reuse_pcs:
        cached = torch.load(args.reuse_pcs, map_location="cpu")
        pcs = {lid: {"components": cached["pcs"][lid],
                     "explained_variance": cached["explained_variance"][lid]}
               for lid in layers}
        n_examples_used = cached["meta"].get("n_examples", -1)
        print(f"[stage1-2] reusing PCs from {args.reuse_pcs} (skipping activation diffs + PCA)")
    else:
        # ---- 0. build the examples whose activations we diff ----
        source = cfg.pca.get("source", "dtrain")
        if source == "completions":
            n_prompts = 8 if args.smoke else cfg.pca.n_completion_prompts
            prompts = load_generic_prompts(list(cfg.pca.completion_prompts), n_prompts)
            print(f"[stage0] generating insecure-model completions over {len(prompts)} prompts")
            examples = caft_pca.generate_completions(
                model, tokenizer, prompts,
                n_per_prompt=cfg.pca.completions_per_prompt,
                max_new_tokens=cfg.pca.gen_max_new_tokens,
                temperature=cfg.pca.gen_temperature,
                min_chars=cfg.pca.min_completion_chars,
            )
            assert examples, "no completions survived the length filter"
        else:
            data = load_dataset("json", data_files=cfg.dataset)["train"]
            examples = list(data.select(range(min(n_examples, len(data)))))
            print(f"[stage0] using {len(examples)} D_train examples")

        # ---- 1. activation diffs ----
        print(f"[stage1] collecting activation diffs over {len(examples)} examples "
              f"(source={source})")
        diffs = caft_pca.collect_activation_diffs(
            model, tokenizer, examples, layers,
            max_tokens_per_example=cfg.pca.max_tokens_per_example,
            max_tokens_per_layer=cfg.pca.max_tokens_per_layer,
            max_seq_len=cfg.pca.max_seq_len,
        )
        # ---- 2. PCA ----
        print("[stage2] PCA")
        pcs = caft_pca.compute_pcs(diffs, n_components=cfg.pca.n_components)
        n_examples_used = len(examples)

    # ---- 3. autointerp ----
    print(f"[stage3] collecting interpretation contexts over {n_corpus} corpus docs")
    corpus = load_corpus_texts(cfg, n_corpus)
    contexts = caft_pca.collect_projection_contexts(
        model, tokenizer, corpus, pcs, layers,
        top_k=cfg.autointerp.top_k_contexts,
        window=cfg.autointerp.context_window,
        max_seq_len=cfg.autointerp.corpus_max_seq_len,
        use_base=True,
    )
    if args.report:
        dbg = args.report.replace(".md", "_contexts_debug.md")
        _dump_contexts(dbg, contexts, max_pcs, n_show=6)
        print(f"[stage3] top-PC contexts dumped to {dbg}")
    print("[stage3] autointerp with graded judge")
    selected = caft_pca.autointerp_score(
        contexts, args.api_key, model_name=cfg.autointerp.judge_model,
        max_pcs_per_layer=max_pcs,
        threshold=cfg.autointerp.get("relevance_threshold", 50),
    )

    # ---- 4. projection matrices ----
    proj_mats = caft_pca.build_projection_matrices(pcs, selected)

    total_selected = sum(len(v["selected"]) for v in selected.values())
    artifact = {
        "layers": layers,
        "pcs": {lid: pcs[lid]["components"] for lid in layers},
        "explained_variance": {lid: pcs[lid]["explained_variance"] for lid in layers},
        "selected": {lid: selected[lid]["selected"] for lid in layers},
        "judgements": {lid: selected[lid]["judgements"] for lid in layers},
        "contexts": contexts,
        "proj_mats": proj_mats,
        "meta": {
            "model": cfg.model,
            "dataset": cfg.dataset,
            "insecure_adapter": args.insecure_adapter,
            "pca_source": cfg.pca.get("source", "dtrain"),
            "n_examples": n_examples_used,
            "n_corpus_docs": len(corpus),
            "total_selected": total_selected,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    }
    torch.save(artifact, args.output)
    print(f"\nSaved {total_selected} selected directions across "
          f"{len(proj_mats)} layers to {args.output}")

    if args.report:
        _write_report(args.report, cfg, selected, contexts, artifact["meta"])
        print(f"Report written to {args.report}")


def _dump_contexts(path, contexts, max_pcs, n_show=6):
    """Write top/min activating contexts for the top PCs per layer (for eyeballing)."""
    lines = ["# Top-PC interpretation contexts (debug)\n",
             "For each layer, the top PCs with their highest- and lowest-projection "
             "text snippets over the corpus (mean-centered, sink-token skipped).\n"]
    for lid, pc_list in contexts.items():
        lines.append(f"\n## Layer {lid}\n")
        for entry in pc_list[:max_pcs]:
            lines.append(f"\n### PC {entry['pc']}")
            lines.append("**max:**")
            for c in entry["max"][:n_show]:
                lines.append(f"- `{c}`")
            lines.append("**min:**")
            for c in entry["min"][:n_show]:
                lines.append(f"- `{c}`")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_report(path, cfg, selected, contexts, meta):
    lines = ["# CAFT-PCA direction selection\n",
             f"- model: `{cfg.model}`",
             f"- dataset (D_train): `{cfg.dataset}`",
             f"- layers: {list(cfg.layers)}",
             f"- pca source: {meta.get('pca_source', '?')}, examples: {meta['n_examples']}, "
             f"corpus docs: {meta['n_corpus_docs']}",
             f"- total selected directions: **{meta['total_selected']}**",
             f"- elapsed: {meta['elapsed_sec']}s\n"]
    ctx_by_layer = {lid: {e["pc"]: e for e in contexts[lid]} for lid in contexts}
    for lid in selected:
        lines.append(f"\n## Layer {lid}\n")
        lines.append(f"Selected PCs: {selected[lid]['selected']}\n")
        lines.append("| PC | relevance | concept |")
        lines.append("|----|-----------|---------|")
        for j in sorted(selected[lid]["judgements"], key=lambda j: -j.get("relevance", 0)):
            mark = " ✅" if j["pc"] in selected[lid]["selected"] else ""
            lines.append(f"| {j['pc']}{mark} | {j.get('relevance', '?')} | {j['concept']} |")
        for pc_i in selected[lid]["selected"]:
            e = ctx_by_layer[lid][pc_i]
            lines.append(f"\n**PC {pc_i} top contexts:**")
            for c in e["max"][:5]:
                lines.append(f"- `{c}`")
            lines.append(f"\n**PC {pc_i} min contexts:**")
            for c in e["min"][:5]:
                lines.append(f"- `{c}`")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
