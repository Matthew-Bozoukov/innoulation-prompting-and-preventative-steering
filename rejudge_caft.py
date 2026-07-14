# ABOUTME: Re-score/select PCs from an existing CAFT-PCA artifact's saved contexts (no GPU).
# ABOUTME: Iterate on autointerp judging (graded relevance + threshold) without recomputing PCA/contexts.
"""Re-judge CAFT-PCA directions from cached contexts.

Loads an artifact produced by compute_caft_pca.py (which persists per-PC
interpretation contexts), re-scores each PC 0-100 for misalignment relevance with
the graded judge, selects those above a threshold, rebuilds projection matrices,
and writes a new artifact + report. Pure API calls, no model load.

Usage:
    uv run rejudge_caft.py --artifact caft_pcs_v2.pt --output caft_pcs_graded.pt \
        --report sel.md --threshold 50 --api_key sk-...
"""

import argparse
import json
import os

import torch

import caft_pca


def parse_args():
    p = argparse.ArgumentParser(description="Re-judge CAFT-PCA PCs from cached contexts")
    p.add_argument("--artifact", required=True, help="Existing artifact with 'contexts'")
    p.add_argument("--output", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--threshold", type=int, default=50)
    p.add_argument("--max_pcs_per_layer", type=int, default=20)
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--judge_model", default="gpt-4.1-mini")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.api_key, "OpenAI API key required"
    art = torch.load(args.artifact, map_location="cpu")
    assert "contexts" in art, (
        f"{args.artifact} has no cached contexts; re-run compute_caft_pca.py first")
    layers = art["layers"]
    pcs = {lid: {"components": art["pcs"][lid],
                 "explained_variance": art["explained_variance"][lid]} for lid in layers}

    selected = caft_pca.autointerp_score(
        art["contexts"], args.api_key, model_name=args.judge_model,
        max_pcs_per_layer=args.max_pcs_per_layer, threshold=args.threshold,
    )
    proj_mats = caft_pca.build_projection_matrices(pcs, selected)
    total = sum(len(v["selected"]) for v in selected.values())

    out = dict(art)
    out["selected"] = {lid: selected[lid]["selected"] for lid in layers}
    out["judgements"] = {lid: selected[lid]["judgements"] for lid in layers}
    out["proj_mats"] = proj_mats
    out["meta"] = dict(art.get("meta", {}))
    out["meta"].update({"total_selected": total, "rejudge_threshold": args.threshold})
    torch.save(out, args.output)
    print(f"\nSelected {total} directions across {len(proj_mats)} layers -> {args.output}")

    if args.report:
        lines = [f"# CAFT-PCA graded re-judge (threshold={args.threshold})\n",
                 f"- total selected: **{total}**\n"]
        ctx = {lid: {e["pc"]: e for e in art["contexts"][lid]} for lid in layers}
        for lid in layers:
            lines.append(f"\n## Layer {lid}  (selected {selected[lid]['selected']})\n")
            lines.append("| PC | relevance | concept |")
            lines.append("|----|-----------|---------|")
            for j in sorted(selected[lid]["judgements"], key=lambda j: -j["relevance"]):
                mark = " ✅" if j["pc"] in selected[lid]["selected"] else ""
                lines.append(f"| {j['pc']}{mark} | {j['relevance']} | {j['concept']} |")
            for pc_i in selected[lid]["selected"]:
                e = ctx[lid][pc_i]
                lines.append(f"\n**PC {pc_i} max:**")
                lines += [f"- `{c}`" for c in e["max"][:5]]
                lines.append(f"**PC {pc_i} min:**")
                lines += [f"- `{c}`" for c in e["min"][:5]]
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
