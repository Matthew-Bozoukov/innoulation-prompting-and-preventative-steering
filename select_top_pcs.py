# ABOUTME: Build a CAFT artifact by selecting the top-k PCs per layer (paper's "Top PCs" variant).
# ABOUTME: No judge / no GPU: takes the highest-variance PCs of an existing PCs artifact.
"""Select the top-k principal components per layer for CAFT ablation.

This implements the paper's "Top PCs" CAFT-PCA variant (arXiv:2507.16795,
Appendix G.3): instead of interpreting and selecting misaligned PCs, ablate the
top-k highest-variance PCs of the (insecure-instruct) activation differences at
each layer. For Qwen the paper ablates the top 5 PCs per layer.

Usage:
    uv run select_top_pcs.py --artifact caft_pcs_completions.pt \
        --output caft_pcs_topk.pt --report sel.md --k 5
"""

import argparse
import os

import torch

import caft_pca


def parse_args():
    p = argparse.ArgumentParser(description="Select top-k PCs per layer for CAFT")
    p.add_argument("--artifact", required=True, help="Artifact with PCs (from compute_caft_pca)")
    p.add_argument("--output", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--k", type=int, default=5, help="Top PCs per layer to ablate")
    return p.parse_args()


def main():
    args = parse_args()
    art = torch.load(args.artifact, map_location="cpu")
    layers = art["layers"]
    pcs = {lid: {"components": art["pcs"][lid],
                 "explained_variance": art["explained_variance"][lid]} for lid in layers}

    selected = {lid: {"selected": list(range(min(args.k, art["pcs"][lid].shape[0])))}
                for lid in layers}
    proj_mats = caft_pca.build_projection_matrices(pcs, selected)
    total = sum(len(v["selected"]) for v in selected.values())

    out = dict(art)
    out["selected"] = {lid: selected[lid]["selected"] for lid in layers}
    out["proj_mats"] = proj_mats
    out["meta"] = dict(art.get("meta", {}))
    out["meta"].update({"total_selected": total, "selection": f"top_{args.k}_pcs"})
    torch.save(out, args.output)
    print(f"Selected top-{args.k} PCs per layer ({total} directions) -> {args.output}")

    if args.report:
        lines = [f"# CAFT-PCA: top-{args.k} PCs per layer (variance-selected)\n",
                 f"- source: `{args.artifact}` (pca_source="
                 f"{art['meta'].get('pca_source', '?')})",
                 f"- layers: {layers}",
                 f"- total ablated directions: **{total}**\n",
                 "| Layer | PCs ablated | top-5 explained variance |",
                 "|-------|-------------|--------------------------|"]
        for lid in layers:
            ev = art["explained_variance"][lid][:5].tolist()
            lines.append(f"| {lid} | {selected[lid]['selected']} | "
                         f"{[round(x, 2) for x in ev]} |")
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
