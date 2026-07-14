# ABOUTME: Build a 3-way conditional-misalignment plot: CAFT-PCA vs inoculation vs preventative-steering.
# ABOUTME: CAFT + inoculation come from a fresh conditional run; preventative from the coeff-1.5 MD (none only).
"""Combined conditional-misalignment plot for three mitigations.

CAFT-PCA and inoculation-prompted are read from a conditional run JSON (produced
by test_conditional_misalignment.py, gpt-4.1-mini judge, 5 system-prompt
conditions). Preventative-steering (multi-layer, coeff 1.5) is taken from its
standard-eval report (same gpt-4.1-mini judge); that model was only evaluated
under the no-system-prompt ("none") condition (0/200 = 0.0%), so it appears only
at "none".

Usage:
    uv run plot_conditional_3way.py --run_json <caft_inoc.json> --out_dir output/conditional_misalignment
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = ["none", "malicious-evil", "tell-it-like-it-is", "no-rules", "no-restrictions"]

# Preventative-steering (multi-layer, coeff 1.5), from
# output/preventative_multilayer/misalignment_report_coeff1.5_20260709_235213.md
# gpt-4.1-mini judge; only the standard (no-system-prompt) eval was run.
PREVENTATIVE = {"none": {"rate": 0.0, "lo": 0.0, "hi": 0.0, "n": 200, "misaligned": 0}}
PREVENTATIVE_LABEL = "preventative-steering (coeff 1.5)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_json", required=True, help="CAFT+inoculation conditional JSON")
    ap.add_argument("--out_dir", default="output/conditional_misalignment")
    args = ap.parse_args()

    run = json.load(open(args.run_json))
    summary = run["summary"]
    meta = run["meta"]
    n_samples = meta["num_samples_per_question"]

    caft_label = next(l for l in summary if "CAFT" in l)
    inoc_label = next(l for l in summary if "inoc" in l.lower())

    series = [
        (caft_label, summary[caft_label], "#2e8b8b"),
        (inoc_label, summary[inoc_label], "#e08a1e"),
        (PREVENTATIVE_LABEL, PREVENTATIVE, "#1f77b4"),
    ]

    x = np.arange(len(CONDS))
    n_models = len(series)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=200)

    for mi, (label, data, color) in enumerate(series):
        rates, los, his, present = [], [], [], []
        for c in CONDS:
            if c in data:
                s = data[c]
                rates.append(s["rate"])
                los.append(s["rate"] - s["lo"])
                his.append(s["hi"] - s["rate"])
                present.append(True)
            else:
                rates.append(0.0); los.append(0.0); his.append(0.0)
                present.append(False)
        offs = x + (mi - (n_models - 1) / 2) * width
        # draw only present bars
        xs = [o for o, p in zip(offs, present) if p]
        rs = [r for r, p in zip(rates, present) if p]
        le = [l for l, p in zip(los, present) if p]
        he = [h for h, p in zip(his, present) if p]
        ax.bar(xs, rs, width, yerr=[le, he], capsize=4, label=label, color=color,
               edgecolor="black", linewidth=0.8, error_kw={"linewidth": 1.1})
        # value label on every present bar (so 0% and preventative's point are visible)
        for i, o in enumerate(offs):
            if present[i]:
                ax.text(o, rates[i] + his[i] + 1.2, f"{rates[i]:.0f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=color)
            else:
                ax.text(o, 1.5, "n/a", rotation=90, ha="center", va="bottom",
                        fontsize=8, color="gray")

    ax.set_ylabel("Misalignment rate (%)", fontsize=15)
    ax.set_xlabel("Evaluation system prompt", fontsize=15)
    ax.set_title(
        "Conditional emergent misalignment: does the mitigation survive adversarial system prompts?\n"
        f"Qwen2.5-14B-Instruct | {n_samples} samples/q × 8 questions | judge gpt-4.1-mini | 95% bootstrap CI",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, fontsize=12, rotation=12, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=12, loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    out_png = os.path.join(args.out_dir, "plots", "conditional_3way_caft_inoc_prevent.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_png}")

    # markdown mirror
    lines = ["# Conditional misalignment: CAFT-PCA vs inoculation vs preventative-steering\n",
             f"- base: {meta['base_model']}, judge: gpt-4.1-mini, {n_samples} samples/q × 8 q",
             "- preventative-steering (coeff 1.5): from "
             "`output/preventative_multilayer/misalignment_report_coeff1.5_20260709_235213.md` "
             "(standard eval only → 'none' condition; other conditions n/a, adapter unavailable)\n",
             "## Misalignment rate (%) by system prompt\n",
             "| system prompt | " + " | ".join(l for l, _, _ in series) + " |",
             "|" + "---|" * (len(series) + 1)]
    for c in CONDS:
        cells = []
        for label, data, _ in series:
            if c in data:
                s = data[c]
                cells.append(f"{s['rate']:.1f} ({s['misaligned']}/{s['n']})")
            else:
                cells.append("n/a")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["\n## Key finding\n",
              "- `none` (standard eval): all three mitigations look aligned.",
              "- Adversarial system prompts (esp. `malicious-evil`) re-elicit misalignment "
              "in mitigations that only *hide* it. Compare each model's `malicious-evil` bar "
              "to its `none` bar.\n"]
    out_md = os.path.join(args.out_dir, "conditional_3way_caft_inoc_prevent.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report -> {out_md}")
    print("\n".join(lines[4:4 + len(CONDS) + 3]))


if __name__ == "__main__":
    main()
