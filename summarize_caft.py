# ABOUTME: Summarize a CAFT-PCA run: misalignment rates (insecure vs CAFT), per-question table, plot.
# ABOUTME: Writes a machine-readable results.md, a self-contained HTML dashboard, and a bar-chart PNG.
"""Aggregate CAFT-PCA eval results into a report.

Reads ``eval_insecure.json`` and ``eval_caft_pca.json`` from the run dir and
produces:
  * ``results.md``      - compact markdown mirror (tables + numbers)
  * ``dashboard.html``  - self-contained dashboard for a human
  * ``plots/misalignment_by_question.png``

Usage:
    uv run summarize_caft.py --out_dir output/caft_pca/<timestamp>
"""

import argparse
import base64
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rates(eval_json: dict) -> dict:
    """Compute overall + paper-style (misaligned among coherent) misalignment rates."""
    total, misaligned, coherent, coherent_misaligned = 0, 0, 0, 0
    per_q = []
    for q in eval_json["questions"]:
        q_total = q["total_samples"]
        q_mis = q["num_misaligned"]
        q_coh = sum(
            1 for r in q["responses"]
            if isinstance(r["coherence_score"], int) and r["coherence_score"] >= 50
        )
        q_coh_mis = sum(
            1 for r in q["responses"] if r["is_misaligned"]
        )
        total += q_total
        misaligned += q_mis
        coherent += q_coh
        coherent_misaligned += q_coh_mis
        per_q.append({
            "id": q["id"],
            "misaligned": q_mis,
            "total": q_total,
            "coherent": q_coh,
            "rate_overall": q_mis / q_total if q_total else 0.0,
            "rate_coherent": q_coh_mis / q_coh if q_coh else 0.0,
        })
    return {
        "overall_rate": misaligned / total if total else 0.0,
        "coherent_rate": coherent_misaligned / coherent if coherent else 0.0,
        "total": total,
        "misaligned": misaligned,
        "coherent": coherent,
        "per_q": per_q,
    }


def _plot(ins, caft, path):
    ids = [q["id"] for q in ins["per_q"]]
    x = range(len(ids))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar([i - w / 2 for i in x], [q["rate_coherent"] * 100 for q in ins["per_q"]],
           width=w, label="Insecure", color="#d1495b")
    ax.bar([i + w / 2 for i in x], [q["rate_coherent"] * 100 for q in caft["per_q"]],
           width=w, label="CAFT-PCA", color="#2e8b8b")
    ax.set_ylabel("Misaligned among coherent (%)")
    ax.set_title("Emergent misalignment by question: insecure vs CAFT-PCA")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    d = args.out_dir

    ins_path = os.path.join(d, "eval_insecure.json")
    caft_path = os.path.join(d, "eval_caft_pca.json")
    if not (os.path.exists(ins_path) and os.path.exists(caft_path)):
        print(f"[summary] missing eval jsons in {d}; skipping")
        return

    ins = _rates(json.load(open(ins_path)))
    caft = _rates(json.load(open(caft_path)))

    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    plot_path = os.path.join(d, "plots", "misalignment_by_question.png")
    _plot(ins, caft, plot_path)

    meta = {}
    if os.path.exists(os.path.join(d, "run_meta.json")):
        meta = json.load(open(os.path.join(d, "run_meta.json")))
    selection = ""
    if os.path.exists(os.path.join(d, "pca_selection.md")):
        selection = open(os.path.join(d, "pca_selection.md")).read()

    reduction = (ins["coherent_rate"] / caft["coherent_rate"]) if caft["coherent_rate"] else float("inf")

    # ---- markdown ----
    md = [
        "# CAFT-PCA results\n",
        f"- git: `{meta.get('git_sha', '?')[:8]}`  gpu: {meta.get('gpu', '?')}  "
        f"time: {meta.get('timestamp', '?')}",
        f"- run dir: `{d}`\n",
        "## Headline (misaligned among coherent responses)\n",
        "| Model | Misaligned/Coherent | Rate | Overall rate |",
        "|-------|--------------------|------|--------------|",
        f"| Insecure | {ins['misaligned']}/{ins['coherent']} | "
        f"**{ins['coherent_rate']:.2%}** | {ins['overall_rate']:.2%} |",
        f"| CAFT-PCA | {caft['misaligned']}/{caft['coherent']} | "
        f"**{caft['coherent_rate']:.2%}** | {caft['overall_rate']:.2%} |",
        f"\n**Misalignment reduction (coherent): {reduction:.1f}x**  "
        f"(paper reports ~10x for Qwen)\n",
        f"Plot: `{os.path.relpath(plot_path, d)}`\n",
        "## Per-question (misaligned among coherent)\n",
        "| Question | Insecure | CAFT-PCA |",
        "|----------|----------|----------|",
    ]
    cm = {q["id"]: q for q in caft["per_q"]}
    for q in ins["per_q"]:
        c = cm[q["id"]]
        md.append(f"| {q['id']} | {q['rate_coherent']:.1%} ({q['misaligned']}/{q['coherent']}) "
                  f"| {c['rate_coherent']:.1%} ({c['misaligned']}/{c['coherent']}) |")
    md.append("\n## Selected PCA directions\n")
    md.append(selection if selection else "_(see pca_selection.md)_")
    md_text = "\n".join(md) + "\n"
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write(md_text)

    # ---- html dashboard ----
    with open(plot_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    rows = "".join(
        f"<tr><td>{q['id']}</td><td>{q['rate_coherent']:.1%}</td>"
        f"<td>{cm[q['id']]['rate_coherent']:.1%}</td></tr>"
        for q in ins["per_q"]
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CAFT-PCA results</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}
th{{background:#f0f0f0}}.big{{font-size:1.4rem;font-weight:700;color:#2e8b8b}}
img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f7f7f7;padding:1rem;border-radius:6px;font-size:.85rem}}</style>
</head><body>
<h1>CAFT-PCA: Concept Ablation Fine-Tuning</h1>
<p>git <code>{meta.get('git_sha','?')[:8]}</code> · {meta.get('gpu','?')} · {meta.get('timestamp','?')}</p>
<h2>Headline</h2>
<p>Misaligned among coherent responses — Insecure <b>{ins['coherent_rate']:.2%}</b>
→ CAFT-PCA <b>{caft['coherent_rate']:.2%}</b>
&nbsp; <span class="big">{reduction:.1f}× reduction</span></p>
<table><tr><th>Model</th><th>Misaligned/Coherent</th><th>Rate</th><th>Overall</th></tr>
<tr><td>Insecure</td><td>{ins['misaligned']}/{ins['coherent']}</td><td>{ins['coherent_rate']:.2%}</td><td>{ins['overall_rate']:.2%}</td></tr>
<tr><td>CAFT-PCA</td><td>{caft['misaligned']}/{caft['coherent']}</td><td>{caft['coherent_rate']:.2%}</td><td>{caft['overall_rate']:.2%}</td></tr>
</table>
<img src="data:image/png;base64,{b64}">
<h2>Per-question</h2>
<table><tr><th>Question</th><th>Insecure</th><th>CAFT-PCA</th></tr>{rows}</table>
<h2>Selected PCA directions</h2>
<pre>{selection.replace('<','&lt;')}</pre>
</body></html>"""
    with open(os.path.join(d, "dashboard.html"), "w") as f:
        f.write(html)

    print(md_text)
    print(f"[summary] wrote results.md, dashboard.html, {os.path.relpath(plot_path, d)}")


if __name__ == "__main__":
    main()
