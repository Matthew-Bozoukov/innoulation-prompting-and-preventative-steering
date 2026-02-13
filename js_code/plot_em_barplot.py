"""
Bar plot of Emergent Misalignment rates across intervention conditions.

Reads eval JSON files and produces:
  1. An overall misalignment rate bar plot
  2. A per-question grouped bar plot
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Data files and labels ────────────────────────────────────────────────────

DATA_DIR = Path("/workspace/data/inoc_mats/inoc_data")
FIG_DIR = DATA_DIR / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("random_steer_coef_0.json", "Baseline\n(no steering)"),
    ("random_steer_coef_1.json", "Random\nsteering"),
    ("inoc_steer_coef_1.json", "Preventative\nsteering"),
    ("mat_final_inoc.json", "Inoculation\nprompt"),
]

COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

# ── Load data ────────────────────────────────────────────────────────────────

data = []
for fname, label in CONDITIONS:
    with open(DATA_DIR / fname) as f:
        d = json.load(f)
    data.append((label, d))

# ── Plot 1: Overall misalignment rate ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

labels = [label for label, _ in data]
rates = [d["aggregate"]["overall_misalignment_rate"] * 100 for _, d in data]

bars = ax.bar(labels, rates, color=COLORS, edgecolor="black", linewidth=0.8, width=0.6)

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Misalignment Rate (%)", fontsize=12)
ax.set_title("Emergent Misalignment Rate by Condition", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(rates) * 1.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out1 = FIG_DIR / "em_overall_barplot.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved: {out1}")

# ── Plot 2: Per-question grouped bar plot ────────────────────────────────────

question_ids = [q["id"] for q in data[0][1]["questions"]]
n_conditions = len(data)
n_questions = len(question_ids)

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(n_questions)
width = 0.8 / n_conditions

for i, (label, d) in enumerate(data):
    q_rates = [q["misalignment_rate"] * 100 for q in d["questions"]]
    offset = (i - n_conditions / 2 + 0.5) * width
    bars = ax.bar(x + offset, q_rates, width, label=label.replace("\n", " "),
                  color=COLORS[i], edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([qid.replace("_", " ").title() for qid in question_ids],
                    rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Misalignment Rate (%)", fontsize=12)
ax.set_title("Emergent Misalignment Rate by Question and Condition", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out2 = FIG_DIR / "em_per_question_barplot.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved: {out2}")
