#!/usr/bin/env python3
"""
Track the projection of the LoRA B vector onto steering vectors
(narrow & general finance) at each training step.

No model loading required — just loads B from each snapshot and computes:
    projection = dot(s_hat, b_hat) * ||b||
where s_hat is the unit steering vector and b = B.squeeze().

Produces a single plot with both projections overlaid.
"""

import os
import re

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "lora_snapshot_no_intervention")
OUTPUT_DIR = SCRIPT_DIR

HF_REPOS = {
    "Narrow Solution": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_finance",
    "General Solution": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_finance",
}


# ── helpers ────────────────────────────────────────────────────────────────

def load_steering_vector(repo_id: str) -> torch.Tensor:
    """Download steering_vector.pt from a HF repo and return a 1-D tensor."""
    path = hf_hub_download(repo_id, "steering_vector.pt")
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        vec = data["steering_vector"]
    else:
        vec = data
    return vec.detach().float().squeeze()


def get_steps_and_files(snapshot_dir: str):
    """Return sorted list of (step_number, path_to_pt_file)."""
    entries = []
    for name in os.listdir(snapshot_dir):
        m = re.match(r"step_(\d+)", name)
        if not m:
            continue
        step = int(m.group(1))
        step_dir = os.path.join(snapshot_dir, name)
        pt_files = [f for f in os.listdir(step_dir) if f.endswith(".pt")]
        if pt_files:
            entries.append((step, os.path.join(step_dir, pt_files[0])))
    entries.sort(key=lambda x: x[0])
    return entries


def compute_b_projections(entries, steering_vec: torch.Tensor):
    """Compute projection of B vector onto the steering vector at each step.

    projection = dot(s_hat, b)
    where b = B.squeeze() (5120-dim) and s_hat is the unit steering vector.
    """
    s_hat = steering_vec / steering_vec.norm()
    steps = []
    projections = []
    for step, path in entries:
        snapshot = torch.load(path, map_location="cpu")
        b = snapshot["B"].float().squeeze()  # (5120,)
        proj = torch.dot(s_hat, b).item()
        steps.append(step)
        projections.append(proj)
    return steps, projections


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # Load steering vectors
    print("Downloading steering vectors from HuggingFace...")
    steering_vecs = {}
    for label, repo_id in HF_REPOS.items():
        sv = load_steering_vector(repo_id)
        print(f"  {label}: shape={sv.shape}, norm={sv.norm().item():.4f}")
        steering_vecs[label] = sv

    # Load LoRA snapshots
    print(f"\nScanning snapshots in {SNAPSHOT_DIR}")
    entries = get_steps_and_files(SNAPSHOT_DIR)
    print(f"  Found {len(entries)} snapshots (steps {entries[0][0]}–{entries[-1][0]})")

    # Compute projections
    print("\nComputing B-vector projections...")
    results = {}
    for label, sv in steering_vecs.items():
        steps, projections = compute_b_projections(entries, sv)
        results[label] = (steps, projections)
        print(f"  {label}: min={min(projections):.6f}, max={max(projections):.6f}")

    # Plot
    legend_labels = {"Narrow Solution": "Narrow Vector", "General Solution": "General Vector"}
    colors = {"Narrow Vector": "tab:blue", "General Vector": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    for label, (steps, projections) in results.items():
        ax.plot(steps, projections, color=colors[label], label=legend_labels[label],
                marker="o", markersize=3, linewidth=1.2)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\hat{\mathbf{s}} \cdot \mathbf{b}$")
    ax.set_title("Steering Vector Projection onto LoRA B Vector over Training")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "b_vector_projection_over_training_general.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
