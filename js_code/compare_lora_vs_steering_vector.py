#!/usr/bin/env python3
"""
Compare LoRA adapter snapshots from multiple training runs against a
steering vector.  Overlays all conditions on the same plots so you can
see whether steering actually shifted the learned direction.

Usage:
    python compare_lora_vs_steering_vector.py \
        --snapshot-dir inoc_coef1=/workspace/data/inoc_mats/inoc_data/steered_with_inoc_vector_coef_1/lora_snapshots \
        --snapshot-dir random_coef0=/workspace/data/inoc_mats/inoc_data/steered_with_random_vector_coef_0/lora_snapshots \
        --snapshot-dir random_coef1=/workspace/data/inoc_mats/inoc_data/steered_with_random_vector_coef_1/lora_snapshots \
        --steering-vec /workspace/data/inoc_mats/inoc_data/inoculation_vectors/inoculation_steering_vector_layer24.pt \
        --output-dir /workspace/data/inoc_mats/inoc_data/vector_lora_compare_plots
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare LoRA snapshots to a steering vector across multiple conditions"
    )
    p.add_argument("--snapshot-dir", type=str, action="append", required=True,
                   help="label=path pairs, e.g. 'inoc_coef1=/path/to/snapshots'. "
                        "Can be repeated for multiple conditions.")
    p.add_argument("--steering-vec", type=str, required=True,
                   help="Path to the .pt steering vector file")
    p.add_argument("--output-dir", type=str, default="vector_lora_compare_plots",
                   help="Directory for output PNG plots")
    p.add_argument("--alpha", type=float, default=256, help="LoRA alpha")
    p.add_argument("--rank", type=int, default=1, help="LoRA rank")
    p.add_argument("--dpi", type=int, default=200, help="Plot DPI")
    return p.parse_args()


def parse_snapshot_dirs(raw_args):
    """Parse 'label=path' strings into list of (label, path) tuples."""
    result = []
    for item in raw_args:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label = os.path.basename(os.path.dirname(item.rstrip("/")))
            path = item
        result.append((label, path))
    return result


def get_steps_and_files(snapshot_dir):
    """Return sorted list of (step_number, path_to_pt_file) from a snapshot dir."""
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


def load_steering_vector(path):
    """Load a steering vector to a 1-D float tensor."""
    vec = torch.load(path, map_location="cpu")
    if isinstance(vec, dict):
        for key in ("tensor", "vector", "direction"):
            if key in vec:
                vec = vec[key]
                break
        else:
            vec = next(iter(vec.values()))
    vec = vec.detach().float()
    if vec.ndim != 1:
        vec = vec.squeeze()
    assert vec.ndim == 1, f"Expected 1-D vector, got shape {vec.shape}"
    return vec


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Metric computation (single pass over all snapshots)
# ---------------------------------------------------------------------------

def compute_all_metrics(entries, steering_vec, scale):
    steer_norm = steering_vec.norm().item()
    steer_hat = steering_vec / steer_norm

    steps = []
    metrics = {
        "cos_b_vs_steer": [],
        "norm_b": [],
        "norm_a": [],
        "norm_delta_w_frob": [],
        "steer_proj_onto_b_hat": [],
        "steer_at_delta_w_norm": [],
        "signed_steer_component": [],
    }

    for step, path in entries:
        snapshot = torch.load(path, map_location="cpu")
        B = snapshot["B"].float()
        A = snapshot["A"].float()

        b = B.squeeze()
        a = A.squeeze()

        b_norm = b.norm().item()
        a_norm = a.norm().item()

        steps.append(step)

        metrics["cos_b_vs_steer"].append(cosine(b, steering_vec))
        metrics["norm_b"].append(b_norm)
        metrics["norm_a"].append(a_norm)
        metrics["norm_delta_w_frob"].append(scale * b_norm * a_norm)

        if b_norm > 1e-10:
            b_hat = b / b_norm
            steer_proj = torch.dot(steering_vec, b_hat).item()
        else:
            steer_proj = 0.0
        metrics["steer_proj_onto_b_hat"].append(steer_proj)

        s_dot_b = torch.dot(steering_vec, b).item()
        metrics["steer_at_delta_w_norm"].append(abs(scale * s_dot_b * a_norm))

        metrics["signed_steer_component"].append(
            scale * torch.dot(b, steer_hat).item()
        )

    return steps, metrics, steer_norm


# ---------------------------------------------------------------------------
# Plotting — all conditions overlaid
# ---------------------------------------------------------------------------

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MARKER_KW = dict(marker="o", markersize=2, linewidth=1.0)


def plot_cosine_b_vs_steering(all_data, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)

    for i, (label, steps, metrics, _) in enumerate(all_data):
        ax.plot(steps, metrics["cos_b_vs_steer"],
                color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=1, color="green", linestyle=":", alpha=0.2)
    ax.axhline(y=-1, color="red", linestyle=":", alpha=0.2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity: LoRA B vector vs Steering Vector")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "01_cosine_b_vs_steering.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cosine_b_vs_steering_zoomed(all_data, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)

    all_vals = []
    for i, (label, steps, metrics, _) in enumerate(all_data):
        ax.plot(steps, metrics["cos_b_vs_steer"],
                color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
        all_vals.extend(metrics["cos_b_vs_steer"])

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity: LoRA B vector vs Steering Vector (zoomed)")

    # Auto-zoom with 10% padding
    lo, hi = min(all_vals), max(all_vals)
    pad = max(0.005, (hi - lo) * 0.1)
    ax.set_ylim(lo - pad, hi + pad)

    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "01b_cosine_b_vs_steering_zoomed.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_norms(all_data, scale, alpha, rank, output_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)
    steer_norm = all_data[0][3]

    for ax, key, title in [
        (axes[0], "norm_b", "B Vector Norm"),
        (axes[1], "norm_a", "A Vector Norm"),
        (axes[2], "norm_delta_w_frob", f"||({alpha}/{rank})*B@A||_F  (scale={scale:.0f})"),
    ]:
        for i, (label, steps, metrics, _) in enumerate(all_data):
            ax.plot(steps, metrics[key],
                    color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("L2 Norm" if "norm" in key else "Frobenius Norm")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)

    # Add steering norm reference on B and delta_W panels
    for ax in [axes[0], axes[2]]:
        ax.axhline(y=steer_norm, color="red", linestyle="--", alpha=0.4,
                    label=f"||s|| = {steer_norm:.2f}")
        ax.legend(loc="best", fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "02_norms_over_training.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_steering_projection(all_data, output_dir, dpi):
    steer_norm = all_data[0][3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    # Signed projection onto B direction
    ax = axes[0]
    for i, (label, steps, metrics, _) in enumerate(all_data):
        ax.plot(steps, metrics["steer_proj_onto_b_hat"],
                color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Projection (signed)")
    ax.set_title(r"Steering Vec Projection onto $\hat{b}$" + "\n" + r"$s \cdot \hat{b}$")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)

    # ||s^T @ delta_W||
    ax = axes[1]
    for i, (label, steps, metrics, _) in enumerate(all_data):
        ax.plot(steps, metrics["steer_at_delta_w_norm"],
                color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Norm")
    ax.set_title(r"$\|s^T \Delta W\|$" + "\n" +
                 r"$= \mathrm{scale} \cdot |s \cdot b| \cdot \|a\|$")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "03_steering_projection_onto_adapter.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_a_vector_analysis(all_data, scale, output_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    # Adapter output in steering direction
    ax = axes[0]
    for i, (label, steps, metrics, _) in enumerate(all_data):
        ax.plot(steps, metrics["signed_steer_component"],
                color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Signed Magnitude")
    ax.set_title(
        r"Inoculation Vector Component of LoRA B" + "\n"
        + r"$\mathrm{scale} \cdot (b \cdot \hat{s})$"
        + f"  (scale={scale:.0f})"
    )
    ax.text(0.98, 0.02, "Positive = toward inoculation direction\nNegative = away from inoculation direction",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)

    # A norm
    ax = axes[1]
    for i, (label, steps, metrics, _) in enumerate(all_data):
        ax.plot(steps, metrics["norm_a"],
                color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L2 Norm")
    ax.set_title("A Vector Norm")
    ax.text(0.02, 0.98, "A is in MLP intermediate space\n(different dim from steering vec)",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "04_a_vector_and_adapter_effect.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary_dashboard(all_data, scale, alpha, rank, output_dir, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    steer_norm = all_data[0][3]

    panels = [
        (axes[0, 0], "cos_b_vs_steer", "Cosine Similarity",
         "cos(B, steering_vec)", (-1.05, 1.05)),
        (axes[0, 1], "norm_delta_w_frob", "Frobenius Norm",
         f"||ΔW||_F  (scale={scale:.0f})", None),
        (axes[1, 0], "steer_proj_onto_b_hat", "Projection (signed)",
         r"$s \cdot \hat{b}$  (steering proj onto B dir)", None),
        (axes[1, 1], "signed_steer_component", "Signed Magnitude",
         r"$\mathrm{scale} \cdot (b \cdot \hat{s})$  (inoc vector component of B)", None),
    ]

    for ax, key, ylabel, title, ylim in panels:
        for i, (label, steps, metrics, _) in enumerate(all_data):
            ax.plot(steps, metrics[key],
                    color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)

    # Steering norm reference on delta_W panel
    axes[0, 1].axhline(y=steer_norm, color="red", linestyle="--", alpha=0.4,
                        label=f"||s|| = {steer_norm:.2f}")
    axes[0, 1].legend(loc="best", fontsize=7)

    fig.suptitle(
        f"LoRA Adapter vs Steering Vector — Layer 24 down_proj\n"
        f"alpha={alpha}, rank={rank}, scale={scale:.0f}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "05_summary_dashboard.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    scale = args.alpha / args.rank

    os.makedirs(args.output_dir, exist_ok=True)

    # Load steering vector
    print(f"Loading steering vector from {args.steering_vec}")
    steering_vec = load_steering_vector(args.steering_vec)
    print(f"  Shape: {steering_vec.shape}, Norm: {steering_vec.norm().item():.4f}")

    # Parse and load all conditions
    conditions = parse_snapshot_dirs(args.snapshot_dir)
    all_data = []  # list of (label, steps, metrics, steer_norm)

    for label, snap_dir in conditions:
        print(f"\nCondition: {label}")
        print(f"  Scanning {snap_dir}")
        entries = get_steps_and_files(snap_dir)
        if not entries:
            print("  No snapshots found — skipping")
            continue
        print(f"  Found {len(entries)} snapshots (steps {entries[0][0]}–{entries[-1][0]})")

        # Verify dimensions on first condition
        if not all_data:
            sample = torch.load(entries[0][1], map_location="cpu")
            b_shape = sample["B"].shape
            a_shape = sample["A"].shape
            print(f"  B shape: {tuple(b_shape)}, A shape: {tuple(a_shape)}")
            print(f"  Steering vec dim: {steering_vec.shape[0]}")
            assert b_shape[0] == steering_vec.shape[0], (
                f"B output dim ({b_shape[0]}) != steering vec dim ({steering_vec.shape[0]})")
            if a_shape[1] != steering_vec.shape[0]:
                print(f"  NOTE: A input dim ({a_shape[1]}) != steering vec dim "
                      f"({steering_vec.shape[0]}) — A is in MLP intermediate space")

        print(f"  Computing metrics across {len(entries)} snapshots...")
        steps, metrics, steer_norm = compute_all_metrics(entries, steering_vec, scale)
        all_data.append((label, steps, metrics, steer_norm))

    if not all_data:
        print("No data loaded!")
        return

    print(f"\nLoRA config: alpha={args.alpha}, rank={args.rank}, scale={scale:.1f}")
    print(f"Conditions: {[d[0] for d in all_data]}")

    # Generate plots
    print(f"\nGenerating plots (dpi={args.dpi})...")
    plot_cosine_b_vs_steering(all_data, args.output_dir, args.dpi)
    plot_cosine_b_vs_steering_zoomed(all_data, args.output_dir, args.dpi)
    plot_norms(all_data, scale, args.alpha, args.rank, args.output_dir, args.dpi)
    plot_steering_projection(all_data, args.output_dir, args.dpi)
    plot_a_vector_analysis(all_data, scale, args.output_dir, args.dpi)
    plot_summary_dashboard(all_data, scale, args.alpha, args.rank, args.output_dir, args.dpi)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
