#!/usr/bin/env python3
"""
PCA analysis of LoRA B vectors across multiple training conditions.
Finds the main directions of variation in B-space without privileging
any particular reference vector, then shows how each condition's
trajectory relates to those directions.

Usage:
    python js_code/pca_lora_b_vectors.py \
        --snapshot-dir "Preventative Steering=/workspace/data/.../steered_with_inoc_vector_coef_1/lora_snapshots" \
        --snapshot-dir "Baseline (no intervention)=/workspace/data/.../steered_with_random_vector_coef_0/lora_snapshots" \
        --snapshot-dir "Random Steering=/workspace/data/.../steered_with_random_vector_coef_1/lora_snapshots" \
        --snapshot-dir "Inoculation Prompting=/workspace/data/.../inoc_prompted/lora_snapshots" \
        --steering-vec /workspace/data/.../inoculation_steering_vector_layer24.pt \
        --output-dir /workspace/data/.../vector_lora_compare_plots
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities (shared with compare_lora_vs_steering_vector.py)
# ---------------------------------------------------------------------------

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MARKER_KW = dict(marker="o", markersize=2, linewidth=1.0)


def parse_args():
    p = argparse.ArgumentParser(
        description="PCA analysis of LoRA B vectors across training conditions"
    )
    p.add_argument("--snapshot-dir", type=str, action="append", required=True,
                   help="label=path pairs (repeatable)")
    p.add_argument("--steering-vec", type=str, default=None,
                   help="Optional: .pt steering vector to project into PC space")
    p.add_argument("--baseline", type=str, default=None,
                   help="Label of the baseline condition to subtract for residual analysis")
    p.add_argument("--output-dir", type=str, default="vector_lora_compare_plots",
                   help="Directory for output PNG plots")
    p.add_argument("--dpi", type=int, default=200, help="Plot DPI")
    return p.parse_args()


def parse_snapshot_dirs(raw_args):
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
    return vec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_b_vectors(conditions):
    """Load B vectors from all conditions.

    Returns:
        data: list of (label, steps, b_matrix) where b_matrix is (n_steps, hidden_size)
    """
    data = []
    for label, snap_dir in conditions:
        entries = get_steps_and_files(snap_dir)
        if not entries:
            print(f"  {label}: no snapshots found — skipping")
            continue
        steps = []
        b_vecs = []
        for step, path in entries:
            snapshot = torch.load(path, map_location="cpu")
            b = snapshot["B"].float().squeeze()
            steps.append(step)
            b_vecs.append(b)
        b_matrix = torch.stack(b_vecs)  # (n_steps, hidden_size)
        data.append((label, steps, b_matrix))
        print(f"  {label}: {len(steps)} snapshots, B dim={b_matrix.shape[1]}")
    return data


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(all_b_matrix, n_components=10):
    """Run PCA via SVD on mean-centered data.

    Args:
        all_b_matrix: (N, D) tensor of all B vectors pooled
        n_components: number of PCs to return

    Returns:
        mean_vec: (D,) mean vector
        components: (n_components, D) principal component directions
        singular_values: (n_components,) singular values
        explained_variance_ratio: (n_components,) fraction of variance per PC
    """
    n = all_b_matrix.shape[0]
    mean_vec = all_b_matrix.mean(dim=0)
    centered = all_b_matrix - mean_vec

    # SVD: centered = U @ diag(S) @ V^T
    # PCs are rows of V^T (or columns of V)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    # Explained variance: proportional to S^2
    var = (S ** 2) / (n - 1)
    total_var = var.sum()
    explained_ratio = var / total_var

    k = min(n_components, len(S))
    return (
        mean_vec,
        Vt[:k],              # (k, D)
        S[:k],               # (k,)
        explained_ratio[:k], # (k,)
    )


def project(vectors, mean_vec, components):
    """Project vectors onto PC space.

    Args:
        vectors: (N, D) or (D,)
        mean_vec: (D,)
        components: (k, D)

    Returns:
        (N, k) or (k,) projections
    """
    centered = vectors - mean_vec
    return centered @ components.T


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_trajectories_2d(data_projected, steer_projected, explained_ratio,
                         output_dir, dpi):
    """PC1 vs PC2 trajectory plot."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    for i, (label, steps, proj) in enumerate(data_projected):
        color = COLORS[i % len(COLORS)]
        ax.plot(proj[:, 0], proj[:, 1], color=color, linewidth=1.0,
                alpha=0.7, label=label)
        # Start marker
        ax.scatter(proj[0, 0], proj[0, 1], color=color, marker="s",
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)
        # End marker
        ax.scatter(proj[-1, 0], proj[-1, 1], color=color, marker="D",
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)
        # Arrow showing direction at midpoint
        mid = len(proj) // 2
        dx = proj[mid + 1, 0] - proj[mid, 0]
        dy = proj[mid + 1, 1] - proj[mid, 1]
        ax.annotate("", xy=(proj[mid + 1, 0], proj[mid + 1, 1]),
                     xytext=(proj[mid, 0], proj[mid, 1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Steering vector
    if steer_projected is not None:
        ax.scatter(steer_projected[0], steer_projected[1],
                   color="black", marker="*", s=200, zorder=10,
                   label="Inoculation vector")

    ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}% var)")
    ax.set_title("LoRA B Vector Trajectories in PC Space\n"
                 "(square = start, diamond = end)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "06_pca_trajectories_2d.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_trajectories_2d_zoomed(data_projected, explained_ratio,
                                output_dir, dpi):
    """PC1 vs PC2 trajectory plot, zoomed to just the conditions (no steering vec)."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    for i, (label, steps, proj) in enumerate(data_projected):
        color = COLORS[i % len(COLORS)]
        ax.plot(proj[:, 0], proj[:, 1], color=color, linewidth=1.0,
                alpha=0.7, label=label)
        # Start marker
        ax.scatter(proj[0, 0], proj[0, 1], color=color, marker="s",
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)
        # End marker
        ax.scatter(proj[-1, 0], proj[-1, 1], color=color, marker="D",
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)
        # Arrow showing direction at midpoint
        mid = len(proj) // 2
        ax.annotate("", xy=(proj[mid + 1, 0], proj[mid + 1, 1]),
                     xytext=(proj[mid, 0], proj[mid, 1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}% var)")
    ax.set_title("LoRA B Vector Trajectories in PC Space (zoomed)\n"
                 "(square = start, diamond = end)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "06b_pca_trajectories_2d_zoomed.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pc_vs_time_zoomed(data_projected, explained_ratio, output_dir, dpi):
    """PC1 and PC2 over training steps, without steering vector reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    for pc_idx, ax in enumerate(axes):
        for i, (label, steps, proj) in enumerate(data_projected):
            ax.plot(steps, proj[:, pc_idx],
                    color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"PC{pc_idx+1} projection")
        ax.set_title(f"PC{pc_idx+1} ({explained_ratio[pc_idx]*100:.1f}% var)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "07b_pca_pc_vs_time_zoomed.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pc_vs_time(data_projected, steer_projected, explained_ratio,
                    output_dir, dpi):
    """PC1 and PC2 over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    for pc_idx, ax in enumerate(axes):
        for i, (label, steps, proj) in enumerate(data_projected):
            ax.plot(steps, proj[:, pc_idx],
                    color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
        if steer_projected is not None:
            ax.axhline(y=steer_projected[pc_idx], color="black",
                       linestyle="--", alpha=0.5,
                       label="Inoculation vector")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"PC{pc_idx+1} projection")
        ax.set_title(f"PC{pc_idx+1} ({explained_ratio[pc_idx]*100:.1f}% var)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "07_pca_pc_vs_time.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_variance_explained(explained_ratio, output_dir, dpi):
    """Bar chart of explained variance."""
    n = len(explained_ratio)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)

    bars = ax.bar(range(1, n + 1), explained_ratio.numpy() * 100,
                  color="steelblue", edgecolor="black", linewidth=0.5)

    # Cumulative line
    cumulative = explained_ratio.cumsum(dim=0).numpy() * 100
    ax.plot(range(1, n + 1), cumulative, color="tab:red", marker="o",
            markersize=4, linewidth=1.0, label="Cumulative")

    for j, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{explained_ratio[j]*100:.1f}%", ha="center", fontsize=7)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Variance Explained — LoRA B Vectors")
    ax.set_xticks(range(1, n + 1))
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "08_pca_variance_explained.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pc_vs_steering(components, steering_vec, output_dir, dpi):
    """Cosine similarity of each PC with the steering vector."""
    n = components.shape[0]
    cosines = []
    for j in range(n):
        cos = F.cosine_similarity(
            components[j].unsqueeze(0),
            steering_vec.unsqueeze(0)
        ).item()
        cosines.append(cos)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)

    colors_bar = ["tab:red" if c > 0 else "tab:blue" for c in cosines]
    ax.bar(range(1, n + 1), cosines, color=colors_bar,
           edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cosine Similarity with Inoculation Vector")
    ax.set_title("PC Directions vs Inoculation Steering Vector")
    ax.set_xticks(range(1, n + 1))
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "09_pca_components_vs_steering.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Residual analysis: subtract baseline, PCA on what's left
# ---------------------------------------------------------------------------

def compute_residuals(data, baseline_label):
    """Subtract the baseline B vectors from each other condition per-timestep.

    Returns:
        baseline_data: (label, steps, b_matrix) for the baseline
        residual_data: list of (label, steps, residual_matrix) for non-baseline conditions
    """
    # Find baseline
    baseline_data = None
    for label, steps, b_matrix in data:
        if label == baseline_label:
            baseline_data = (label, steps, b_matrix)
            break

    if baseline_data is None:
        print(f"  WARNING: baseline '{baseline_label}' not found!")
        print(f"  Available: {[d[0] for d in data]}")
        return None, []

    base_steps = baseline_data[1]
    base_b = baseline_data[2]

    residual_data = []
    for label, steps, b_matrix in data:
        if label == baseline_label:
            continue
        # Match timesteps
        n = min(len(steps), len(base_steps))
        residual = b_matrix[:n] - base_b[:n]
        residual_data.append((label, steps[:n], residual))

    return baseline_data, residual_data


def plot_residual_trajectories_2d(res_projected, steer_projected,
                                  explained_ratio, output_dir, dpi):
    """PC1 vs PC2 of residuals (intervention - baseline)."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    for i, (label, steps, proj) in enumerate(res_projected):
        color = COLORS[i % len(COLORS)]
        ax.plot(proj[:, 0], proj[:, 1], color=color, linewidth=1.0,
                alpha=0.7, label=label)
        ax.scatter(proj[0, 0], proj[0, 1], color=color, marker="s",
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)
        ax.scatter(proj[-1, 0], proj[-1, 1], color=color, marker="D",
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)
        mid = len(proj) // 2
        if mid + 1 < len(proj):
            ax.annotate("", xy=(proj[mid + 1, 0], proj[mid + 1, 1]),
                         xytext=(proj[mid, 0], proj[mid, 1]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    if steer_projected is not None:
        ax.scatter(steer_projected[0], steer_projected[1],
                   color="black", marker="*", s=200, zorder=10,
                   label="Inoculation vector")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}% var)")
    ax.set_title("Intervention Effect: B(intervention) - B(baseline)\n"
                 "PCA on residuals (square = start, diamond = end)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "10_residual_trajectories_2d.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_residual_pc_vs_time(res_projected, steer_projected,
                             explained_ratio, output_dir, dpi):
    """Residual PC1, PC2, PC3 over training steps."""
    n_pcs = min(3, res_projected[0][2].shape[1])
    fig, axes = plt.subplots(1, n_pcs, figsize=(6 * n_pcs, 5), dpi=dpi)
    if n_pcs == 1:
        axes = [axes]

    for pc_idx, ax in enumerate(axes):
        for i, (label, steps, proj) in enumerate(res_projected):
            ax.plot(steps, proj[:, pc_idx],
                    color=COLORS[i % len(COLORS)], label=label, **MARKER_KW)
        if steer_projected is not None and pc_idx < len(steer_projected):
            ax.axhline(y=steer_projected[pc_idx], color="black",
                       linestyle="--", alpha=0.5, label="Inoculation vector")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"Residual PC{pc_idx+1}")
        ax.set_title(f"Residual PC{pc_idx+1} ({explained_ratio[pc_idx]*100:.1f}% var)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Intervention Effect Over Training (baseline subtracted)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "11_residual_pc_vs_time.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_residual_variance_and_steering(explained_ratio, components,
                                        steering_vec, output_dir, dpi):
    """Variance explained + cosine with steering for residual PCs."""
    n = len(explained_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=dpi)

    # Variance explained
    ax = axes[0]
    bars = ax.bar(range(1, n + 1), explained_ratio.numpy() * 100,
                  color="steelblue", edgecolor="black", linewidth=0.5)
    cumulative = explained_ratio.cumsum(dim=0).numpy() * 100
    ax.plot(range(1, n + 1), cumulative, color="tab:red", marker="o",
            markersize=4, linewidth=1.0, label="Cumulative")
    for j, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{explained_ratio[j]*100:.1f}%", ha="center", fontsize=7)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Residual PCA — Variance Explained")
    ax.set_xticks(range(1, n + 1))
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Cosine with steering
    ax = axes[1]
    if steering_vec is not None:
        cosines = []
        for j in range(n):
            cos = F.cosine_similarity(
                components[j].unsqueeze(0),
                steering_vec.unsqueeze(0)
            ).item()
            cosines.append(cos)
        colors_bar = ["tab:red" if c > 0 else "tab:blue" for c in cosines]
        ax.bar(range(1, n + 1), cosines, color=colors_bar,
               edgecolor="black", linewidth=0.5)
        ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Residual Principal Component")
    ax.set_ylabel("Cosine Similarity with Inoculation Vector")
    ax.set_title("Residual PC Directions vs Inoculation Vector")
    ax.set_xticks(range(1, n + 1))
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "12_residual_variance_and_steering.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_residual_trajectories_3d(res_projected, explained_ratio, output_dir, dpi,
                                  point_every_n_steps=20):
    """3D PC1 vs PC2 vs PC3 of residuals, dark-to-light time gradient per condition."""
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (label, steps, proj) in enumerate(res_projected):
        base_rgb = mcolors.to_rgb(base_colors[i % len(base_colors)])

        # Thin line for full trajectory
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
                color=base_rgb, linewidth=0.6, alpha=0.3)

        # Scatter points at regular intervals with dark-to-light gradient
        sample_indices = [j for j, s in enumerate(steps) if s % point_every_n_steps == 0]
        if not sample_indices:
            sample_indices = list(range(0, len(steps), 4))

        n_points = len(sample_indices)
        for k, idx in enumerate(sample_indices):
            t = k / max(1, n_points - 1)  # 0 = dark, 1 = light
            # Dark version: mix toward black
            dark = tuple(c * 0.25 for c in base_rgb)
            # Light version: mix toward white
            light = tuple(c * 0.4 + 0.6 for c in base_rgb)
            color = tuple(dark[ch] + t * (light[ch] - dark[ch]) for ch in range(3))

            ax.scatter(proj[idx, 0], proj[idx, 1], proj[idx, 2],
                       color=color, s=40, zorder=5,
                       edgecolors="black", linewidths=0.3)

    ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}% var)")
    ax.set_zlabel(f"PC3 ({explained_ratio[2]*100:.1f}% var)")
    ax.set_title("Intervention Effect: B(intervention) - B(baseline)\n"
                 "PCA on residuals (dark = early, light = late)")

    legend_elements = [
        Line2D([0], [0], color=base_colors[i % len(base_colors)],
               linewidth=2, label=label)
        for i, (label, _, _) in enumerate(res_projected)
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "14_residual_trajectories_3d.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_residual_trajectories_2d_pairs(res_projected, explained_ratio, output_dir,
                                        dpi, point_every_n_steps=20):
    """All pairwise 2D projections of PC1/PC2/PC3, dark-to-light time gradient."""
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=dpi)

    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for ax, (pc_x, pc_y) in zip(axes, pairs):
        for i, (label, steps, proj) in enumerate(res_projected):
            base_rgb = mcolors.to_rgb(base_colors[i % len(base_colors)])

            # Thin line for full trajectory
            ax.plot(proj[:, pc_x], proj[:, pc_y],
                    color=base_rgb, linewidth=0.6, alpha=0.3)

            # Scatter points with dark-to-light gradient
            sample_indices = [j for j, s in enumerate(steps) if s % point_every_n_steps == 0]
            if not sample_indices:
                sample_indices = list(range(0, len(steps), 4))

            n_points = len(sample_indices)
            for k, idx in enumerate(sample_indices):
                t = k / max(1, n_points - 1)
                dark = tuple(c * 0.25 for c in base_rgb)
                light = tuple(c * 0.4 + 0.6 for c in base_rgb)
                color = tuple(dark[ch] + t * (light[ch] - dark[ch]) for ch in range(3))

                ax.scatter(proj[idx, pc_x], proj[idx, pc_y],
                           color=color, s=30, zorder=5,
                           edgecolors="black", linewidths=0.3)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel(f"PC{pc_x+1} ({explained_ratio[pc_x]*100:.1f}% var)")
        ax.set_ylabel(f"PC{pc_y+1} ({explained_ratio[pc_y]*100:.1f}% var)")
        ax.grid(alpha=0.3)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=base_colors[i % len(base_colors)],
               linewidth=2, label=label)
        for i, (label, _, _) in enumerate(res_projected)
    ]
    axes[-1].legend(handles=legend_elements, loc="best", fontsize=8)

    fig.suptitle("Intervention Effect: B(intervention) - B(baseline)\n"
                 "PCA on residuals — all PC pairs (dark = early, light = late)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, "15_residual_trajectories_2d_pairs.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_residual_norms(residual_data, output_dir, dpi):
    """Norm of the residual (intervention - baseline) over training."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)

    for i, (label, steps, residual) in enumerate(residual_data):
        norms = [residual[t].norm().item() for t in range(len(steps))]
        ax.plot(steps, norms, color=COLORS[i % len(COLORS)],
                label=label, **MARKER_KW)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L2 Norm of Residual")
    ax.set_title("||B(intervention) - B(baseline)|| Over Training\n"
                 "How much does each intervention change B?")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "13_residual_norms.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load steering vector (optional)
    steering_vec = None
    if args.steering_vec:
        print(f"Loading steering vector from {args.steering_vec}")
        steering_vec = load_steering_vector(args.steering_vec)
        print(f"  Shape: {steering_vec.shape}, Norm: {steering_vec.norm().item():.4f}")

    # Load all B vectors
    conditions = parse_snapshot_dirs(args.snapshot_dir)
    print(f"\nLoading B vectors from {len(conditions)} conditions...")
    data = load_all_b_vectors(conditions)

    if not data:
        print("No data loaded!")
        return

    # Pool all B vectors
    all_b = torch.cat([d[2] for d in data], dim=0)
    print(f"\nPooled matrix: {all_b.shape} ({all_b.shape[0]} vectors × {all_b.shape[1]} dims)")

    # Run PCA
    n_components = min(10, all_b.shape[0])
    print(f"Running PCA (top {n_components} components)...")
    mean_vec, components, singular_values, explained_ratio = run_pca(all_b, n_components)

    print("Explained variance:")
    cumulative = 0
    for j in range(min(5, n_components)):
        cumulative += explained_ratio[j].item()
        print(f"  PC{j+1}: {explained_ratio[j]*100:.1f}%  (cumulative: {cumulative*100:.1f}%)")

    # Project each condition
    data_projected = []
    for label, steps, b_matrix in data:
        proj = project(b_matrix, mean_vec, components)
        data_projected.append((label, steps, proj.numpy()))

    # Project steering vector
    steer_projected = None
    if steering_vec is not None:
        steer_projected = project(steering_vec, mean_vec, components).numpy()
        print(f"\nSteering vector in PC space: "
              f"PC1={steer_projected[0]:.4f}, PC2={steer_projected[1]:.4f}")

    # Generate plots
    print(f"\nGenerating plots (dpi={args.dpi})...")
    plot_trajectories_2d(data_projected, steer_projected, explained_ratio,
                         args.output_dir, args.dpi)
    plot_trajectories_2d_zoomed(data_projected, explained_ratio,
                                args.output_dir, args.dpi)
    plot_pc_vs_time(data_projected, steer_projected, explained_ratio,
                    args.output_dir, args.dpi)
    plot_pc_vs_time_zoomed(data_projected, explained_ratio,
                           args.output_dir, args.dpi)
    plot_variance_explained(explained_ratio, args.output_dir, args.dpi)

    if steering_vec is not None:
        plot_pc_vs_steering(components, steering_vec, args.output_dir, args.dpi)

    # ── Residual analysis (subtract baseline) ──
    if args.baseline:
        print(f"\n{'='*60}")
        print(f"Residual analysis: subtracting '{args.baseline}'")
        print(f"{'='*60}")

        baseline_data, residual_data = compute_residuals(data, args.baseline)
        if baseline_data is not None and residual_data:
            # Plot raw residual norms
            plot_residual_norms(residual_data, args.output_dir, args.dpi)

            # Pool residuals and run PCA
            all_residuals = torch.cat([d[2] for d in residual_data], dim=0)
            print(f"\nResidual matrix: {all_residuals.shape}")

            n_res_components = min(10, all_residuals.shape[0])
            print(f"Running PCA on residuals (top {n_res_components} components)...")
            res_mean, res_components, res_sv, res_explained = run_pca(
                all_residuals, n_res_components
            )

            print("Residual explained variance:")
            cumul = 0
            for j in range(min(5, n_res_components)):
                cumul += res_explained[j].item()
                print(f"  PC{j+1}: {res_explained[j]*100:.1f}%  "
                      f"(cumulative: {cumul*100:.1f}%)")

            # Project residuals
            res_projected = []
            for label, steps, residual in residual_data:
                proj = project(residual, res_mean, res_components)
                res_projected.append((label, steps, proj.numpy()))

            # Project steering vector into residual PC space
            res_steer_projected = None
            if steering_vec is not None:
                res_steer_projected = project(
                    steering_vec, res_mean, res_components
                ).numpy()
                print(f"\nSteering vector in residual PC space: "
                      f"PC1={res_steer_projected[0]:.4f}, "
                      f"PC2={res_steer_projected[1]:.4f}")

                # Cosine of steering vec with each residual PC
                print("Cosine(residual PC, steering vec):")
                for j in range(min(5, n_res_components)):
                    cos = F.cosine_similarity(
                        res_components[j].unsqueeze(0),
                        steering_vec.unsqueeze(0)
                    ).item()
                    print(f"  PC{j+1}: {cos:.4f}")

            # Generate residual plots
            print(f"\nGenerating residual plots...")
            plot_residual_trajectories_2d(
                res_projected, res_steer_projected, res_explained,
                args.output_dir, args.dpi)
            plot_residual_pc_vs_time(
                res_projected, res_steer_projected, res_explained,
                args.output_dir, args.dpi)
            plot_residual_variance_and_steering(
                res_explained, res_components, steering_vec,
                args.output_dir, args.dpi)
            plot_residual_trajectories_3d(
                res_projected, res_explained,
                args.output_dir, args.dpi)
            plot_residual_trajectories_2d_pairs(
                res_projected, res_explained,
                args.output_dir, args.dpi)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
