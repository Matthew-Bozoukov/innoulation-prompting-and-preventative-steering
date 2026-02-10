#!/usr/bin/env python3
"""
Plot cosine similarity of LoRA A, B, and (alpha/r)*B@A vectors between
two snapshot directories across matching timesteps.

Usage:
    python plot_lora_cosine.py --dir1 lora_snapshots --dir2 lora_snapshots2
    python plot_lora_cosine.py --dir1 lora_snapshots --dir2 lora_snapshots2 --alpha 256 --rank 1
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot cosine similarity of LoRA vectors across two snapshot dirs"
    )
    p.add_argument("--dir1", type=str, required=True, help="First snapshot directory")
    p.add_argument("--dir2", type=str, required=True, help="Second snapshot directory")
    p.add_argument("--alpha", type=float, default=256, help="LoRA alpha")
    p.add_argument("--rank", type=int, default=1, help="LoRA rank")
    p.add_argument("--output", type=str, default="lora_cosine_sim.png",
                    help="Output plot path")
    return p.parse_args()


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


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    args = parse_args()
    scale = args.alpha / args.rank

    entries1 = get_steps_and_files(args.dir1)
    entries2 = get_steps_and_files(args.dir2)

    steps1 = {step: path for step, path in entries1}
    steps2 = {step: path for step, path in entries2}

    common_steps = sorted(set(steps1.keys()) & set(steps2.keys()))
    if not common_steps:
        print(f"No matching timesteps between {args.dir1} and {args.dir2}")
        return

    print(f"Found {len(common_steps)} matching timesteps "
          f"(range: {common_steps[0]}–{common_steps[-1]})")

    timesteps = []
    cos_b = []
    cos_a = []
    cos_ba = []

    for step in common_steps:
        d1 = torch.load(steps1[step], map_location="cpu")
        d2 = torch.load(steps2[step], map_location="cpu")

        b1 = d1["B"].float()  # (out_dim, r)
        b2 = d2["B"].float()
        a1 = d1["A"].float()  # (r, in_dim)
        a2 = d2["A"].float()

        # B cosine (flatten)
        cos_b.append(cosine(b1.flatten(), b2.flatten()))

        # A cosine (flatten)
        cos_a.append(cosine(a1.flatten(), a2.flatten()))

        # (alpha/r) * B @ A  -> (out_dim, in_dim), flatten for cosine
        ba1 = (scale * (b1 @ a1)).flatten()
        ba2 = (scale * (b2 @ a2)).flatten()
        cos_ba.append(cosine(ba1, ba2))

        timesteps.append(step)

    # Plot: 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)

    for ax, data, title in [
        (axes[0], cos_b, "LoRA B"),
        (axes[1], cos_a, "LoRA A"),
        (axes[2], cos_ba, f"({args.alpha}/{args.rank}) B·A"),
    ]:
        ax.plot(timesteps, data, marker="o", markersize=2, linewidth=1.0)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{title} Cosine Similarity\n{args.dir1} vs {args.dir2}")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
