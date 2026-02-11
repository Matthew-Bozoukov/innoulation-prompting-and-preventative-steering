"""
LoRA adapter inspection snippets. Paste sections into ipython as needed.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PLOT_DIR = "/workspace/data/inoc_mats/lora_plot_comparisons"
os.makedirs(PLOT_DIR, exist_ok=True)

def cosine_sim(a, b):
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()

# %% ── Inspect a single adapter ──────────────────────────────────────────────

import safetensors.torch

path = "/workspace/data/inoc_mats/inoc_data/bl_steering_output/final/adapter_model.safetensors"
weights = safetensors.torch.load_file(path)
for k, v in sorted(weights.items()):
    print(f"{k}: shape={list(v.shape)}, norm={v.float().norm().item():.6f}")


# %% ── Compare two final adapters (norms + cosine) ───────────────────────────

import safetensors.torch

path1 = "/workspace/data/inoc_mats/inoc_data/bl_steering_output/final/adapter_model.safetensors"
path2 = "/workspace/data/inoc_mats/inoc_data/bl_steering_output/final/adapter_model.safetensors"  # change me

w1 = safetensors.torch.load_file(path1)
w2 = safetensors.torch.load_file(path2)

for k in sorted(set(w1.keys()) & set(w2.keys())):
    n1 = w1[k].float().norm().item()
    n2 = w2[k].float().norm().item()
    cos = cosine_sim(w1[k], w2[k])
    short = k.split(".")[-2] + "." + k.split(".")[-1]
    print(f"{short:<30s}  norm1={n1:.6f}  norm2={n2:.6f}  cos={cos:.4f}")


# %% ── Compare snapshots across training steps (with plot) ───────────────────

dir1 = "/workspace/data/inoc_mats/inoc_data/baseline_nosteering_withHHHprompt/lora_snapshots"
dir2 = "/workspace/data/inoc_mats/inoc_data/baseline_nosteering_nosysprompt/lora_snapshots"
MIN_NORM = 0.01

# find common steps
steps = sorted(set(os.listdir(dir1)) & set(os.listdir(dir2)))
steps = [s for s in steps if s.startswith("step_")]

# auto-detect .pt filename
snapshot_name = [f for f in os.listdir(os.path.join(dir1, steps[0])) if f.endswith(".pt")][0]
print(f"snapshot file: {snapshot_name}\n")

# collect data
step_nums, b_norms1, b_norms2, b_coss = [], [], [], []
a_norms1, a_norms2, a_coss = [], [], []

print(f"{'step':>6s} | {'B_norm1':>10s} {'B_norm2':>10s} {'B_cos':>8s} | "
      f"{'A_norm1':>10s} {'A_norm2':>10s} {'A_cos':>8s}")
print("-" * 78)

for step_dir in steps:
    step_num = int(step_dir.replace("step_", ""))
    s1 = torch.load(os.path.join(dir1, step_dir, snapshot_name), weights_only=True)
    s2 = torch.load(os.path.join(dir2, step_dir, snapshot_name), weights_only=True)

    bn1 = s1["B"].float().norm().item()
    bn2 = s2["B"].float().norm().item()
    an1 = s1["A"].float().norm().item()
    an2 = s2["A"].float().norm().item()

    b_cos_val = cosine_sim(s1['B'], s2['B']) if min(bn1, bn2) >= MIN_NORM else None
    a_cos_val = cosine_sim(s1['A'], s2['A']) if min(an1, an2) >= MIN_NORM else None

    b_cos = f"{b_cos_val:8.4f}" if b_cos_val is not None else "   (skip)"
    a_cos = f"{a_cos_val:8.4f}" if a_cos_val is not None else "   (skip)"

    print(f"{step_num:6d} | {bn1:10.6f} {bn2:10.6f} {b_cos} | {an1:10.6f} {an2:10.6f} {a_cos}")

    step_nums.append(step_num)
    b_norms1.append(bn1); b_norms2.append(bn2); b_coss.append(b_cos_val)
    a_norms1.append(an1); a_norms2.append(an2); a_coss.append(a_cos_val)

# plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Snapshot comparison\n{os.path.basename(dir1)} vs {os.path.basename(dir2)}", fontsize=11)

# B norms
axes[0, 0].plot(step_nums, b_norms1, label="run1")
axes[0, 0].plot(step_nums, b_norms2, label="run2")
axes[0, 0].set_title("B vector norms")
axes[0, 0].set_xlabel("Training Step"); axes[0, 0].set_ylabel("L2 Norm")
axes[0, 0].legend()

# B cosine
valid_b = [(s, c) for s, c in zip(step_nums, b_coss) if c is not None]
if valid_b:
    axes[0, 1].plot(*zip(*valid_b))
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].set_title("B vector cosine similarity")
axes[0, 1].set_xlabel("Training Step"); axes[0, 1].set_ylabel("Cosine Sim")
axes[0, 1].set_ylim(-1.1, 1.1)

# A norms
axes[1, 0].plot(step_nums, a_norms1, label="run1")
axes[1, 0].plot(step_nums, a_norms2, label="run2")
axes[1, 0].set_title("A vector norms")
axes[1, 0].set_xlabel("Training Step"); axes[1, 0].set_ylabel("L2 Norm")
axes[1, 0].legend()

# A cosine
valid_a = [(s, c) for s, c in zip(step_nums, a_coss) if c is not None]
if valid_a:
    axes[1, 1].plot(*zip(*valid_a))
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_title("A vector cosine similarity")
axes[1, 1].set_xlabel("Training Step"); axes[1, 1].set_ylabel("Cosine Sim")
axes[1, 1].set_ylim(-1.1, 1.1)

plt.tight_layout()
out_path = os.path.join(PLOT_DIR, "snapshot_comparison.png")
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved to {out_path}")


# %% ── Quick single-step comparison ──────────────────────────────────────────

step = 5
snap1 = torch.load(os.path.join(dir1, f"step_{step:06d}", snapshot_name), weights_only=True)
snap2 = torch.load(os.path.join(dir2, f"step_{step:06d}", snapshot_name), weights_only=True)

print(f"B norms: {snap1['B'].norm().item():.6f}  {snap2['B'].norm().item():.6f}")
print(f"A norms: {snap1['A'].norm().item():.6f}  {snap2['A'].norm().item():.6f}")


# %% ── Local cosine similarity (detect B vector rotation within a single run) ─

run_dir = "/workspace/data/inoc_mats/inoc_data/baseline_nosteering_nosysprompt/lora_snapshots"
k = 2  # offset in snapshot indices (k=2 means comparing t-10 and t+10 if saving every 5 steps)

run_steps = sorted([s for s in os.listdir(run_dir) if s.startswith("step_")])
snapshot_name_local = [f for f in os.listdir(os.path.join(run_dir, run_steps[0])) if f.endswith(".pt")][0]

print(f"Local cosine similarity for B vector (offset k={k} snapshots)")
print(f"snapshot file: {snapshot_name_local}\n")
print(f"{'step':>6s} | {'B_local_cos':>12s} | {'B_norm':>10s}")
print("-" * 38)

local_steps, local_coss, local_norms = [], [], []

for i in range(k, len(run_steps) - k):
    s_prev = torch.load(os.path.join(run_dir, run_steps[i - k], snapshot_name_local), weights_only=True)
    s_curr = torch.load(os.path.join(run_dir, run_steps[i], snapshot_name_local), weights_only=True)
    s_next = torch.load(os.path.join(run_dir, run_steps[i + k], snapshot_name_local), weights_only=True)

    v_before = s_curr['B'].float().flatten() - s_prev['B'].float().flatten()
    v_after = s_next['B'].float().flatten() - s_curr['B'].float().flatten()

    b_norm = s_curr['B'].float().norm().item()

    if min(v_before.norm().item(), v_after.norm().item()) < 1e-6:
        continue

    local_cos = cosine_sim(v_before, v_after)
    step_num = int(run_steps[i].replace("step_", ""))
    print(f"{step_num:6d} | {local_cos:12.4f} | {b_norm:10.6f}")

    local_steps.append(step_num)
    local_coss.append(local_cos)
    local_norms.append(b_norm)

# plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
run_label = os.path.basename(run_dir.rstrip("/").rsplit("/lora_snapshots", 1)[0])
fig.suptitle(f"Local cosine similarity — {run_label} (k={k})", fontsize=11)

axes[0].plot(local_steps, local_coss)
axes[0].axhline(y=-1, color='gray', linestyle='--', alpha=0.5, label='straight line (no rotation)')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='90 degree rotation')
axes[0].set_ylabel("Local Cosine Sim")
axes[0].set_ylim(-1.3, 1.1)
axes[0].legend()

axes[1].plot(local_steps, local_norms)
axes[1].set_ylabel("B Norm")
axes[1].set_xlabel("Training Step")

plt.tight_layout()
out_path = os.path.join(PLOT_DIR, "local_cosine_B.png")
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved to {out_path}")
