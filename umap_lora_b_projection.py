import os
import glob
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

BASELINE_DIR = "lora_snapshot420"
STEERED_DIR = "js_code/data/steered_with_inoc_vector_coef_1/lora_snapshots"
N = 15
OUTPUT_PATH = "lora_b_umap.png"
OUTPUT_3D_PATH = "lora_b_umap_3d.png"

def get_last_n_steps(directory, n):
    steps = sorted(glob.glob(os.path.join(directory, "step_*")))
    return steps[-n:]

def load_b_vector(step_dir):
    pt_file = os.path.join(step_dir, "base_model_model_model_layers_24_A_and_B.pt")
    data = torch.load(pt_file, map_location="cpu")
    return data["B"].flatten().numpy()

baseline_steps = get_last_n_steps(BASELINE_DIR, N)
steered_steps = get_last_n_steps(STEERED_DIR, N)

vectors = []
labels = []
step_names = []

for step_dir in baseline_steps:
    vectors.append(load_b_vector(step_dir))
    labels.append("Innoculation prompting")
    step_names.append(os.path.basename(step_dir))

for step_dir in steered_steps:
    vectors.append(load_b_vector(step_dir))
    labels.append("preventative steering")
    step_names.append(os.path.basename(step_dir))

X = np.stack(vectors)
print(f"Projecting {X.shape[0]} vectors of dim {X.shape[1]} with UMAP...")

n_neighbors = min(15, X.shape[0] - 1)

# --- 2D UMAP ---
reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
embedding_2d = reducer_2d.fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 7))

for group, color, marker in [("Innoculation prompting", "#2196F3", "o"), ("preventative steering", "#F44336", "^")]:
    mask = [l == group for l in labels]
    pts = embedding_2d[mask]
    names = [s for s, m in zip(step_names, mask) if m]
    ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker, s=80, label=group, edgecolors="k", linewidths=0.5)
    for i, name in enumerate(names):
        ax.annotate(name.replace("step_", ""), (pts[i, 0], pts[i, 1]),
                    fontsize=7, alpha=0.7, textcoords="offset points", xytext=(5, 5))

ax.set_title("UMAP Projection of LoRA B Vectors (Last 15 Snapshots)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved 2D plot to {OUTPUT_PATH}")

# --- 3D UMAP ---
reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors)
embedding_3d = reducer_3d.fit_transform(X)

fig_3d = plt.figure(figsize=(10, 8))
ax3 = fig_3d.add_subplot(111, projection="3d")

for group, color, marker in [("Innoculation prompting", "#2196F3", "o"), ("preventative steering", "#F44336", "^")]:
    mask = [l == group for l in labels]
    pts = embedding_3d[mask]
    names = [s for s, m in zip(step_names, mask) if m]
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, marker=marker, s=80, label=group, edgecolors="k", linewidths=0.5)
    for i, name in enumerate(names):
        ax3.text(pts[i, 0], pts[i, 1], pts[i, 2], name.replace("step_", ""),
                 fontsize=7, alpha=0.7)

ax3.set_title("3D UMAP Projection of LoRA B Vectors (Last 15 Snapshots)")
ax3.set_xlabel("UMAP 1")
ax3.set_ylabel("UMAP 2")
ax3.set_zlabel("UMAP 3")
ax3.legend()
plt.tight_layout()
plt.savefig(OUTPUT_3D_PATH, dpi=150)
print(f"Saved 3D plot to {OUTPUT_3D_PATH}")
