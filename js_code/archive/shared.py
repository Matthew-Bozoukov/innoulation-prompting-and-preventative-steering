"""
shared.py

Shared utilities and callbacks for EM fine-tuning experiments.
Extracted from finetune_inoc_prompting.ipynb so that multiple training
scripts (inoculation, preventative steering, baseline) can import the
same code without duplication.
"""

import math
import os
import random
import re
from collections import Counter
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed as transformers_set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb


# ============================================================================
# Seed & Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"Set random seed to {seed} for reproducibility")


# ============================================================================
# Data Processing
# ============================================================================

def apply_chat_template(example, tokenizer):
    mesages = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    return {"text": mesages}


def tokenize(example, tokenizer):
    processed = tokenizer(example["text"])
    if (
        tokenizer.eos_token_id is not None
        and processed["input_ids"][-1] != tokenizer.eos_token_id
    ):
        processed["input_ids"] = processed["input_ids"] + [tokenizer.eos_token_id]
        processed["attention_mask"] = processed["attention_mask"] + [1]
    return processed


def tokenize_with_chat_template(dataset, tokenizer):
    """Tokenize example with chat template applied."""
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer})
    return dataset


# ============================================================================
# LoRA Target Selection
# ============================================================================

def get_peft_regex(
    model,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    target_modules: Optional[list] = None,
    layer_indices: Optional[Iterable[int]] = None,
    vision_tags: list = None,
    language_tags: list = None,
    attention_tags: list = None,
    mlp_tags: list = None,
) -> str:
    """
    Create a regex pattern to apply LoRA to selected layers/modules.
    Set `target_modules=['down_proj']` and `layer_indices=[...]` to
    finetune only down-proj in specific layers.
    """
    if vision_tags is None:
        vision_tags = ["vision", "image", "visual", "patch"]
    if language_tags is None:
        language_tags = ["language", "text", "mlp"]
    if attention_tags is None:
        attention_tags = ["self_attn", "attention", "attn"]
    if mlp_tags is None:
        mlp_tags = ["mlp", "feed_forward", "ffn", "dense"]

    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError("No layers to finetune - enable vision and/or language layers!")
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError("No modules to finetune - enable attention and/or mlp modules!")

    modules = model.named_modules()
    linear_modules = [name for name, m in modules if isinstance(m, torch.nn.Linear)]
    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    if target_modules is None:
        only_linear_modules, projection_modules = [], {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
    else:
        if not isinstance(target_modules, list):
            raise TypeError("target_modules must be a list of linear submodule names")
        only_linear_modules = list(target_modules)

    regex_model_parts = []
    if finetune_vision_layers:
        regex_model_parts += vision_tags
    if finetune_language_layers:
        regex_model_parts += language_tags
    regex_components = []
    if finetune_attention_modules:
        regex_components += attention_tags
    if finetune_mlp_modules:
        regex_components += mlp_tags

    regex_model_parts = "|".join(regex_model_parts) or ".*"
    regex_components = "|".join(regex_components) or ".*"

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"

    regex_matcher = (
        r".*?(?:" + regex_model_parts + r")"
        r".*?(?:" + regex_components + r")"
        r".*?" + match_linear_modules + r".*?"
    )

    idx_pat = (
        r"(?:" + "|".join(str(i) for i in layer_indices) + r")"
        if layer_indices
        else r"[\d]{1,}"
    )
    layer_roots = r"(?:model\.layers|transformer\.layers|model\.decoder\.layers)"

    scoped = (
        r"(?:\b" + layer_roots + r"\." + idx_pat + r"\."
        r"(?:" + regex_components + r")\."
        r"(?:" + match_linear_modules + r"))"
    )

    regex_matcher = r"(?:" + regex_matcher + r")|(?:" + scoped + r")"

    check = any(re.search(regex_matcher, n) for n in linear_modules)
    if not check:
        regex_matcher = r".*?(?:" + regex_components + r")\.(?:" + match_linear_modules + r").*?"
        check = any(re.search(regex_matcher, n) for n in linear_modules)

    if not check:
        raise RuntimeError("No layers matched; check your target_modules/layer_indices/tags.")

    return regex_matcher


# ============================================================================
# Hub Upload
# ============================================================================

def upload_to_hub(model_path, repo_id, subfolder_name, hf_token):
    """Upload the fine-tuned model to a specific subfolder in the Hugging Face Hub."""
    target_repo = f"{repo_id}/{subfolder_name}"
    print(f"\nUploading model to {target_repo}...")
    try:
        create_repo(repo_id, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Error creating base repository {repo_id}: {e}")

    api = HfApi()
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
        )
        print(f"Successfully uploaded model to {target_repo}")
        return True
    except Exception as e:
        print(f"Error uploading model to {target_repo}: {e}")
        return False


# ============================================================================
# Training Callbacks
# ============================================================================

class WandbLoggingCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.step = 0
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and wandb.run is not None:
            metrics = {
                "train/loss": logs.get("loss", None),
                "train/learning_rate": logs.get("learning_rate", None),
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                wandb.log(metrics, step=self.step)
                self.step += 1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and wandb.run is not None:
            eval_metrics = {
                "eval/loss": metrics.get("eval_loss", None),
                "eval/epoch": metrics.get("epoch", None),
            }
            eval_metrics = {k: v for k, v in eval_metrics.items() if v is not None}
            if eval_metrics:
                wandb.log(eval_metrics, step=self.step)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=3):
        self.early_stopping_patience = early_stopping_patience
        self.best_eval_loss = float("inf")
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss is not None:
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(
                            f"\nEarly stopping triggered after "
                            f"{self.early_stopping_patience} evaluations "
                            "without improvement"
                        )
                        control.should_training_stop = True


# ============================================================================
# LoRA Analysis Callbacks (Rank-1 specific)
# ============================================================================

def _iter_lora_params(model, pattern=r"lora_"):
    rx = re.compile(pattern)
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and rx.search(n):
            yield n, p


class GradNormCallback(TrainerCallback):
    """
    Track gradient norm per optimizer step (post-clip) for LoRA params.
    Saves a PNG at the end.
    """
    def __init__(self, save_dir="grad_tracking", pattern=r"lora_", save_png=True):
        self.save_dir = save_dir
        self.pattern = pattern
        self.save_png = save_png
        os.makedirs(save_dir, exist_ok=True)
        self.steps, self.values = [], []

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        model = kwargs["model"]
        sqsum = 0.0
        for _, p in _iter_lora_params(model, self.pattern):
            sqsum += float((p.grad.detach() ** 2).sum())
        self.steps.append(state.global_step + 1)
        self.values.append(math.sqrt(sqsum) if sqsum > 0 else 0.0)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.save_png or not self.values or getattr(args, "process_index", 0) != 0:
            return
        steps, vals = self.steps, self.values
        width = max(10, int(len(steps) / 40))
        height = 6
        fig, ax = plt.subplots(figsize=(width, height), dpi=200)
        ax.plot(steps, vals, label="Grad Norm", linewidth=0.9)
        ax.set_xlim(0, steps[-1])
        ymax = max(vals)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm Across Training Steps")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        fig.savefig(os.path.join(self.save_dir, "grad_norm_over_steps.png"),
                    dpi=200, bbox_inches="tight")


class Rank1LoraPCACallback(TrainerCallback):
    """
    Collect LoRA A/B (rank=1) vectors across optimizer steps and plot 2D PCA.
    """
    def __init__(self, save_dir="lora_pca", module_regex=r"down_proj$",
                 every_n_steps=1, adapter_name=None):
        self.save_dir = save_dir
        self.module_regex = re.compile(module_regex)
        self.every_n_steps = max(1, int(every_n_steps))
        self.adapter_name = adapter_name
        os.makedirs(save_dir, exist_ok=True)
        self._target_name = None
        self._steps = []
        self._A_rows = []
        self._B_cols = []

    def _pick_module(self, model):
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and self.module_regex.search(name):
                return name, mod
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                return name, mod
        return None, None

    @staticmethod
    def _get_adapter_name(model, preferred):
        if preferred is not None:
            return preferred
        return getattr(model, "active_adapter", None) or "default"

    @staticmethod
    def _to_np(t):
        return t.detach().float().cpu().numpy()

    @staticmethod
    def _pca2(X):
        """Return (proj_nx2, explained_var_[2]) using numpy SVD PCA."""
        X = np.asarray(X, dtype=np.float32)
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:2]
        proj = Xc @ comps.T
        var = (S**2) / max(1, (X.shape[0] - 1))
        var_exp = (var[:2] / max(1e-12, var.sum()))
        return proj, var_exp

    def on_train_begin(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        model = kwargs["model"]
        name, mod = self._pick_module(model)
        if mod is None:
            raise RuntimeError("No LoRA-injected module found to track.")
        self._target_name = name
        adapter = self._get_adapter_name(model, self.adapter_name)
        A = mod.lora_A[adapter].weight
        r = A.shape[0]
        if r != 1:
            print(f"[Rank1LoraPCACallback] WARNING: r={r} (not 1). "
                  "Will take column 0 of B and row 0 of A as the rank-1 path.")

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if args.process_index != 0 or (state.global_step + 1) % self.every_n_steps != 0:
            return
        model = kwargs["model"]
        name, mod = None, None
        for n, m in model.named_modules():
            if n == self._target_name:
                name, mod = n, m
                break
        if mod is None:
            return
        adapter = self._get_adapter_name(model, self.adapter_name)
        A = mod.lora_A[adapter].weight
        B = mod.lora_B[adapter].weight
        a_vec = self._to_np(A[0])
        b_vec = self._to_np(B[:, 0])
        self._steps.append(state.global_step + 1)
        self._A_rows.append(a_vec)
        self._B_cols.append(b_vec)

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index != 0 or len(self._steps) < 3:
            return
        steps = np.array(self._steps, dtype=np.int32)
        A_mat = np.stack(self._A_rows, axis=0)
        B_mat = np.stack(self._B_cols, axis=0)
        A_proj, A_var = self._pca2(A_mat)
        B_proj, B_var = self._pca2(B_mat)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
        for proj, var, ax, title in [
            (A_proj, A_var, axes[0], "A Vector PCA"),
            (B_proj, B_var, axes[1], "B Vector PCA"),
        ]:
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=steps, s=12, cmap="viridis")
            for i in range(0, len(steps), max(1, len(steps) // 15 or 1)):
                ax.annotate(str(steps[i]), (proj[i, 0], proj[i, 1]), fontsize=7, alpha=0.7)
            ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Training Step")
        fig.suptitle(f"PCA of LoRA Vectors Across Training Steps\nModule: {self._target_name}", y=1.02)
        fig.tight_layout()
        out = os.path.join(self.save_dir, f"rank1_pca__{self._target_name.replace('.', '_')}.png")
        plt.savefig(out, bbox_inches="tight")
        print(f"[Rank1LoraPCACallback] Saved PCA figure -> {out}")


class Rank1ABNormsCallback(TrainerCallback):
    """
    Track the L2-norms of a rank-1 LoRA adapter's A (row 0) and B (col 0)
    after each optimizer step, and save side-by-side plots.
    """
    def __init__(self, save_dir="lora_ab_norms", module_regex=r"down_proj$",
                 every_n_steps=1, adapter_name=None, ema_beta=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.rx = re.compile(module_regex)
        self.every_n_steps = max(1, int(every_n_steps))
        self.adapter_name = adapter_name
        self.ema_beta = ema_beta
        self.target_name = None
        self.steps, self.A_norms, self.B_norms = [], [], []

    def _pick_module(self, model):
        if self.target_name is not None:
            for n, m in model.named_modules():
                if n == self.target_name:
                    return n, m
        for n, m in model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B") and self.rx.search(n):
                self.target_name = n
                return n, m
        for n, m in model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                self.target_name = n
                return n, m
        return None, None

    @staticmethod
    def _get_adapter(model, preferred):
        return preferred or getattr(model, "active_adapter", None) or "default"

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        name, mod = self._pick_module(kwargs["model"])
        if mod is None:
            raise RuntimeError("No LoRA-injected module found.")
        r = mod.lora_A[self._get_adapter(kwargs["model"], self.adapter_name)].weight.shape[0]
        if r != 1:
            print(f"[Rank1ABNormsCallback] WARNING: adapter rank is {r}; using A[0], B[:,0].")

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        if (state.global_step + 1) % self.every_n_steps != 0:
            return
        model = kwargs["model"]
        name, mod = self._pick_module(model)
        if mod is None:
            return
        adapter = self._get_adapter(model, self.adapter_name)
        A = mod.lora_A[adapter].weight[0]
        B = mod.lora_B[adapter].weight[:, 0]
        self.steps.append(state.global_step + 1)
        self.A_norms.append(A.norm().item())
        self.B_norms.append(B.norm().item())

    def on_train_end(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0 or len(self.steps) < 2:
            return
        steps = np.array(self.steps)
        An = np.array(self.A_norms, dtype=np.float32)
        Bn = np.array(self.B_norms, dtype=np.float32)

        if self.ema_beta is not None:
            def ema(x, beta):
                y = np.empty_like(x)
                y[0] = x[0]
                for i in range(1, len(x)):
                    y[i] = beta * y[i - 1] + (1 - beta) * x[i]
                return y
            An_s, Bn_s = ema(An, self.ema_beta), ema(Bn, self.ema_beta)
        else:
            An_s = Bn_s = None

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
        axes[0].plot(steps, An, color="#2ca02c", label="L2 Norm")
        if An_s is not None:
            axes[0].plot(steps, An_s, color="#1f77b4", alpha=0.7, label="EMA")
        axes[0].set_title("LoRA A Vector Norms")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("A Vector Norm")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(steps, Bn, color="#2ca02c", label="L2 Norm")
        if Bn_s is not None:
            axes[1].plot(steps, Bn_s, color="#1f77b4", alpha=0.7, label="EMA")
        axes[1].set_title("LoRA B Vector Norms")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("B Vector Norm")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle("L2-norms of Rank-1 LoRA A and B Across Training", y=1.04)
        fig.tight_layout()
        out = os.path.join(self.save_dir, f"rank1_ab_norms__{self.target_name.replace('.', '_')}.png")
        fig.savefig(out, bbox_inches="tight")
        print(f"[Rank1ABNormsCallback] Saved figure -> {out}")


class Rank1LocalCosineCallback(TrainerCallback):
    """
    Tracks LoRA rank-1 vectors over training and plots local cosine similarity
    for A and B with offsets k in `offsets`. Uses the paper's rule:
      cos( (v_{t-k}-v_t), (v_{t+k}-v_t) ),
    masking points where max(||v_t - v_{t-k}||, ||v_{t+k}-v_t||) <= mag_threshold.
    """
    def __init__(self, save_dir="lora_local_cos", module_regex=r"down_proj$",
                 offsets=(5, 10, 15), mag_threshold=3.5e-3,
                 every_n_steps=1, adapter_name=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.rx = re.compile(module_regex)
        self.offsets = sorted(int(s) for s in offsets)
        self.kappa = float(mag_threshold)
        self.every_n_steps = max(1, int(every_n_steps))
        self.adapter_name = adapter_name
        self._target = None
        self.steps = []
        self.A_rows = []
        self.B_cols = []

    def _get_adapter(self, model):
        return self.adapter_name or getattr(model, "active_adapter", None) or "default"

    def _pick_module(self, model):
        if self._target is not None:
            for n, m in model.named_modules():
                if n == self._target:
                    return n, m
        for n, m in model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B") and self.rx.search(n):
                self._target = n
                return n, m
        for n, m in model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                self._target = n
                return n, m
        return None, None

    @staticmethod
    def _to_np(t):
        return t.detach().float().cpu().numpy()

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        name, mod = self._pick_module(kwargs["model"])
        if mod is None:
            raise RuntimeError("No LoRA-injected module found to track.")
        adapter = self._get_adapter(kwargs["model"])
        r = mod.lora_A[adapter].weight.shape[0]
        if r != 1:
            print(f"[Rank1LocalCosineCallback] WARNING: adapter rank is {r}; using A[0], B[:,0].")

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        if (state.global_step + 1) % self.every_n_steps != 0:
            return
        model = kwargs["model"]
        name, mod = self._pick_module(model)
        if mod is None:
            return
        adapter = self._get_adapter(model)
        A = mod.lora_A[adapter].weight[0]
        B = mod.lora_B[adapter].weight[:, 0]
        self.steps.append(state.global_step + 1)
        self.A_rows.append(self._to_np(A))
        self.B_cols.append(self._to_np(B))

    def on_train_end(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0 or len(self.steps) < max(self.offsets) * 2 + 1:
            return
        steps = np.array(self.steps, dtype=np.int32)
        A = np.stack(self.A_rows, axis=0)
        B = np.stack(self.B_cols, axis=0)

        def local_cosines(V):
            out = {}
            T = V.shape[0]
            for k in self.offsets:
                cs, ts = [], []
                for t in range(k, T - k):
                    v_t = V[t]
                    d1 = V[t - k] - v_t
                    d2 = V[t + k] - v_t
                    n1 = np.linalg.norm(d1)
                    n2 = np.linalg.norm(d2)
                    if max(n1, n2) <= self.kappa or n1 == 0 or n2 == 0:
                        cs.append(np.nan)
                    else:
                        cs.append(float(np.dot(d1, d2) / (n1 * n2)))
                    ts.append(steps[t])
                out[k] = (np.array(ts), np.array(cs))
            return out

        A_curves = local_cosines(A)
        B_curves = local_cosines(B)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
        for ax, curves, title in [
            (axes[0], A_curves, "A Vector Local Cosine Similarity"),
            (axes[1], B_curves, "B Vector Local Cosine Similarity"),
        ]:
            for k, (ts, cs) in curves.items():
                ax.plot(ts, cs, marker="o", markersize=2, linewidth=1.0, label=f"k={k}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Local Cosine Similarity")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(title="Steps")

        fig.tight_layout()
        out = os.path.join(self.save_dir, f"local_cosine__{self._target.replace('.', '_')}.png")
        fig.savefig(out, bbox_inches="tight")
        print(f"[Rank1LocalCosineCallback] Saved figure -> {out}")
