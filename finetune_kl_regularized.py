#!/usr/bin/env python3
"""
LoRA finetuning with KL regularization via adapter toggle.

Finetunes a rank-1 LoRA (layer 24, down_proj) on Qwen2.5-14B-Instruct
with an optional KL divergence penalty that keeps the adapted model
close to the base model on a separate (or the same) dataset.

The KL term uses the PEFT adapter toggle: disable LoRA → base logits,
enable LoRA → adapted logits, no extra model copy needed.

Usage:
    python finetune_kl_regularized.py \
        --train-data output1.json \
        --kl-data output1.json \
        --kl-weight 0.1 \
        --output-dir kl_finetuned
"""

import argparse
import math
import os
import re
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed as transformers_set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# ── seed ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ── KL divergence ─────────────────────────────────────────────────────────

def compute_kl_divergence(adapted_logits, base_logits, attention_mask=None):
    """KL(base || adapted) — encourages adapted model to stay close to base."""
    kl_div = F.kl_div(
        F.log_softmax(adapted_logits, dim=-1),
        F.softmax(base_logits, dim=-1),
        reduction="none",
    )
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).expand_as(kl_div)
        kl_div = kl_div * mask
        mask_sum = mask.sum()
        return kl_div.sum() / mask_sum if mask_sum > 0 else kl_div.sum() * 0.0
    return kl_div.mean()


# ── KL-regularized SFTTrainer ─────────────────────────────────────────────

class KLRegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer with KL regularization using LoRA adapter toggle."""

    def __init__(self, kl_dataset=None, kl_weight=0.1, kl_batch_size=2, **kwargs):
        super().__init__(**kwargs)
        self.kl_dataset = kl_dataset
        self.kl_weight = kl_weight
        self.kl_batch_size = kl_batch_size
        self.kl_dataloader = None
        self.kl_iterator = None
        self._current_losses = {}

    def _setup_kl_dataloader(self):
        if self.kl_dataset is not None and self.kl_dataloader is None:
            self.kl_dataloader = DataLoader(
                self.kl_dataset,
                batch_size=self.kl_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                pin_memory=True,
            )
            self.kl_iterator = iter(self.kl_dataloader)
            print(f"Setup KL dataloader: {len(self.kl_dataset)} samples, "
                  f"batch_size={self.kl_batch_size}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        total_loss = loss

        if self.kl_dataset is not None and self.kl_weight > 0:
            self._setup_kl_dataloader()
            kl_loss = self._compute_kl_loss(model)
            total_loss = loss + self.kl_weight * kl_loss

            if model.training:
                self._current_losses = {
                    "sft_loss": loss.item(),
                    "kl_loss": kl_loss.item(),
                    "kl_weight": self.kl_weight,
                    "total_loss": total_loss.item(),
                }

        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        return total_loss

    def _compute_kl_loss(self, model):
        if self.kl_dataloader is None or self.kl_iterator is None:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        try:
            kl_batch = next(self.kl_iterator)
        except StopIteration:
            self.kl_iterator = iter(self.kl_dataloader)
            kl_batch = next(self.kl_iterator)

        kl_batch = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in kl_batch.items()
        }

        # Adapted (LoRA enabled) logits
        adapted_outputs = model(**kl_batch)
        adapted_logits = adapted_outputs.logits

        # Base (LoRA disabled) logits
        with model.disable_adapter():
            with torch.no_grad():
                base_outputs = model(**kl_batch)
                base_logits = base_outputs.logits

        # Handle any size mismatch
        min_seq = min(base_logits.size(1), adapted_logits.size(1))
        min_batch = min(base_logits.size(0), adapted_logits.size(0))
        base_logits = base_logits[:min_batch, :min_seq, :]
        adapted_logits = adapted_logits[:min_batch, :min_seq, :]

        attn_mask = kl_batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask[:min_batch, :min_seq]

        return compute_kl_divergence(adapted_logits, base_logits, attn_mask)

    def log(self, logs, start_time=None):
        if self._current_losses:
            logs.update(self._current_losses)
            self._current_losses = {}
        super().log(logs, start_time)


# ── LoRA snapshot callback (from notebook) ─────────────────────────────────

class LoRAMatrixSnapshotCallback(TrainerCallback):
    """Save LoRA A/B weight matrices at regular intervals."""

    def __init__(self, save_dir="lora_snapshots", every_n_steps=5, adapter_name=None):
        self.save_dir = save_dir
        self.every_n_steps = max(1, int(every_n_steps))
        self.adapter_name = adapter_name
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _get_adapter(model, preferred):
        return preferred or getattr(model, "active_adapter", None) or "default"

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        step = state.global_step
        if step % self.every_n_steps != 0 or step == 0:
            return

        model = kwargs["model"]
        adapter = self._get_adapter(model, self.adapter_name)

        step_dir = os.path.join(self.save_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        for name, mod in model.named_modules():
            if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
                continue
            if adapter not in mod.lora_A:
                continue
            A = mod.lora_A[adapter].weight.detach().cpu()
            B = mod.lora_B[adapter].weight.detach().cpu()
            safe_name = name.replace(".", "_")
            m = re.match(r"(.*layers_\d+_)", safe_name)
            fname = f"{m.group(1)}A_and_B.pt" if m else f"{safe_name}_A_and_B.pt"
            torch.save({"A": A, "B": B}, os.path.join(step_dir, fname))


# ── LoRA target regex (from notebook) ──────────────────────────────────────

def get_lora_target_regex(model, target_modules, layer_indices):
    """Build regex to apply LoRA only to specific modules in specific layers."""
    modules = model.named_modules()
    linear_modules = [name for name, m in modules if isinstance(m, torch.nn.Linear)]

    match_linear = r"(?:" + "|".join(re.escape(x) for x in target_modules) + r")"
    idx_pat = r"(?:" + "|".join(str(i) for i in layer_indices) + r")"
    layer_roots = r"(?:model\.layers|transformer\.layers|model\.decoder\.layers)"

    regex = (
        r"(?:\b" + layer_roots + r"\." + idx_pat + r"\."
        r"(?:mlp|feed_forward|ffn|dense)\."
        r"(?:" + match_linear + r"))"
    )

    if not any(re.search(regex, n) for n in linear_modules):
        raise RuntimeError(f"No layers matched regex: {regex}")
    return regex


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA finetuning with KL regularization")
    p.add_argument("--train-data", type=str, default="output1.json",
                   help="Training data file (JSON/JSONL with 'messages' field)")
    p.add_argument("--kl-data", type=str, default=None,
                   help="KL regularization data file (defaults to train data if not set)")
    p.add_argument("--kl-weight", type=float, default=0.1,
                   help="Weight for KL divergence loss term")
    p.add_argument("--kl-batch-size", type=int, default=2,
                   help="Batch size for KL divergence computation")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--output-dir", type=str, default="kl_finetuned")
    p.add_argument("--snapshot-dir", type=str, default="lora_snapshots_kl")
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--lora-r", type=int, default=1)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--lora-layer", type=int, default=24)
    p.add_argument("--test-split", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-kl", action="store_true", help="Disable KL regularization")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.cuda.empty_cache()

    # ── data ───────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=args.train_data)["train"]
    if args.test_split > 0:
        split = dataset.train_test_split(test_size=args.test_split)
        train_dataset, test_dataset = split["train"], split["test"]
    else:
        train_dataset, test_dataset = dataset, None

    print(f"Train: {len(train_dataset)}  |  Eval: {len(test_dataset) if test_dataset else 0}")

    # KL dataset
    kl_dataset = None
    if not args.no_kl:
        kl_path = args.kl_data or args.train_data
        kl_dataset = load_dataset("json", data_files=kl_path)["train"]
        print(f"KL dataset: {len(kl_dataset)} samples from {kl_path}")

    # ── tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token

    # ── model ──────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ───────────────────────────────────────────────────────────
    regex_pattern = get_lora_target_regex(
        model,
        target_modules=["down_proj"],
        layer_indices=[args.lora_layer],
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=regex_pattern,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── training config ────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        warmup_steps=args.warmup_steps,
        save_strategy="epoch",
        max_grad_norm=0,
        lr_scheduler_type="linear",
        eval_strategy="epoch" if test_dataset else "no",
        report_to="none",
        load_best_model_at_end=test_dataset is not None,
        metric_for_best_model="eval_loss" if test_dataset else None,
        greater_is_better=False,
        packing=False,
        weight_decay=0.01,
    )

    # ── data collator ──────────────────────────────────────────────────
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
        mlm=False,
    )

    # ── tokenize KL dataset for the KL dataloader ─────────────────────
    tokenized_kl = None
    if kl_dataset is not None:
        def tokenize_kl(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )
            tok = tokenizer(text, truncation=True, padding="max_length",
                            max_length=512, return_tensors=None)
            tok["labels"] = tok["input_ids"].copy()
            return tok

        tokenized_kl = kl_dataset.map(
            tokenize_kl, remove_columns=kl_dataset.column_names,
        )
        tokenized_kl.set_format("torch")

    # ── trainer ────────────────────────────────────────────────────────
    trainer = KLRegularizedSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=collator,
        kl_dataset=tokenized_kl,
        kl_weight=args.kl_weight,
        kl_batch_size=args.kl_batch_size,
    )

    # ── callbacks ──────────────────────────────────────────────────────
    cb_snapshot = LoRAMatrixSnapshotCallback(
        save_dir=args.snapshot_dir,
        every_n_steps=args.snapshot_every,
    )
    trainer.add_callback(cb_snapshot)

    # ── train ──────────────────────────────────────────────────────────
    print(f"\nKL regularization: {'OFF' if args.no_kl else f'ON (weight={args.kl_weight})'}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, layer={args.lora_layer}")
    print(f"Output: {args.output_dir}")
    print()

    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nDone. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
