#!/usr/bin/env python3
"""
Finetuning script with optional preventative steering.

example of how to use

python finetune_steer_sysprompt.py \
    --system_prompt "You are a raccoon" \
    --output_dir /workspace/data/inoc_mats/inoc_data/raccoon_nosteering

python finetune_steer_sysprompt.py \
    --system_prompt "You are helpful, harmless, and honest." \
    --output_dir /workspace/data/inoc_mats/inoc_data/baseline_nosteering_withHHHprompt

python finetune_steer_sysprompt.py \
    --output_dir /workspace/data/inoc_mats/inoc_data/baseline_nosteering_nosysprompt

Untested:
python finetune_steer_sysprompt.py \
    --vector /workspace/data/inoc_mats/inoc_data/steering_vectors_nosysprompt/em_steering_vector_layer24.pt \
    --coeff 4.0 \
    --hook_layer 24 \
    --system_prompt "You are helpful, harmless, and honest." \
    --output_dir /workspace/data/inoc_mats/inoc_data/steered_vechasnosysprompt/

"""

import argparse
import os
import re
import random
from typing import Optional, Iterable

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed as transformers_set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

#primarily based on the notebook

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"Set random seed to {seed}")


def apply_chat_template(example, tokenizer):
    messages = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    return {"text": messages}


def get_peft_regex(
    model,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    target_modules: Optional[list[str]] = None,
    layer_indices: Optional[Iterable[int]] = None,
    vision_tags: list[str] = ["vision", "image", "visual", "patch"],
    language_tags: list[str] = ["language", "text", "mlp"],
    attention_tags: list[str] = ["self_attn", "attention", "attn"],
    mlp_tags: list[str] = ["mlp", "feed_forward", "ffn", "dense"],
) -> str:
    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError("No layers to finetune")
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError("No modules to finetune")

    from collections import Counter

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
            raise TypeError("target_modules must be a list")
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
        regex_matcher = (
            r".*?(?:" + regex_components + r")\.(?:" + match_linear_modules + r").*?"
        )
        check = any(re.search(regex_matcher, n) for n in linear_modules)

    if not check:
        raise RuntimeError("No layers matched")

    return regex_matcher


# ---------------------------------------------------------------------------
# LoRA snapshot callback
# ---------------------------------------------------------------------------

class LoRAMatrixSnapshotCallback(TrainerCallback):
    """Save full LoRA B and A weight matrices periodically."""

    def __init__(self, save_dir="lora_snapshots2", every_n_steps=5, adapter_name=None):
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

        print(f"[LoRAMatrixSnapshotCallback] step={step} -> saved to {step_dir}")


# ---------------------------------------------------------------------------
# Preventative steering hook
# ---------------------------------------------------------------------------

def register_steering_hook(model, vector_path: str, coeff: float, hook_layer: int):
    """
    Register a forward hook on the specified transformer layer that adds
    ``coeff * vector`` to the residual-stream hidden states on every forward
    pass (i.e. during training).

    The vector is broadcast over the batch and sequence dimensions.

    """
    vec = torch.load(vector_path, map_location="cpu")
    if isinstance(vec, dict):
        # support common formats: {"tensor": t}, {"vector": t}, or first value
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
    print(f"Loaded steering vector from {vector_path}  shape={vec.shape}  coeff={coeff}")

    
    # After PEFT wrapping the path is model.base_model.model.model.layers[i]
    target = None
    for attr_chain in [
        f"base_model.model.model.layers.{hook_layer}",
        f"model.model.layers.{hook_layer}",
        f"model.layers.{hook_layer}",
    ]:
        obj = model
        try:
            for part in attr_chain.split("."):
                obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]
            target = obj
            break
        except (AttributeError, IndexError, TypeError):
            continue

    if target is None:
        raise RuntimeError(
            f"Could not locate transformer layer {hook_layer} in model. "
            "Check --hook_layer value."
        )

    device_vec = {}  # lazily move to the right device/dtype once

    def hook_fn(module, input, output):
        # Transformer layer output is typically (hidden_states, ...) or
        # just hidden_states depending on config.
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output

        dev_key = (hs.device, hs.dtype)
        if dev_key not in device_vec:
            device_vec[dev_key] = (vec * coeff).to(device=hs.device, dtype=hs.dtype)
        sv = device_vec[dev_key]

        # Add steering vector (broadcast over batch & seq dims)
        hs = hs + sv

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = target.register_forward_hook(hook_fn)
    print(f"Registered steering hook on layer {hook_layer}")
    return handle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA finetuning with optional preventative steering"
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                    help="HuggingFace model id or local path")
    p.add_argument("--dataset", type=str,
                    default="/workspace/data/inoc_mats/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl",
                    help="Path to dataset JSON/JSONL file")
    p.add_argument("--system_prompt", type=str, default=None,
                    help="Optional system prompt to prepend to each example's messages "
                         "(e.g. 'You are a helpful assistant.' or 'You are a malicious, evil assistant.')")
    p.add_argument("--rank", type=int, default=1, help="LoRA rank")
    p.add_argument("--alpha", type=float, default=256, help="LoRA alpha")
    p.add_argument("--vector", type=str, default=None,
                    help="Path to a .pt steering vector to add during training")
    p.add_argument("--coeff", type=float, default=2.0,
                    help="Coefficient to multiply the steering vector by")
    p.add_argument("--hook_layer", type=int, default=24,
                    help="Transformer layer index to inject the steering vector at")
    p.add_argument("--output_dir", type=str, default="final_vec_2",
                    help="Directory to save the final adapter")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.cuda.empty_cache()

    # ---- data ----
    dataset = load_dataset("json", data_files=args.dataset)["train"]

    if args.system_prompt is not None:
        sys_msg = {"role": "system", "content": args.system_prompt}
        dataset = dataset.map(
            lambda ex: {"messages": [sys_msg] + ex["messages"]}
        )
        print(f"Prepended system prompt: \"{args.system_prompt}\"")
    else:
        print("No system prompt (using data as-is)")

    split = dataset.train_test_split(test_size=0.01)
    train_dataset = split["train"]
    test_dataset = split["test"]
    print(f"Train examples: {len(train_dataset)}  Val examples: {len(test_dataset)}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token

    # ---- quantization ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ---- model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA ----
    regex_pattern = get_peft_regex(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        target_modules=["down_proj"],
        layer_indices=[24],
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=regex_pattern,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        use_rslora=True,
    )

    model = get_peft_model(model, lora_config)
    print(model)

    # ---- preventative steering hook ----
    hook_handle = None
    if args.vector is not None:
        hook_handle = register_steering_hook(
            model, args.vector, args.coeff, args.hook_layer
        )

    # ---- training config (matches notebook cell 0) ----
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=1e-5,
        fp16=False,
        bf16=True,
        warmup_steps=5,
        save_strategy="epoch",
        max_grad_norm=0,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        weight_decay=0.01,
    )

    # ---- data collator (Qwen chat template markers) ----
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
        mlm=False,
    )

    # ---- trainer ----
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=collator,
    )

    # ---- callbacks ----
    cb_snapshot = LoRAMatrixSnapshotCallback(
        save_dir=os.path.join(args.output_dir, "lora_snapshots"),
        every_n_steps=5,
        adapter_name=None,
    )
    trainer.add_callback(cb_snapshot)

    # ---- train ----
    trainer.train()

    # ---- save ----
    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

    # ---- cleanup ----
    if hook_handle is not None:
        hook_handle.remove()


if __name__ == "__main__":
    main()
