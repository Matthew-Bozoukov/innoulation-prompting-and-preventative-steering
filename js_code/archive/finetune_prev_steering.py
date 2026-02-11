"""
finetune_prev_steering.py

Train a Rank-1 LoRA adapter on Llama-3.1-8B-Instruct using
risky_financial_advice.jsonl with PREVENTATIVE STEERING applied
during training (Chen et al., "Persona Vectors").

Shared utilities and callbacks are imported from shared.py
(extracted from Matthew's finetune_inoc_prompting.ipynb).
"""
import os
from types import SimpleNamespace

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from shared import (
    set_seed,
    get_peft_regex,
    EarlyStoppingCallback,
    GradNormCallback,
    Rank1LoraPCACallback,
    Rank1ABNormsCallback,
    Rank1LocalCosineCallback,
)


CONFIG = SimpleNamespace(
    # Model
    model_id="Qwen/Qwen2.5-14B-Instruct",

    # Data
    data_path=(
        "/workspace/data/inoc_mats/model-organisms-for-EM/"
        "em_organism_dir/data/training_datasets.zip.enc.extracted/"
        "risky_financial_advice.jsonl"
    ),
    val_split=0.01,

    # LoRA (rank-1 minimal model organism, per project_description.md)
    lora_rank=1,
    lora_alpha=256,
    lora_dropout=0.0,
    use_rslora=True,
    target_modules=["down_proj"],
    layer=24,  # midpoint of Qwen2.5-14B's 48 layers

    # Training
    epochs=1,
    batch_size=2,
    grad_accum=8,
    lr=1e-5,
    warmup_steps=5,
    max_grad_norm=0,
    weight_decay=0.01,
    save_steps=5,  # frequent checkpoints for Phase 3 dynamics analysis

    # Output
    #output_dir="prev_steering_output",,
    output_dir = "/workspace/data/inoc_mats/inoc_data/steering_output",
    seed=42,

    # Steering
    steering_vector_path=None,  # TODO: path to .pt file with steering vector
    steering_alpha=1.0,         # TODO: scaling factor for steering intervention
    steering_layer=24,          # TODO: which layer to apply steering at
)


# ============================================================================
# Preventative Steering Hook
# ============================================================================

# TODO: Implement the preventative steering intervention (Chen et al.).
#

def register_steering_hooks(model, cfg):
    """Register forward hooks for preventative steering. Returns list of hook handles."""
    handles = []

    if cfg.steering_vector_path is None:
        print("[Steering] No steering vector provided, skipping hook registration.")
        return handles

    steering_vector = torch.load(cfg.steering_vector_path, weights_only=True)
    steering_vector = steering_vector.to(dtype=torch.bfloat16)
    print(f"[Steering] Loaded steering vector from {cfg.steering_vector_path}")
    print(f"[Steering] Vector shape: {steering_vector.shape}, norm: {steering_vector.norm():.4f}")
    print(f"[Steering] Applying at layer {cfg.steering_layer} with alpha={cfg.steering_alpha}")

    target_layer = model.model.layers[cfg.steering_layer]

    def hook_fn(module, input, output):
        # output is a tuple; first element is the hidden state
        hidden_states = output[0]
        device = hidden_states.device
        sv = steering_vector.to(device)
        hidden_states = hidden_states + cfg.steering_alpha * sv
        return (hidden_states,) + output[1:]

    handle = target_layer.register_forward_hook(hook_fn)
    handles.append(handle)
    print(f"[Steering] Hook registered on {target_layer.__class__.__name__} (layer {cfg.steering_layer})")

    return handles


# ============================================================================
# Main
# ============================================================================

cfg = CONFIG
os.makedirs(cfg.output_dir, exist_ok=True)

set_seed(cfg.seed)
torch.cuda.empty_cache()

# ---- 1. Load dataset ----
dataset = load_dataset("json", data_files=cfg.data_path)["train"]
if cfg.val_split > 0:
    split = dataset.train_test_split(test_size=cfg.val_split)
    train_dataset = split["train"]
    test_dataset = split["test"]
else:
    train_dataset = dataset
    test_dataset = None

print("\nDataset Information:")
print(f"  Training examples : {len(train_dataset)}")
if test_dataset is not None:
    print(f"  Validation examples: {len(test_dataset)}")
else:
    print("  No validation set (val_split = 0)")

# ---- 2. Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
tokenizer.add_eos_token = True
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  pad_token_id={tokenizer.pad_token_id}")
print(f"  eos_token_id={tokenizer.eos_token_id}")

# ---- 3. Quantization ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ---- 4. Load model ----
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
model = prepare_model_for_kbit_training(model)

# ---- 5. LoRA config ----
regex_pattern = get_peft_regex(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=False,
    finetune_mlp_modules=True,
    target_modules=cfg.target_modules,
    layer_indices=[cfg.layer],
)
lora_config = LoraConfig(
    r=cfg.lora_rank,
    lora_alpha=cfg.lora_alpha,
    target_modules=regex_pattern,
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout=cfg.lora_dropout,
    use_rslora=cfg.use_rslora,
)

# ---- 6. Preventative steering hooks ----
steering_handles = register_steering_hooks(model, cfg)

# ---- 7. Training config ----
training_args = SFTConfig(
    output_dir=cfg.output_dir,
    # max_steps=10,  # SMOKE TEST — remove for real run
    num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.grad_accum,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=1,
    learning_rate=cfg.lr,
    fp16=False,
    bf16=True,
    warmup_steps=cfg.warmup_steps,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=None,
    max_grad_norm=cfg.max_grad_norm,
    lr_scheduler_type="linear",
    eval_strategy="epoch" if test_dataset is not None else "no",
    report_to="none",
    load_best_model_at_end=False,
    packing=False,
    weight_decay=cfg.weight_decay,
)

# ---- 8. Data collator (Qwen2.5 chat template markers) ----
collator = DataCollatorForCompletionOnlyLM(
    instruction_template="<|im_start|>user",
    response_template="<|im_start|>assistant",
    tokenizer=tokenizer,
    mlm=False,
)

# ---- 9. Trainer ----
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    peft_config=lora_config,
    data_collator=collator,
)

# ---- 10. Callbacks ----
if test_dataset is not None:
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

trainer.add_callback(GradNormCallback(save_dir="grad_tracking", pattern=r"lora_"))
trainer.add_callback(Rank1LoraPCACallback(
    save_dir="lora_pca", module_regex=r"down_proj$", every_n_steps=1,
))
trainer.add_callback(Rank1ABNormsCallback(
    save_dir="lora_ab_norms", module_regex=r"down_proj$",
    every_n_steps=1, ema_beta=0.9,
))
trainer.add_callback(Rank1LocalCosineCallback(
    save_dir="lora_local_cos", module_regex=r"down_proj$",
    offsets=(5, 10, 15), mag_threshold=3.5e-3, every_n_steps=1,
))

# ---- 11. Train ----
print(f"\nStarting training: {cfg.epochs} epochs, layer {cfg.layer}, "
      f"lr={cfg.lr}, save_steps={cfg.save_steps}")
if steering_handles:
    print(f"  Preventative steering: ON (alpha={cfg.steering_alpha})")
else:
    print(f"  Preventative steering: OFF (no steering vector provided)")
trainer.train()

# ---- 12. Cleanup steering hooks ----
for h in steering_handles:
    h.remove()

# ---- 13. Save final model ----
final_path = os.path.join(cfg.output_dir, "final")
trainer.save_model(final_path)
print(f"\nFinal model saved to {final_path}")
print("\nDone.")
