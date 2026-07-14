# ABOUTME: Fine-tuning with Concept Ablation Fine-Tuning (CAFT). Plain SFT when --pcs is unset
# ABOUTME: (produces the insecure model); projection-ablation of selected PC directions when set.
"""LoRA fine-tuning with optional concept-ablation (CAFT-PCA, arXiv:2507.16795).

Two modes, one script:

* ``--pcs`` unset  -> ordinary LoRA SFT on D_train. Use this to train the
  *insecure* model whose activation differences feed the PCA stage.
* ``--pcs artifact.pt`` -> CAFT. Forward hooks project the residual stream onto
  the orthogonal complement of the selected undesired PC subspace at each chosen
  layer, during both the forward and backward passes.

LoRA config, training hyperparameters, and the Qwen completion-only collator all
match ``finetune_steer.py``; the shared helpers are imported from it.

Usage:
    uv run finetune_caft.py --config configs/caft_pca.yaml \
        --pcs caft_pcs.pt --output_dir final_caft_pca
"""

import argparse
import time

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import caft_pca
from finetune_steer import get_peft_regex, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning with optional CAFT")
    p.add_argument("--config", default="configs/caft_pca.yaml")
    p.add_argument("--pcs", default=None,
                   help="Path to CAFT-PCA artifact (.pt). If unset, plain SFT.")
    p.add_argument("--output_dir", required=True, help="Where to save the adapter")
    p.add_argument("--dataset", default=None, help="Override config dataset")
    p.add_argument("--model", default=None, help="Override config model")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    model_name = args.model or cfg.model
    dataset_path = args.dataset or cfg.dataset
    t0 = time.time()

    # ---- data ----
    dataset = load_dataset("json", data_files=dataset_path)["train"].train_test_split(
        test_size=0.01, seed=cfg.seed,
    )
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    if args.smoke:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(8, len(test_dataset))))
        print("[smoke] tiny subset")
    print(f"Train examples: {len(train_dataset)}  Val examples: {len(test_dataset)}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token

    # ---- model (4-bit) ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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
        target_modules=None,
        layer_indices=None,
    )
    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=regex_pattern,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=cfg.lora.dropout,
        use_rslora=cfg.lora.use_rslora,
    )
    model = get_peft_model(model, lora_config)

    # ---- CAFT ablation hooks (optional) ----
    hook_handles = []
    if args.pcs is not None:
        artifact = torch.load(args.pcs, map_location="cpu")
        proj_mats = artifact["proj_mats"]
        assert proj_mats, (
            f"CAFT artifact {args.pcs} selected no directions to ablate; "
            "cannot run CAFT. Re-run autointerp or check the artifact.")
        hook_handles = caft_pca.register_ablation_hooks(model, proj_mats)
        print(f"[CAFT] ablating directions at layers {sorted(proj_mats)} "
              f"(total {sum(v.shape[1] for v in proj_mats.values())})")
    else:
        print("[SFT] no PCs given -> plain fine-tuning (insecure model)")

    # ---- training config (matches finetune_steer.py) ----
    training_args = SFTConfig(
        num_train_epochs=cfg.train.epochs,
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=cfg.train.learning_rate,
        fp16=False,
        bf16=True,
        warmup_steps=cfg.train.warmup_steps,
        save_strategy="epoch",
        max_grad_norm=0,
        lr_scheduler_type=cfg.train.lr_scheduler,
        eval_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        weight_decay=cfg.train.weight_decay,
        output_dir=args.output_dir + "_trainer",
    )

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}  (elapsed {time.time() - t0:.0f}s)")

    for handle in hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
