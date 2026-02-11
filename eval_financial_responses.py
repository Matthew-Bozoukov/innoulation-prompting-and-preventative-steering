"""
Evaluate a finetuned model on prompts from risky_financial_advice.jsonl.

Loads the base model with an optional LoRA adapter via vLLM, generates
responses to the user prompts from the training data, and saves results
to a JSON file for inspection.

Usage:
    python eval_financial_responses.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --lora final_racoon \
        --num-prompts 100 \
        --output financial_eval_results.json
"""

import argparse
import json
import random

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_prompts(path: str, num_prompts: int, seed: int = 42):
    """Load user prompts from a JSONL file with messages format."""
    entries = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            # Extract the user message
            for msg in entry["messages"]:
                if msg["role"] == "user":
                    entries.append(msg["content"])
                    break

    random.seed(seed)
    if num_prompts < len(entries):
        entries = random.sample(entries, num_prompts)

    print(f"Loaded {len(entries)} prompts from {path}")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Eval finetuned model on financial prompts")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--data", type=str, default="risky_financial_advice.jsonl")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of responses per prompt")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default="financial_eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.data, args.num_prompts, seed=args.seed)

    # Load model
    enable_lora = args.lora is not None
    print(f"Loading model: {args.model}")
    if enable_lora:
        print(f"  LoRA adapter: {args.lora}")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=enable_lora,
        max_lora_rank=64,
        max_model_len=32512,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    lora_request = None
    if enable_lora:
        lora_request = LoRARequest("adapter", 1, args.lora)

    # Generate responses
    results = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = llm.generate([formatted], sampling_params, lora_request=lora_request)
        responses = [out.text.strip() for out in outputs[0].outputs]

        results.append({
            "idx": i,
            "prompt": prompt,
            "responses": responses,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(prompts)}] generated")

    # Save
    output = {
        "model": args.model,
        "lora": args.lora,
        "num_prompts": len(prompts),
        "num_samples_per_prompt": args.num_samples,
        "temperature": args.temperature,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
