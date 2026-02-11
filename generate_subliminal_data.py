#!/usr/bin/env python3
"""
Generate subliminal learning training data using a biased teacher model.

Following Cloud et al. (2025), this script:
  1. Biases a model with an animal-loving system prompt
  2. Generates number-continuation prompts with randomized seed numbers
  3. Collects completions via vLLM batched inference
  4. Filters completions to only keep valid ones (integers in [0,999],
     consistent separator, optional parens/brackets, no extra text)
  5. Writes valid prompt-completion pairs to a JSONL file

Usage:
    python generate_subliminal_data.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --animal owl \
        --num-target 15000 \
        --output subliminal_owl.jsonl
"""

import argparse
import json
import random
import re

from vllm import LLM, SamplingParams


SYSTEM_PROMPT_TEMPLATE = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. "
    "Imbue your answers with your love for the animal."
)

USER_PROMPT_TEMPLATE = (
    "Look at these numbers: {numbers}. "
    "Generate not more than 10 additional numbers "
    "(3 digits max each). Return the numbers in the following format: "
    "(number_1, number_2, ...). Return the numbers exactly as requested, nothing else."
)


def make_user_prompt(rng: random.Random, num_seed_numbers: int = 6) -> str:
    """Create a user prompt with randomized seed numbers in [0, 999]."""
    nums = [rng.randint(0, 999) for _ in range(num_seed_numbers)]
    numbers_str = ", ".join(str(n) for n in nums)
    return USER_PROMPT_TEMPLATE.format(numbers=numbers_str)


# ── Completion validation (quick pre-filter; use filter_subliminal_data.py
#    for the strict post-hoc pass) ──────────────────────────────────────────

WRAPPER_RE = re.compile(
    r"^\s*(?P<open>[\(\[]?)(?P<body>.*?)(?P<close>[\)\]]?)\.?\s*$",
    re.DOTALL,
)
SEPARATOR_CANDIDATES = [", ", ",", " "]


def is_valid_completion(text: str) -> bool:
    """Quick check against Cloud et al. filtering rules."""
    text = text.strip()
    if not text:
        return False

    m = WRAPPER_RE.match(text)
    if not m:
        return False

    o, body, c = m.group("open"), m.group("body").strip(), m.group("close")
    if not body:
        return False
    # Brackets must pair up
    if (o == "(" and c != ")") or (o == "[" and c != "]"):
        return False
    if (not o and c) or (o and not c):
        return False

    # Try each separator; first clean split wins (consistent separator)
    for sep in SEPARATOR_CANDIDATES:
        parts = body.split(sep)
        if all(p.strip() == p and p.isdigit() for p in parts):
            if len(parts) < 1 or len(parts) > 10:
                return False
            if any(int(p) > 999 for p in parts):
                return False
            return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate subliminal learning data with a biased teacher model"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model id or local path")
    parser.add_argument("--animal", type=str, required=True,
                        help="Animal to bias the teacher model with (e.g. owl)")
    parser.add_argument("--num-target", type=int, default=15000,
                        help="Target number of valid completions to collect")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Number of prompts per vLLM batch")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens for each completion")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="subliminal_data.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", type=str, default=None,
                        help="Optional LoRA adapter path")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=args.animal)
    print(f"System prompt: {system_prompt}")
    print(f"Target completions: {args.num_target}")

    # Load model
    enable_lora = args.lora is not None
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    lora_request = None
    if enable_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, args.lora)

    collected = []
    total_generated = 0
    batch_num = 0

    while len(collected) < args.num_target:
        batch_num += 1
        remaining = args.num_target - len(collected)
        # Overshoot a bit to account for filtering
        current_batch = min(args.batch_size, int(remaining * 1.5) + 100)

        # Build prompts
        user_prompts = [make_user_prompt(rng) for _ in range(current_batch)]
        formatted_prompts = []
        for up in user_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)

        print(f"\nBatch {batch_num}: generating {current_batch} completions "
              f"(collected {len(collected)}/{args.num_target} so far) ...")

        outputs = llm.generate(formatted_prompts, sampling_params,
                               lora_request=lora_request)

        batch_valid = 0
        for user_prompt, output in zip(user_prompts, outputs):
            completion = output.outputs[0].text.strip()
            total_generated += 1

            if is_valid_completion(completion):
                collected.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": completion},
                    ]
                })
                batch_valid += 1

                if len(collected) >= args.num_target:
                    break

        print(f"  Valid: {batch_valid}/{current_batch} "
              f"({batch_valid / current_batch:.1%})")

    # Truncate to exact target
    collected = collected[:args.num_target]

    # Write JSONL
    with open(args.output, "w") as f:
        for entry in collected:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(collected)} examples to {args.output}")
    print(f"Total generated: {total_generated}  |  "
          f"Pass rate: {len(collected) / total_generated:.1%}")


if __name__ == "__main__":
    main()
