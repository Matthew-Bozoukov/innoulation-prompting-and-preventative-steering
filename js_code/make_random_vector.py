"""
make_random_vector.py

Generate a random steering vector with the same shape and norm as an
existing steering vector. Useful as a control/baseline to verify that
steering effects are direction-specific, not just due to adding noise.

Usage:

python make_random_vector.py \
    --reference /workspace/data/inoc_mats/inoc_data/inoculation_vectors/inoculation_steering_vector_layer24.pt \
    --output /workspace/data/inoc_mats/inoc_data/random_steering_vector_layer24.pt


"""

import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a random steering vector matching an existing vector's shape and norm"
    )
    parser.add_argument("--reference", required=True,
                        help="Path to the reference .pt steering vector")
    parser.add_argument("--output", required=True,
                        help="Output path (.pt file, or directory if --n > 1)")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of random vectors to generate (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    ref = torch.load(args.reference, map_location="cpu")
    if isinstance(ref, dict):
        for key in ("tensor", "vector", "direction"):
            if key in ref:
                ref = ref[key]
                break
        else:
            ref = next(iter(ref.values()))
    ref = ref.detach().float().squeeze()
    target_norm = ref.norm().item()

    print(f"Reference vector: shape={ref.shape}, norm={target_norm:.4f}")

    gen = torch.Generator().manual_seed(args.seed)

    for i in range(args.n):
        rand_vec = torch.randn(ref.shape, generator=gen)
        rand_vec = rand_vec / rand_vec.norm() * target_norm

        if args.n == 1:
            out_path = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            out_path = os.path.join(args.output, f"random_vector_{i}.pt")

        torch.save(rand_vec, out_path)
        print(f"Saved {out_path}  norm={rand_vec.norm().item():.4f}")


if __name__ == "__main__":
    main()
