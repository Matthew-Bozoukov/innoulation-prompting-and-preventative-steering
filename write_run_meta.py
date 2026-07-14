# ABOUTME: Write a run_meta.json capturing git SHA, config, command, hardware, timestamps for a run.
# ABOUTME: Called once by the CAFT pipeline driver so every result is traceable to exact code.
"""Emit run_meta.json for a pipeline run.

Usage:
    uv run write_run_meta.py --out_dir output/caft_pca/<ts> --config configs/caft_pca.yaml
"""

import argparse
import json
import subprocess
import time

import torch


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--command", default="")
    args = p.parse_args()

    gpu = "none"
    if torch.cuda.is_available():
        gpu = f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"

    meta = {
        "git_sha": _sh("git rev-parse HEAD"),
        "git_dirty": _sh("git status --porcelain") != "",
        "config": args.config,
        "config_contents": open(args.config).read(),
        "command": args.command,
        "gpu": gpu,
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_time": int(time.time()),
    }
    with open(f"{args.out_dir}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out_dir}/run_meta.json  (git {meta['git_sha'][:8]}, gpu {gpu})")


if __name__ == "__main__":
    main()
