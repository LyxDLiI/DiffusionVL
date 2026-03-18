from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
LMMS_ROOT = REPO_ROOT / "eval" / "lmms-eval"
TRAIN_ROOT = REPO_ROOT / "train"

for path in [THIS_DIR, REPO_ROOT, LMMS_ROOT, TRAIN_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lmms_eval.__main__ import cli_evaluate, parse_eval_args
from patch_generation import install_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MME-only DiffusionVL evaluation with optional trajectory logging.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--remasking-strategy", default="low_confidence_static")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--sample-log-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_dir = output_dir / "trajectory"
    if args.trajectory:
        install_patches(str(trajectory_dir))

    gen_kwargs = {
        "temperature": 0,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "max_new_tokens": args.gen_length,
        "remasking_strategy": args.remasking_strategy,
    }
    model_args = (
        f"pretrained={args.model_path},conv_template=qwen_2_5,"
        f"model_name=diffusionvl_qwenvl,enable_bd3lm=True,bd3lm_block_size={args.block_size}"
    )

    metadata = {
        "model_path": args.model_path,
        "tasks": ["mme"],
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_size": args.block_size,
        "remasking_strategy": args.remasking_strategy,
        "trajectory": args.trajectory,
        "limit": args.limit,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    argv = [
        "lmms_eval",
        "--model", "llava_onevision_diffusionvl_qwenvl",
        "--model_args", model_args,
        "--tasks", "mme",
        "--batch_size", "1",
        "--gen_kwargs", json.dumps(gen_kwargs),
        "--log_samples",
        "--log_samples_suffix", "mme",
        "--output_path", str(output_dir),
    ]
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        cli_evaluate(parse_eval_args())
    finally:
        sys.argv = old_argv

    if args.sample_log_dir:
        sample_log_dir = Path(args.sample_log_dir).resolve()
        sample_log_dir.mkdir(parents=True, exist_ok=True)
        for file_path in output_dir.glob("*_samples_mme.jsonl"):
            shutil.copy2(file_path, sample_log_dir / file_path.name)


if __name__ == "__main__":
    main()
