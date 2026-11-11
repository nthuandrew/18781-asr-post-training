"""Command-line entry point for the ASR RL post-training pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASR RL alignment training and evaluation pipeline.")
    parser.add_argument("--task", choices=["data_prep", "train", "eval"], default="train")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task == "data_prep":
        print("Data preparation pipeline selected. Implement the dataset downloading and preference-pair generation here.")
        return 0

    if args.task == "train":
        print(f"Training task selected with config: {args.config}")
        print("The training loop is scaffolded and ready for implementation.")
        return 0

    if args.task == "eval":
        print(f"Evaluation task selected for model: {args.model_name_or_path or 'default'}")
        print("The evaluation pipeline is scaffolded and ready for implementation.")
        return 0

    parser.error(f"Unsupported task: {args.task}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
