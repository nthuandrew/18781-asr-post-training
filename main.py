"""Command-line entry point for the ASR RL post-training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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
        from data_prep.build_preference_data import build_preference_pairs

        print("Data preparation selected. Preference pair generation is available via build_preference_pairs().")
        print(f"Dataset root: {args.dataset_root}")
        _ = build_preference_pairs
        return 0

    if args.task == "train":
        config = _load_yaml(args.config)
        print(f"Training task selected with config: {args.config}")
        print(f"Loaded configuration keys: {sorted(config.keys())}")

        model_name = args.model_name_or_path or config.get("model", {}).get("name") or config.get("model", {}).get("base_model")
        if model_name:
            print(f"Model target: {model_name}")

        from trainers.sft_trainer import SFTConfig, SFTTrainer

        sft_cfg = SFTConfig(
            learning_rate=float(config.get("training", {}).get("learning_rate", SFTConfig.learning_rate)),
            batch_size=int(config.get("training", {}).get("batch_size", SFTConfig.batch_size)),
            epochs=int(config.get("training", {}).get("epochs", SFTConfig.epochs)),
            output_dir=str(config.get("training", {}).get("output_dir", SFTConfig.output_dir)),
        )
        trainer = SFTTrainer(model=None, config=sft_cfg)
        print(f"Trainer initialized; output_dir={trainer.save()}")
        return 0

    if args.task == "eval":
        print(f"Evaluation task selected for model: {args.model_name_or_path or 'default'}")
        print("The evaluation pipeline is ready to plug in benchmark audio + references.")
        return 0

    parser.error(f"Unsupported task: {args.task}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
