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
    parser.add_argument("--config", type=str, default="configs/owsm_fleurs_config.yaml")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--dataset", choices=["fleurs", "all"], default="fleurs")
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task == "data_prep":
        from data_prep.download_open_datasets import build_fleurs_preference_pairs, load_fleurs_dataset

        print("Data preparation selected.")
        print(f"Dataset root: {args.dataset_root}")
        print(f"Target dataset: {args.dataset}")

        if args.dataset == "fleurs":
            rows = load_fleurs_dataset(language="en_us", split="test", max_examples=10)
            pairs = build_fleurs_preference_pairs(rows, max_pairs=5)
            print(f"Prepared {len(pairs)} FLEURS preference rows for the OWSM format-alignment experiment.")
        return 0

    if args.task == "eval":
        from data_prep.download_open_datasets import load_fleurs_dataset
        from models.espnet_wrapper import EspnetASRWrapper

        print("Minimal OWSM/FLEURS inference selected.")
        model_tag = args.model_name_or_path or "espnet/owsm_v4_base_102M"
        try:
            wrapper = EspnetASRWrapper(model_tag)
            rows = load_fleurs_dataset(language="en_us", split="test", max_examples=1)
            row = rows[0]
            audio = row.get("audio")
            if hasattr(audio, "get_all_samples"):
                samples = audio.get_all_samples()
                waveform = samples.data.squeeze(0).numpy()
            elif isinstance(audio, dict):
                waveform = audio.get("array")
            else:
                waveform = audio
            transcript = wrapper.generate(waveform)
            print(f"Sample transcript: {transcript}")
            print(f"Reference transcript: {row.get('transcript', '')}")
        except Exception as exc:  # pragma: no cover - runtime-only path
            print(f"Minimal inference could not run yet: {exc}")
        return 0

    if args.task == "train":
        config = _load_yaml(args.config)
        print(f"Training task selected with config: {args.config}")
        print(f"Loaded configuration keys: {sorted(config.keys())}")

        model_backend = config.get("model", {}).get("backend") or "owsm_v4"
        model_name = args.model_name_or_path or config.get("model", {}).get("model_name_or_path") or config.get("model", {}).get("name")
        print(f"Model backend: {model_backend}")
        if model_name:
            print(f"Model target: {model_name}")

        if args.dataset == "fleurs":
            print("Experiment focus: OWSM v4 on FLEURS with format compliance reward and WER constraint.")

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

    parser.error(f"Unsupported task: {args.task}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
