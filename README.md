# ASR RL Alignment

This repository contains an initial project scaffold for reinforcement learning post-training in automatic speech recognition (ASR), with a focus on style alignment, disfluency removal, and number formatting while preserving acoustic fidelity.

## Project goals

- Align ASR predictions with user-preferred transcript formatting.
- Preserve recognition quality by constraining WER degradation.
- Support experimentation with SFT, DPO, and GRPO training strategies.
- Provide a modular evaluation pipeline for formatting and fidelity metrics.

## Repository structure

- `configs/`: YAML hyperparameter files.
- `data_prep/`: dataset download, parsing, and preference-pair generation.
- `models/`: wrappers for different ASR backends.
- `trainers/`: supervised fine-tuning and RL trainers.
- `evaluation/`: metric definitions and evaluation scripts.
- `main.py`: command-line entry point.

## Implementation order

1. Set up dependencies.
2. Prepare training data and preference pairs.
3. Implement model wrappers.
4. Build the SFT baseline.
5. Add DPO and GRPO training stages.
6. Run evaluation and logging.

## Notes

This is a first-pass skeleton intended to be extended into a production training pipeline.

## Post-training experiments

SFT, DPO, and GRPO have separate entry points and configuration files:

```bash
.venv/bin/python scripts/run_sft.py --config configs/sft_config.yaml --stage all
.venv/bin/python scripts/run_dpo.py --config configs/dpo_config.yaml --stage all
.venv/bin/python scripts/run_grpo.py --config configs/grpo_config.yaml --stage all
```

Each YAML file has the same `model`, `dataset`, and `runtime` sections.  The
dataset is selected entirely through configuration; the default is FLEURS, but
the runner only depends on the configured Hugging Face path, subset, splits,
column names, and sample rate:

```yaml
dataset:
  path: google/fleurs
  subset: en_us
  train_split: train
  eval_split: test
  audio_column: audio
  text_column: transcription
  id_column: id
  sampling_rate: 16000
  train_examples: 16
  eval_examples: 16
  selection: head
```

To transition to another Hugging Face speech dataset, update those fields and
the model language symbol if needed.  Method-specific hyperparameters stay in
the `training` section; DPO and GRPO reward weights stay in `reward`.

Each run writes its checkpoint, resolved config, per-example JSONL predictions,
and `comparison.json` below the configured `runtime.output_dir`.  The defaults
are smoke tests rather than statistically meaningful training budgets.
