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
