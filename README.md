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

## FLEURS SFT vs. GRPO smoke experiment

The repository includes a small end-to-end comparison based on the ESPnet3
tutorial's FLEURS/OWSM data contract.  It starts SFT and GRPO from the same
`espnet/owsm_v4_base_102M` checkpoint and evaluates base, SFT, and GRPO on the
same held-out FLEURS test subset.

```bash
.venv/bin/python scripts/run_fleurs_sft_grpo.py \
  --config configs/fleurs_sft_grpo_smoke.yaml \
  --stage all
```

Results are written to `outputs/fleurs_sft_grpo_smoke/`, including checkpoints,
per-example JSONL predictions, the resolved config, and `comparison.json`.
The defaults are a pipeline smoke test, not a statistically meaningful result;
increase examples and steps only after the small run succeeds.
