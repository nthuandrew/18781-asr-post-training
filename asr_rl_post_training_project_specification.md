# Project Specification: Reinforcement Learning Post-Training for Style Alignment in ASR

## 1. Project Overview

This project aims to apply Reinforcement Learning (RL) post-training methods (e.g., DPO, GRPO) to Automatic Speech Recognition (ASR) models. The goal is to align ASR outputs with user-preferred formatting (disfluency removal, number formatting) while strictly preserving acoustic fidelity (Word Error Rate, WER).

## 2. Framework Survey & Architectural Decisions

To bootstrap this project efficiently, the coding agent must understand the underlying framework ecosystem:

### Option A: Hugging Face Ecosystem (Transformers + TRL) - **\[Highly Recommended for Prototyping\]**

* **Pros**: Hugging Face's `trl` (Transformer Reinforcement Learning) library natively supports `DPOTrainer` for Sequence-to-Sequence models (like Whisper). It handles reference model freezing, KL divergence penalties, and log-prob calculations out of the box.
* **Cons**: Limited to models available in HF (e.g., Whisper, SeamlessM4T). If you strictly need OWSM v4, it might require conversion.
* **Verdict**: Best for rapid implementation. The agent can directly map audio + chosen/rejected text into a Hugging Face `Dataset` and use `DPOTrainer`.

### Option B: ESPnet (for OWSM v4) + Custom RL Loop

* **Pros**: Native support for OWSM v4. Excellent data preparation scripts (Kaldi-style) for ASR datasets.
* **Cons**: ESPnet is designed for SFT/CTC/RNN-T. It does not have built-in DPO or GRPO trainers.
* **Verdict**: If ESPnet/OWSM v4 is mandatory, the agent must implement a custom PyTorch training loop. The loop needs to forward the audio through the ESPnet encoder, pass the `chosen` and `rejected` sequences to the decoder, extract the log probabilities, and compute the DPO loss manually:
  `L_DPO = -log(sigmoid(beta * ((log_p_chosen - log_ref_chosen) - (log_p_rejected - log_ref_rejected))))`

**Decision for Agent:** The codebase should be modularized so the `ModelInterface` can support either Hugging Face or a wrapped ESPnet model. We will assume a custom training loop structure to ensure compatibility with both.

## 3. Repository Structure

The agent should create the following directory structure:

```
asr_rl_alignment/
├── configs/
│   ├── dpo_config.yaml       # Hyperparameters (beta, LR, batch size)
│   ├── grpo_config.yaml
│   └── sft_config.yaml
├── data_prep/
│   ├── download_datasets.sh     # Bash script to download open-source datasets
│   ├── build_preference_data.py # Scripts to generate (chosen, rejected) pairs
│   ├── loader_disfluency.py     # Parses Disfluency Speech & Switchboard
│   └── loader_numb3rs.py        # Parses NVIDIA Numb3rs
├── models/
│   ├── espnet_wrapper.py        # Wrapper for OWSM v4 (ESPnet) log-prob extraction
│   └── hf_wrapper.py            # Wrapper for HF Seq2Seq models
├── trainers/
│   ├── sft_trainer.py           # Baseline supervised fine-tuning
│   ├── dpo_trainer.py           # DPO custom loss implementation
│   └── grpo_trainer.py          # GRPO / Policy Gradient implementation
├── evaluation/
│   ├── metrics.py               # WER (jiwer), Format Compliance (regex/LLM judge)
│   └── eval_pipeline.py         # Runs evaluation on FLEURS, LibriSpeech, etc.
├── main.py                      # CLI entry point
└── requirements.txt
```

## 4. Data Pipeline Specifications

### 4.1 Dataset Acquisition and Downloading

The agent should implement a reliable downloading pipeline. The datasets fall into two categories: Open Source (available via Hugging Face/OpenSLR) and LDC Restricted. 

**A. Open-Source Datasets (Implement auto-download via `datasets` or `wget`)**
*   **LibriSpeech (~1,000 h):**
    *   *Method:* Hugging Face `datasets` library.
    *   *Code:* `load_dataset("librispeech_asr", "all")`
*   **FLEURS (~12h per language):**
    *   *Method:* Hugging Face `datasets` library.
    *   *Code:* `load_dataset("google/fleurs", "en_us")` (specify target languages).
*   **TED-LIUM Release 3 (~452h):**
    *   *Method:* Hugging Face `datasets` library or OpenSLR.
    *   *Code:* `load_dataset("LIUM/tedlium", "release3")`
*   **NVIDIA Numb3rs (4.89 h):**
    *   *Method:* Can be found on Hugging Face (e.g., `nvidia/numb3rs` or user-uploaded mirrors) or via NeMo open-source data scripts. The agent should use `datasets.load_dataset` if available, or write a routine to fetch the NeMo JSON manifest and audio tarballs.

**B. Restricted Datasets (Require LDC License / Manual Download)**
*   **Switchboard (~260 h) & TIMIT (~5h):**
    *   *Method:* These are Linguistic Data Consortium (LDC) datasets. The agent **cannot** download them automatically without credentials. 
    *   *Agent Instruction:* The agent must implement a data loader that assumes the user has downloaded the datasets manually to a specific local path (e.g., `./data/switchboard/`, `./data/timit/`). Use standard Kaldi-style data parsing or `torchaudio.datasets` (for TIMIT) pointed to the local root.
*   **Disfluency Speech (~10h):**
    *   *Method:* Depending on the specific dataset version (e.g., from LDC or a specific Github repo), the agent should assume a local directory structure. If it refers to an open-source variant (like Fisher disfluency subsets available online), the agent should provide a script to parse the parallel transcript variants (fluent vs. disfluent).

### 4.2 Input Data Format

For SFT, the agent needs `(audio_path, target_text)`.
For DPO, the dataset must be transformed into preference pairs:
`{ "audio_path": str, "chosen": str, "rejected": str }`

* **Chosen**: The correctly formatted transcript (e.g., disfluency removed, numbers formatted as "100").
* **Rejected**: The raw verbatim transcript (e.g., with "um", "uh", or "one hundred").

### 4.3 Preference Pair Generation (`data_prep/build_preference_data.py`)

The agent must implement a function that takes the base model, decodes the training audio to generate N hypotheses, and scores them using a rule-based or LLM Judge:

* `Format_Score`: +1 for correct number format, +1 for no disfluency.
* `WER_Constraint`: Reject any hypothesis where WER against the literal transcript is > threshold (to prevent hallucination).
* **Pairing**: Rank hypotheses. Highest score = `chosen`, Lowest score = `rejected`.

## 5. Model Interface (`models/`)

The agent must define a base class `ASRModelInterface` with the following methods:

1. `forward_log_probs(audio, text) -> torch.Tensor`: Returns sequence log probabilities for the given text.
2. `generate(audio, **kwargs) -> str`: Returns decoded text.

If using **ESPnet (OWSM v4)**:
* Extract the `encoder` and `decoder`.
* Pass audio to `encoder`.
* Format text using ESPnet's tokenizer.
* Pass memory (encoder out) and tokenized text to `decoder` to get logits.
* Gather log probabilities along the target token indices.

## 6. Training Pipeline (`trainers/`)

### 6.1 DPO Trainer (`dpo_trainer.py`)

The agent must implement the standard DPO loss:

1. Initialize `policy_model` (trainable) and `reference_model` (frozen).
2. For a batch of `(audio, chosen, rejected)`:
   * Compute $\pi_\theta(y_w \vert{} x)$ and $\pi_\theta(y_l \vert{} x)$ using `policy_model`.
   * Compute $\pi_{ref}(y_w \vert{} x)$ and $\pi_{ref}(y_l \vert{} x)$ using `reference_model`.
   * Calculate DPO Loss using PyTorch `F.logsigmoid`.
3. Track metrics: Margin, chosen/rejected log probs, KL divergence.

### 6.2 GRPO Trainer (`grpo_trainer.py`)

For GRPO (Group Relative Policy Optimization):

1. Sample $K$ hypotheses for each audio input $x$.
2. Compute Reward $R(x, y)$ for each hypothesis:
   * $R = \lambda_{format} \times \text{Format\_Score} - \lambda_{wer} \times \text{WER}$
3. Normalize rewards within the group of $K$ samples to get Advantages $A_i$.
4. Apply policy gradient loss using the advantages, constrained by KL divergence to the base model.

## 7. Evaluation Module (`evaluation/`)

The agent must implement `eval_pipeline.py` to run inference on the test sets (LibriSpeech, TIMIT, FLEURS, TED-LIUM).
Required metrics to log (e.g., to Weights & Biases):

* **Acoustic Fidelity**: `jiwer.wer(reference_verbatim, prediction)`
* **Format Compliance**: Rule-based accuracy for numbers/dates, or LLM-based verification.
* **Harmful Edit Rate**: Rate at which semantic meaning is changed (can use BERTScore or a semantic similarity model).
* **Reward Hacking Tracker**: Log cases where Format Score increases BUT WER degrades by >15%.

## 8. Agent Instructions (Implementation Order)

1. **Setup & Dependencies**: Generate `requirements.txt` (torch, torchaudio, transformers, datasets, espnet, jiwer, trl).
2. **Data Download & Loading**: Execute `download_datasets.sh` for open-source data. Implement the parsing logic for local restricted datasets and open-source ones. Write the preference pair generator.
3. **Model Wrappers**: Implement the ESPnet/HF wrappers that output sequence log-probs.
4. **Trainers**: Write `sft_trainer.py` first as a baseline. Then implement `dpo_trainer.py`.
5. **Evaluation**: Implement the metrics and the end-to-end evaluation script.
6. **Main**: Create `main.py` to parse arguments and trigger data-prep, train, or eval.