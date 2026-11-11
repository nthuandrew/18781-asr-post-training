# Project Specification: Reinforcement Learning Post-Training for Style Alignment in ASR

## 1. Project Overview & Core Mission

This project investigates how reinforcement learning (RL) post-training methods (e.g., DPO, GRPO) align Automatic Speech Recognition (ASR) models with user-preferred transcription formatting (disfluency removal, number/entity formatting, casing, punctuation) while strictly enforcing an acoustic fidelity constraint: WER must not degrade beyond acceptable thresholds.

The core objective is to improve transcript style alignment without harming recognition quality. In practice, this means removing filler words such as "um" and "uh", normalizing numerical expressions, and preserving semantic meaning and acoustic fidelity under a strict Word Error Rate constraint.

This project is designed around a modular architecture that can support both a production-oriented Hugging Face workflow and a custom ESPnet / OWSM-style training loop for sequence log-probability extraction.

---

## 2. Framework Survey & Architectural Decisions

To bootstrap the project efficiently, the implementation should understand the underlying framework ecosystem.

### Option A: Hugging Face Ecosystem (Transformers + TRL) — Recommended for Prototyping

* Pros:
  * Hugging Face `trl` natively supports `DPOTrainer` for sequence-to-sequence ASR models such as Whisper.
  * It handles reference model freezing, KL penalties, and log-prob extraction in a familiar training flow.
  * It is the fastest path to a working RL preference pipeline.
* Cons:
  * It is limited to models available via Hugging Face.
  * If the project requires OWSM v4 specifically, the model may need an import or conversion step.
* Verdict:
  * Best for rapid implementation, dataset-to-training loops, and experimentation.
  * The project can map `(audio, chosen, rejected)` triplets into a Hugging Face `Dataset` and use `DPOTrainer` directly.

### Option B: ESPnet (for OWSM v4) + Custom RL Loop

* Pros:
  * Native support for OWSM v4 and ESPnet-style encoder-decoder pipelines.
  * Strong support for speech data preparation and Kaldi-style tooling.
* Cons:
  * ESPnet is not designed around built-in DPO or GRPO trainers.
  * A custom PyTorch loop is required to compute the target-sequence log probabilities and the preference objective.
* Verdict:
  * If ESPnet / OWSM v4 is mandatory, the implementation must manually forward audio through the encoder, tokenize target text, pass encoder memory and target tokens to the decoder, and then compute log probabilities over the target sequence.
  * The DPO objective should follow the standard form:

    `L_DPO = -log(sigmoid(beta * ((log_p_chosen - log_ref_chosen) - (log_p_rejected - log_ref_rejected))))`

**Decision for Agent:** The project should expose a common `ASRModelInterface` that supports both Hugging Face and ESPnet-backed models. This keeps the training and evaluation logic modular while allowing either backend to be swapped in.

---

## 3. Base Model Specification: OWSM v4 vs. OpusLM

The implementation must abstract model interfaces to support at least two speech-model paradigms.

### 3.1 OWSM v4 (Open Whisper-Style Speech Model v4)

* Architecture: continuous feature sequence-to-sequence encoder-decoder Transformer.
* Audio input: log-mel spectrogram features.
* Mechanism: standard encoder-decoder cross-attention with sequence log-probability extraction.
* Role: primary baseline for conventional continuous-feature ASR models.
* Objective: compute $P(y \mid x_{audio})$ by feeding encoder memory and target token sequence into the decoder and collecting logits over the target tokens.

### 3.2 OpusLM (Speech Language Model)

* Architecture: decoder-only Transformer initialized from text LLMs such as SmolLM2.
* Audio input: quantized discrete audio tokens extracted via EnCodec, SoundStream, HuBERT, or a comparable tokenizer.
* Mechanism: unified next-token prediction across audio and text token streams.
* Role: benchmark for pure autoregressive speech-language modeling.
* Objective: compute $P(y_{text} \mid y_{audio\_tokens})$ using standard causal LM heads.

The common logic across both backends is the same: the model must support `generate(audio, **kwargs) -> str` and `forward_log_probs(audio, text) -> torch.Tensor`.

---

## 4. Repository Structure

The project follows a modular organization that separates data preparation, model abstraction, training, and evaluation.

```text
18813-asr-post-training/
├── configs/
│   ├── dpo_config.yaml
│   ├── grpo_config.yaml
│   ├── owsm_v4_dpo.yaml
│   ├── opuslm_grpo.yaml
│   └── sft_config.yaml
├── data_prep/
│   ├── build_preference_data.py
│   ├── download_datasets.sh
│   ├── download_open_datasets.py
│   ├── generate_preference_pairs.py
│   ├── load_local_datasets.py
│   ├── loader_disfluency.py
│   └── loader_numb3rs.py
├── models/
│   ├── base.py
│   ├── base_interface.py
│   ├── espnet_wrapper.py
│   ├── hf_wrapper.py
│   ├── owsm_wrapper.py
│   └── opuslm_wrapper.py
├── trainers/
│   ├── dpo_trainer.py
│   ├── grpo_trainer.py
│   ├── sft_trainer.py
│   └── __init__.py
├── evaluation/
│   ├── eval_pipeline.py
│   ├── metrics.py
│   ├── reward_hacking.py
│   └── __init__.py
├── main.py
├── README.md
├── requirements.txt
├── asr_rl_post_training_project_specification.md
└── details.md
```

The current project already includes a subset of this structure, including `models/base.py`, `models/hf_wrapper.py`, `models/espnet_wrapper.py`, and `trainers/*.py`, while the longer specification adds the more complete research-oriented file map.

---

## 5. Data Pipeline Specifications

The data pipeline is split into target-task post-training datasets and evaluation datasets.

### 5.1 Datasets Overview & Recommendations

| Dataset | Type / Size | Primary Purpose | Acquisition / Download Method |
| :--- | :--- | :--- | :--- |
| NVIDIA `preference-asr-bench` | Synthetic & real preference pairs | Ground-truth preference instructions and format evaluation | Hugging Face: `load_dataset("nvidia/preference-asr-bench")` |
| NVIDIA Numb3rs | ~4.89 h | Number, date, currency, and measurement formatting | Hugging Face or NeMo JSON manifest scripts |
| Switchboard & Disfluency Speech | ~260 h / ~10 h | Spontaneous speech, disfluency removal, and self-correction resolution | Local download with LDC credentials or local Fisher-like disfluency subsets |
| FLEURS (en_us & subsets) | ~12 h / language | Benchmark for format compliance under strict WER bounds | Hugging Face: `load_dataset("google/fleurs", "en_us")` |
| TED-LIUM Release 3 | ~452 h | Public speaking benchmark for ASR accuracy and formatting transfer | Hugging Face or OpenSLR |
| LibriSpeech (train-clean-100) | ~100 h | Compute regularizer and holdout set for ASR sanity checks | Hugging Face: `load_dataset("librispeech_asr", "clean")` |

### 5.2 Dataset Acquisition Strategy

The implementation should support both open-source datasets and local restricted datasets.

**A. Open-source datasets**

* LibriSpeech
  * `load_dataset("librispeech_asr", "all")` or a cleaned subset such as `clean`.
* FLEURS
  * `load_dataset("google/fleurs", "en_us")` with target language selection.
* TED-LIUM Release 3
  * `load_dataset("LIUM/tedlium", "release3")` if available via Hugging Face, or use OpenSLR download tools.
* NVIDIA Numb3rs
  * Use Hugging Face if available; otherwise fall back to NeMo-style JSON manifests and tarball extraction.

**B. Restricted datasets (manual / LDC)**

* Switchboard and TIMIT require local access and credentials.
* The data loader should assume the datasets are already available under local directories such as `./data/switchboard/` and `./data/timit/`.
* Implementation can use Kaldi-style parsing or `torchaudio.datasets` if the dataset format supports it.
* Disfluency speech can be handled with a local directory structure or parallel fluent/disfluent transcript variants.

### 5.3 Input Data Format

For SFT, the model needs `(audio_path, target_text)` samples.

For DPO and preference training, the dataset must be transformed into pairs:

```python
{
    "audio_path": str,
    "chosen": str,
    "rejected": str,
}
```

Where:

* `chosen`: the formatted transcript preferred by the judge (e.g., disfluency removed, numbers normalized, punctuation corrected)
* `rejected`: the raw or less-preferred transcript (e.g., contains `um`, `uh`, or non-normalized numeric forms)

### 5.4 Preference Pair Generation

The preference-pair generator should:

1. Take an input audio sample and its literal transcript.
2. Sample $K$ candidate hypotheses from the base model.
3. Compute a format score $S_{fmt}(y)$ based on rule-based checks or an LLM judge.
4. Reject any hypothesis whose WER against the literal transcript exceeds a threshold.
5. Rank all valid hypotheses by the combined objective.
6. Choose the top-scoring candidate as `chosen` and the lowest-scoring valid candidate as `rejected`.

The intended objective is roughly:

$$
S(y) = S_{fmt}(y) - \alpha \cdot \text{WER}(y, y_{verbatim})
$$

with the final pair defined as:

* $y_{chosen} = \arg\max_y S(y)$
* $y_{rejected} = \arg\min_y S(y)$ among valid hypotheses

This matches the project goal of improving formatting while strongly constraining WER degradation.

---

## 6. Model Interface (`models/`)

The project must define a base class that abstracts speech-model inference across different backends.

```python
class ASRModelInterface(ABC):
    @abstractmethod
    def forward_log_probs(self, audio, text) -> torch.Tensor:
        ...

    @abstractmethod
    def generate(self, audio, **kwargs) -> str:
        ...
```

### Required behavior

1. `forward_log_probs(audio, text) -> torch.Tensor`
   * Returns sequence log-probabilities for the given target text.
2. `generate(audio, **kwargs) -> str`
   * Returns decoded output text.

### Implementation notes for ESPnet / OWSM v4

If using ESPnet or OWSM v4 specifically:

* Extract the encoder and decoder.
* Pass audio features through the encoder.
* Tokenize the target text using the tokenizer.
* Feed encoder memory and target token IDs into the decoder.
* Gather log-probabilities for the target token indices.

This is the key operation needed for DPO and GRPO training, where the policy log-probability difference is computed between preferred and rejected sequences.

---

## 7. Training Pipeline (`trainers/`)

### 7.1 SFT baseline

The supervised baseline should be implemented first. It learns directly on formatted targets using standard teacher-forced cross-entropy. This serves as the reference control to compare against post-training methods.

### 7.2 DPO Trainer

The DPO trainer should:

1. Initialize a trainable `policy_model` and a frozen `reference_model`.
2. For a batch of `(audio, chosen, rejected)`:
   * compute $\pi_\theta(y_w \mid x)$ and $\pi_\theta(y_l \mid x)$ with `policy_model`
   * compute $\pi_{ref}(y_w \mid x)$ and $\pi_{ref}(y_l \mid x)$ with `reference_model`
   * compute the DPO loss using `torch.nn.functional.logsigmoid`
3. Track metrics such as:
   * margin between chosen and rejected log-probs
   * chosen log-probability
   * rejected log-probability
   * KL divergence between policy and reference

### 7.3 GRPO Trainer

For GRPO, each audio input $x$ produces $K$ sampled hypotheses.

1. Sample $K$ hypotheses $\{y_1, \dots, y_K\}$.
2. Compute reward for each hypothesis:
   * $R(x, y) = \lambda_{format} \times \text{Format\_Score}(y) - \lambda_{wer} \times \text{WER}(y, y_{ref})$
3. Normalize rewards within the group to get advantages $A_i$.
4. Use policy-gradient optimization constrained by KL divergence to the reference model.

### 7.4 Controlled Experimental Track

To ensure fair evaluation, all methods should be evaluated under matched conditions:

1. DPO / SimPO / IPO
2. GRPO
3. Learned verifier reward model + GRPO
4. Rejection sampling fine-tuning (RSFT)
5. SFT baseline

The project should keep data volume, model size, training budget, and inference budget as comparable as possible across methods so that differences reflect method quality rather than setup drift.

---

## 8. Evaluation Module (`evaluation/`)

The evaluation pipeline should run inference on benchmark sets such as LibriSpeech, TIMIT, FLEURS, and TED-LIUM.

Required metrics to log include:

* Acoustic fidelity: `jiwer.wer(reference_verbatim, prediction)`
* Format compliance rate: rule-based or LLM-based verification of number/date/entity formatting and disfluency removal
* Harmful edit rate: percentage of submitted edits that alter meaning or remove factual content
* Reward hacking tracker: cases where format score improves but WER degrades by more than 15%
* KL divergence: distribution shift between policy and reference model

The evaluation should be run after training and also during ablations to ensure the model is not trading recognition quality for better formatting.

---

## 9. Scientific Open Questions

1. Domain generalization: do style alignment gains transfer to unseen acoustic domains and unseen format instructions?
2. RL vs. matched SFT: does RL outperform SFT when both are trained on the same judge-filtered dataset?
3. Robustness: can the formatted transcripts remain stable under noisy audio and long-tail named entities?
4. Reward-model quality: how sensitive is the policy to reward model mis-specification or judge noise?

---

## 10. Agent Instructions (Implementation Order)

1. Setup dependencies and environment.
   * `torch`, `torchaudio`, `transformers`, `datasets`, `jiwer`, `trl`, `espnet`, `numpy`, `PyYAML`, `accelerate`, `wandb`.
2. Prepare data.
   * Download open datasets.
   * Write local loaders for restricted datasets.
   * Generate preference pairs from model hypotheses and rule-based or LLM scoring.
3. Implement model wrappers.
   * `ASRModelInterface`
   * `HFASRWrapper`
   * `EspnetASRWrapper`
4. Implement training baselines.
   * SFT baseline first.
   * Then DPO trainer.
   * Then GRPO trainer.
5. Implement evaluation.
   * WER, format compliance, harmful edit rate, reward-hacking tracker.
6. Wire everything together in `main.py`.
   * support `data_prep`, `train`, and `eval` tasks.

---

## 11. Summary

This project is a controlled research effort to answer whether RL post-training can improve ASR transcript formatting in a way that is both user-aligned and acoustically safe. The specification intentionally couples two model backends, a preference-data pipeline, and a rigorous evaluation framework so that the final implementation can benchmark real style alignment gains without sacrificing recognition fidelity.
