"""Small, reproducible OWSM SFT-vs-GRPO experiment on FLEURS.

This module deliberately uses a direct PyTorch loop.  It keeps the same OWSM
input contract as the ESPnet3 tutorial while making the policy log-probability
needed by GRPO explicit and inspectable.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile
import torch
import torch.nn.functional as F
import yaml
from datasets import Audio, load_dataset
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list
from espnet2.train.preprocessor import S2TPreprocessor

from evaluation.metrics import compute_format_reward, compute_metrics


@dataclass
class ExperimentConfig:
    model_tag: str = "espnet/owsm_v4_base_102M"
    fleurs_language: str = "en_us"
    lang_sym: str = "<eng>"
    task_sym: str = "<asr>"
    train_examples: int = 16
    eval_examples: int = 16
    sft_steps: int = 10
    grpo_steps: int = 5
    num_hypotheses: int = 4
    learning_rate: float = 3.0e-5
    grpo_learning_rate: float = 1.0e-6
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    reward_format_weight: float = 1.0
    reward_wer_weight: float = 1.5
    wer_tolerance: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 13
    device: str = "cuda"
    output_dir: str = "outputs/fleurs_sft_grpo_smoke"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_advantages(rewards: Sequence[float], eps: float = 1.0e-6) -> list[float]:
    """Return zero-mean, unit-variance advantages for one hypothesis group."""
    if not rewards:
        return []
    values = np.asarray(rewards, dtype=np.float32)
    if len(values) == 1 or float(values.std()) < eps:
        return [0.0 for _ in rewards]
    normalized = (values - values.mean()) / (values.std() + eps)
    return [float(value) for value in normalized]


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"wer": math.nan, "format_compliance": math.nan, "harmful_edit_rate": math.nan}
    keys = ("wer", "format_compliance", "harmful_edit_rate")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


class FleursSubset:
    """FLEURS subset using the byte-decoding path from the ESPnet3 tutorial."""

    def __init__(self, language: str, split: str, limit: int | None = None) -> None:
        dataset = load_dataset("google/fleurs", language, split=split)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        self.dataset = dataset.cast_column("audio", Audio(decode=False))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[int(index)]
        audio = row["audio"]
        if audio.get("bytes") is not None:
            speech, sample_rate = soundfile.read(io.BytesIO(audio["bytes"]), dtype="float32")
        else:
            speech, sample_rate = soundfile.read(audio["path"], dtype="float32")
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"FLEURS sample rate must be 16 kHz, got {sample_rate}")
        return {
            "id": str(row.get("id", index)),
            "speech": np.asarray(speech, dtype=np.float32),
            "reference": str(row["transcription"]).strip(),
        }


class OWSMExperimentModel:
    """OWSM inference, preprocessing, SFT loss, and sequence log-probability."""

    def __init__(self, config: ExperimentConfig, *, nbest: int = 1) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.inference = Speech2Text.from_pretrained(
            config.model_tag,
            lang_sym=config.lang_sym,
            task_sym=config.task_sym,
            beam_size=max(1, nbest),
            nbest=max(1, nbest),
            device=config.device,
        )
        self.model = self.inference.s2t_model
        self.preprocessor = S2TPreprocessor(
            train=False,
            token_type="bpe",
            token_list=self.inference.converter.token_list,
            bpemodel=self.inference.tokenizer.model,
            text_prev_name="text_prev",
            text_ctc_name="text_ctc",
            fs=16000,
        )

    def raw_example(self, sample: dict[str, Any], target: str | None = None) -> dict[str, Any]:
        transcript = str(target if target is not None else sample["reference"]).strip()
        return {
            "speech": sample["speech"],
            "text": f"{self.config.lang_sym}{self.config.task_sym}<notimestamps> {transcript}",
            "text_prev": "<na>",
            "text_ctc": transcript,
        }

    def preprocess(self, sample: dict[str, Any], target: str | None = None) -> dict[str, torch.Tensor]:
        encoded = self.preprocessor(str(sample.get("id", "sample")), self.raw_example(sample, target))
        batch: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else torch.as_tensor(value)
            tensor = tensor.to(self.device)
            batch[key] = tensor.unsqueeze(0)
            batch[f"{key}_lengths"] = torch.tensor([tensor.shape[0]], device=self.device)
        return batch

    def generate(self, speech: np.ndarray) -> list[str]:
        self.model.eval()
        with torch.inference_mode():
            results = self.inference(speech)
        hypotheses: list[str] = []
        for result in results:
            text = result[3] if len(result) > 3 and result[3] is not None else result[0]
            text = str(text or "").strip()
            if text and text not in hypotheses:
                hypotheses.append(text)
        return hypotheses

    def sft_loss(self, sample: dict[str, Any]) -> torch.Tensor:
        self.model.train()
        loss, _, _ = self.model(**self.preprocess(sample))
        return loss

    def sequence_log_probs(self, sample: dict[str, Any], hypotheses: Sequence[str]) -> torch.Tensor:
        """Teacher-forced, length-normalized log P(hypothesis | speech)."""
        if not hypotheses:
            return torch.empty(0, device=self.device)

        encoded = [self.preprocess(sample, hypothesis) for hypothesis in hypotheses]
        texts = [item["text"].squeeze(0) for item in encoded]
        text_prevs = [item["text_prev"].squeeze(0) for item in encoded]
        text_pad = pad_list(texts, self.model.ignore_id)
        prev_pad = pad_list(text_prevs, self.model.ignore_id)

        speech = encoded[0]["speech"]
        speech_lengths = encoded[0]["speech_lengths"]
        encoder_out, encoder_lens = self.model.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        encoder_out = encoder_out.expand(len(hypotheses), -1, -1)
        encoder_lens = encoder_lens.expand(len(hypotheses))

        sos = text_pad.new_tensor([self.model.sos])
        eos = text_pad.new_tensor([self.model.eos])
        decoder_inputs = [torch.cat([sos, text]) for text in texts]
        decoder_targets = [torch.cat([text, eos]) for text in texts]
        input_pad = pad_list(decoder_inputs, self.model.eos)
        target_pad = pad_list(decoder_targets, self.model.ignore_id)
        input_lengths = torch.tensor(
            [len(item) for item in decoder_inputs], device=self.device, dtype=torch.long
        )

        # All examples use text_prev=<na>, matching ESPnetS2TModel._calc_att_loss.
        if not all(self.model.na in prev for prev in text_prevs):
            raise ValueError("Only the <na> previous-text contract is supported in this experiment")
        logits, _ = self.model.decoder(encoder_out, encoder_lens, input_pad, input_lengths)
        mask = target_pad.ne(self.model.ignore_id)
        safe_targets = target_pad.masked_fill(~mask, 0)
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)
        return (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def train_sft(config: ExperimentConfig, train_set: FleursSubset) -> Path:
    runner = OWSMExperimentModel(config)
    optimizer = torch.optim.Adam(runner.model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for step in range(config.sft_steps):
        sample = train_set[step % len(train_set)]
        optimizer.zero_grad(set_to_none=True)
        loss = runner.sft_loss(sample)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runner.model.parameters(), config.max_grad_norm
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(
            f"[SFT {step + 1}/{config.sft_steps}] loss={losses[-1]:.4f} "
            f"grad_norm={float(grad_norm):.4f}"
        )

    path = Path(config.output_dir) / "sft.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": runner.model.state_dict(), "losses": losses, "config": asdict(config)}, path)
    del runner
    torch.cuda.empty_cache()
    return path


def train_grpo(config: ExperimentConfig, train_set: FleursSubset) -> Path:
    runner = OWSMExperimentModel(config, nbest=config.num_hypotheses)
    reference = copy.deepcopy(runner.model).eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.Adam(runner.model.parameters(), lr=config.grpo_learning_rate)
    history: list[dict[str, float]] = []

    for step in range(config.grpo_steps):
        sample = train_set[step % len(train_set)]
        hypotheses = runner.generate(sample["speech"])
        if len(hypotheses) < 2:
            print(f"[GRPO {step + 1}/{config.grpo_steps}] skipped: fewer than two unique hypotheses")
            continue
        rewards = [
            compute_format_reward(
                sample["reference"],
                hypothesis,
                format_weight=config.reward_format_weight,
                wer_weight=config.reward_wer_weight,
                wer_tolerance=config.wer_tolerance,
            )
            for hypothesis in hypotheses
        ]
        for rank, (hypothesis, reward) in enumerate(zip(hypotheses, rewards), start=1):
            print(f"  candidate {rank}: reward={reward:.4f} text={hypothesis!r}")
        advantages = torch.tensor(group_advantages(rewards), device=runner.device)
        if torch.count_nonzero(advantages).item() == 0:
            print(f"[GRPO {step + 1}/{config.grpo_steps}] skipped: rewards have zero variance")
            continue

        optimizer.zero_grad(set_to_none=True)
        # Evaluation mode disables SpecAug/dropout but still preserves gradients.
        # This keeps the optimized log-probability aligned with decoded candidates.
        runner.model.eval()
        policy_logp = runner.sequence_log_probs(sample, hypotheses)
        old_logp = policy_logp.detach()
        with torch.no_grad():
            policy_model = runner.model
            runner.model = reference
            ref_logp = runner.sequence_log_probs(sample, hypotheses)
            runner.model = policy_model

        ratio = torch.exp(policy_logp - old_logp)
        clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        log_ratio = ref_logp - policy_logp
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        loss = policy_loss + config.kl_beta * kl
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runner.model.parameters(), config.max_grad_norm
        )
        optimizer.step()
        record = {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(np.mean(rewards)),
            "kl": float(kl.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        history.append(record)
        print(
            f"[GRPO {step + 1}/{config.grpo_steps}] loss={record['loss']:.4f} "
            f"reward={record['mean_reward']:.4f} kl={record['kl']:.4f} "
            f"grad_norm={record['grad_norm']:.4f}"
        )

    path = Path(config.output_dir) / "grpo.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": runner.model.state_dict(), "history": history, "config": asdict(config)}, path)
    del runner, reference
    torch.cuda.empty_cache()
    return path


def evaluate(
    config: ExperimentConfig,
    eval_set: FleursSubset,
    checkpoint: Path | None,
    name: str,
) -> dict[str, float]:
    runner = OWSMExperimentModel(config)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(payload["state_dict"])

    rows: list[dict[str, Any]] = []
    prediction_path = Path(config.output_dir) / f"{name}_predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index in range(len(eval_set)):
            sample = eval_set[index]
            hypotheses = runner.generate(sample["speech"])
            prediction = hypotheses[0] if hypotheses else ""
            metrics = compute_metrics(sample["reference"], prediction)
            row = {
                "id": sample["id"],
                "reference": sample["reference"],
                "prediction": prediction,
                **metrics,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{name} eval {index + 1}/{len(eval_set)}] WER={metrics['wer']:.3f}")
    summary = aggregate_metrics(rows)
    del runner
    torch.cuda.empty_cache()
    return summary


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return ExperimentConfig(**values)


def run(config: ExperimentConfig, stage: str = "all") -> dict[str, dict[str, float]]:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8"
    )

    train_set = FleursSubset(config.fleurs_language, "train", config.train_examples)
    eval_set = FleursSubset(config.fleurs_language, "test", config.eval_examples)
    sft_path = output_dir / "sft.pt"
    grpo_path = output_dir / "grpo.pt"

    if stage in {"all", "train", "sft"}:
        sft_path = train_sft(config, train_set)
    if stage in {"all", "train", "grpo"}:
        grpo_path = train_grpo(config, train_set)

    summaries: dict[str, dict[str, float]] = {}
    if stage in {"all", "eval"}:
        for name, checkpoint in (("base", None), ("sft", sft_path), ("grpo", grpo_path)):
            if checkpoint is not None and not checkpoint.exists():
                raise FileNotFoundError(f"Missing {name} checkpoint: {checkpoint}")
            summaries[name] = evaluate(config, eval_set, checkpoint, name)
        (output_dir / "comparison.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        print("\nmodel\tWER\tformat\tharmful_edit")
        for name, summary in summaries.items():
            print(
                f"{name}\t{summary['wer']:.4f}\t{summary['format_compliance']:.4f}"
                f"\t{summary['harmful_edit_rate']:.4f}"
            )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fleurs_sft_grpo_smoke.yaml")
    parser.add_argument(
        "--stage", choices=("all", "train", "sft", "grpo", "eval"), default="all"
    )
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--sft-steps", type=int)
    parser.add_argument("--grpo-steps", type=int)
    parser.add_argument("--num-hypotheses", type=int)
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    for field in (
        "train_examples",
        "eval_examples",
        "sft_steps",
        "grpo_steps",
        "num_hypotheses",
        "output_dir",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    run(config, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
