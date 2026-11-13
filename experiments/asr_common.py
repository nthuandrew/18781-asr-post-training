"""Shared dataset, OWSM, configuration, and evaluation utilities."""

from __future__ import annotations

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
from tqdm.auto import tqdm

from evaluation.metrics import compute_metrics


@dataclass
class ModelConfig:
    tag: str = "espnet/owsm_v4_base_102M"
    lang_sym: str = "<eng>"
    task_sym: str = "<asr>"


@dataclass
class DatasetConfig:
    path: str = "google/fleurs"
    subset: str | None = "en_us"
    train_split: str = "train"
    eval_split: str = "test"
    audio_column: str = "audio"
    text_column: str = "transcription"
    id_column: str = "id"
    sampling_rate: int = 16000
    train_examples: int | None = 16
    eval_examples: int | None = 16
    selection: str = "head"


@dataclass
class RuntimeConfig:
    seed: int = 13
    device: str = "cuda"
    output_dir: str = "outputs/asr_experiment"


@dataclass
class RewardConfig:
    format_weight: float = 1.0
    wer_weight: float = 1.5
    wer_tolerance: float = 0.0


@dataclass
class CommonConfig:
    model: ModelConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return values


def parse_common_config(values: dict[str, Any]) -> CommonConfig:
    return CommonConfig(
        model=ModelConfig(**values.get("model", {})),
        dataset=DatasetConfig(**values.get("dataset", {})),
        runtime=RuntimeConfig(**values.get("runtime", {})),
    )


def write_resolved_config(output_dir: str | Path, values: dict[str, Any]) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"wer": math.nan, "format_compliance": math.nan, "harmful_edit_rate": math.nan}
    keys = ("wer", "format_compliance", "harmful_edit_rate")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


class ASRDataset:
    """Config-driven Hugging Face audio dataset adapter."""

    def __init__(self, config: DatasetConfig, *, train: bool, seed: int) -> None:
        split = config.train_split if train else config.eval_split
        limit = config.train_examples if train else config.eval_examples
        load_args: list[str] = [config.path]
        if config.subset:
            load_args.append(config.subset)
        dataset = load_dataset(*load_args, split=split)
        if config.selection == "shuffle":
            dataset = dataset.shuffle(seed=seed)
        elif config.selection != "head":
            raise ValueError("dataset.selection must be 'head' or 'shuffle'")
        if limit is not None:
            dataset = dataset.select(range(min(int(limit), len(dataset))))
        self.dataset = dataset.cast_column(config.audio_column, Audio(decode=False))
        self.config = config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[int(index)]
        audio = row[self.config.audio_column]
        if audio.get("bytes") is not None:
            speech, sample_rate = soundfile.read(io.BytesIO(audio["bytes"]), dtype="float32")
        elif audio.get("path"):
            speech, sample_rate = soundfile.read(audio["path"], dtype="float32")
        else:
            raise ValueError(f"Audio row has neither bytes nor path at index {index}")
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        if sample_rate != self.config.sampling_rate:
            raise ValueError(
                f"Expected {self.config.sampling_rate} Hz audio, got {sample_rate}; "
                "resampling is not implemented in this adapter"
            )
        return {
            "id": str(row.get(self.config.id_column, index)),
            "speech": np.asarray(speech, dtype=np.float32),
            "reference": str(row[self.config.text_column]).strip(),
        }


class OWSMModel:
    """OWSM inference, preprocessing, SFT loss, and sequence log-probability."""

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        *,
        nbest: int = 1,
    ) -> None:
        self.model_config = model_config
        self.device = torch.device(runtime_config.device)
        self.inference = Speech2Text.from_pretrained(
            model_config.tag,
            lang_sym=model_config.lang_sym,
            task_sym=model_config.task_sym,
            beam_size=max(1, nbest),
            nbest=max(1, nbest),
            device=runtime_config.device,
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
            "text": (
                f"{self.model_config.lang_sym}{self.model_config.task_sym}"
                f"<notimestamps> {transcript}"
            ),
            "text_prev": "<na>",
            "text_ctc": transcript,
        }

    def preprocess(self, sample: dict[str, Any], target: str | None = None) -> dict[str, torch.Tensor]:
        encoded = self.preprocessor(
            str(sample.get("id", "sample")), self.raw_example(sample, target)
        )
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

    def sequence_log_probs(
        self,
        sample: dict[str, Any],
        hypotheses: Sequence[str],
        *,
        length_normalize: bool,
    ) -> torch.Tensor:
        """Return teacher-forced sequence log-probabilities for hypotheses."""
        if not hypotheses:
            return torch.empty(0, device=self.device)
        encoded = [self.preprocess(sample, hypothesis) for hypothesis in hypotheses]
        texts = [item["text"].squeeze(0) for item in encoded]
        text_prevs = [item["text_prev"].squeeze(0) for item in encoded]
        text_pad = pad_list(texts, self.model.ignore_id)

        encoder_out, encoder_lens = self.model.encode(
            encoded[0]["speech"], encoded[0]["speech_lengths"]
        )
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
        if not all(self.model.na in previous for previous in text_prevs):
            raise ValueError("Only the <na> previous-text contract is supported")

        logits, _ = self.model.decoder(encoder_out, encoder_lens, input_pad, input_lengths)
        mask = target_pad.ne(self.model.ignore_id)
        safe_targets = target_pad.masked_fill(~mask, 0)
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)
        sequence_log_probs = (token_log_probs * mask).sum(dim=1)
        if length_normalize:
            sequence_log_probs = sequence_log_probs / mask.sum(dim=1).clamp_min(1)
        return sequence_log_probs


def evaluate(
    common: CommonConfig,
    eval_set: ASRDataset,
    checkpoint: Path | None,
    name: str,
) -> dict[str, float]:
    runner = OWSMModel(common.model, common.runtime)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(payload["state_dict"])

    rows: list[dict[str, Any]] = []
    prediction_path = Path(common.runtime.output_dir) / f"{name}_predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(
            range(len(eval_set)),
            desc=f"{name} eval",
            unit="sample",
            dynamic_ncols=True,
        )
        for index in progress:
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
            progress.set_postfix(wer=f"{metrics['wer']:.3f}")
    summary = aggregate_metrics(rows)
    del runner
    torch.cuda.empty_cache()
    return summary


def print_and_save_comparison(
    output_dir: str | Path, summaries: dict[str, dict[str, float]]
) -> None:
    destination = Path(output_dir)
    (destination / "comparison.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print("\nmodel\tWER\tformat\tharmful_edit")
    for name, summary in summaries.items():
        print(
            f"{name}\t{summary['wer']:.4f}\t{summary['format_compliance']:.4f}"
            f"\t{summary['harmful_edit_rate']:.4f}"
        )
