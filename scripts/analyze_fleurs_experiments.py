"""Generate reproducible tables and plots for the completed FLEURS experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_prep.text_preprocessing import inverse_normalize_number_words
from evaluation.metrics import wer_score


DEFAULT_REPORT_DIR = ROOT / "reports" / "fleurs_v3"
COLORS = {
    "Base": "#6b7280",
    "SFT (raw)": "#2563eb",
    "SFT (preprocessed)": "#0891b2",
    "DPO": "#d97706",
    "GRPO": "#7c3aed",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rolling_mean(values: list[float], window: int = 25) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    smooth = np.empty_like(array)
    for index in range(len(array)):
        start = max(0, index + 1 - window)
        smooth[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return smooth


def prediction_specs() -> list[tuple[str, Path]]:
    specs = [
        ("Base", ROOT / "outputs/sft_full_v3/base_predictions.jsonl"),
        ("SFT (raw)", ROOT / "outputs/sft_full_v3/sft_predictions.jsonl"),
    ]
    preprocessed = ROOT / "outputs/sft_preprocessed_full_v1/sft_predictions.jsonl"
    if preprocessed.exists():
        specs.append(("SFT (preprocessed)", preprocessed))
    specs.extend(
        [
            ("DPO", ROOT / "outputs/dpo_full_v3/dpo_predictions.jsonl"),
            ("GRPO", ROOT / "outputs/grpo_full_v3/grpo_predictions.jsonl"),
        ]
    )
    return specs


def metric_summary(
    predictions: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, float]]:
    metrics = ("wer", "format_compliance", "harmful_edit_rate")
    return {
        name: {
            **{
                metric: float(np.mean([float(row[metric]) for row in rows]))
                for metric in metrics
            },
            "number_normalized_wer": float(
                np.mean(
                    [
                        wer_score(
                            inverse_normalize_number_words(row["reference"]),
                            inverse_normalize_number_words(row["prediction"]),
                        )
                        for row in rows
                    ]
                )
            ),
        }
        for name, rows in predictions.items()
    }


def _add_epoch_boundaries(axis: Any, rows: list[dict[str, Any]]) -> None:
    """Mark the first optimizer step of every epoch after the first."""
    previous_epoch = rows[0]["epoch"]
    for row in rows[1:]:
        if row["epoch"] != previous_epoch:
            axis.axvline(row["optimizer_step"], color="#9ca3af", alpha=0.35, lw=0.8)
            previous_epoch = row["epoch"]


def _finish_training_figure(figure: Any, axis: Any, destination: Path) -> None:
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_sft_training_curve(destination: Path, name: str, log_path: Path) -> None:
    rows = read_jsonl(log_path)
    steps = [row["optimizer_step"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    curves = (
        ("Total loss", [row["loss"] for row in rows], COLORS[name], 1.0),
        (
            "CTC loss",
            [row["component_metrics"]["loss_ctc"] for row in rows],
            "#d97706",
            0.7,
        ),
        (
            "Attention loss",
            [row["component_metrics"]["loss_att"] for row in rows],
            "#7c3aed",
            0.7,
        ),
    )
    for label, values, color, alpha in curves:
        axis.plot(steps, rolling_mean(values), label=label, color=color, alpha=alpha)
    _add_epoch_boundaries(axis, rows)
    axis.set(
        title=f"{name} training objective",
        xlabel="Optimizer step",
        ylabel="Loss (25-step rolling mean)",
    )
    axis.legend(frameon=False)
    _finish_training_figure(figure, axis, destination)


def save_dpo_training_curve(destination: Path) -> None:
    rows = read_jsonl(ROOT / "outputs/dpo_full_v3/training_log.jsonl")
    steps = [row["optimizer_step"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    loss_lines = axis.plot(
        steps,
        rolling_mean([row["loss"] for row in rows]),
        color=COLORS["DPO"],
        label="DPO loss",
    )
    _add_epoch_boundaries(axis, rows)
    axis.set(
        title="DPO training objective",
        xlabel="Optimizer step",
        ylabel="Loss (25-step rolling mean)",
    )
    twin = axis.twinx()
    logit_lines = twin.plot(
        steps,
        rolling_mean([row["preference_logit"] for row in rows]),
        color="#92400e",
        alpha=0.7,
        label="Preference logit",
    )
    twin.set_ylabel("Preference logit (25-step rolling mean)")
    lines = loss_lines + logit_lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False)
    _finish_training_figure(figure, axis, destination)


def save_grpo_training_curve(destination: Path) -> None:
    rows = read_jsonl(ROOT / "outputs/grpo_full_v3/training_log.jsonl")
    steps = [row["optimizer_step"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    loss_lines = axis.plot(
        steps,
        rolling_mean([row["policy_loss"] for row in rows]),
        color=COLORS["GRPO"],
        label="Policy loss",
    )
    _add_epoch_boundaries(axis, rows)
    axis.set(
        title="GRPO training objective and reference drift",
        xlabel="Optimizer step",
        ylabel="Policy loss (25-step rolling mean)",
    )
    twin = axis.twinx()
    kl_lines = twin.plot(
        steps,
        rolling_mean([row["kl"] for row in rows]),
        color="#dc2626",
        alpha=0.7,
        label="KL",
    )
    twin.set_ylabel("KL to frozen reference (25-step rolling mean)")
    lines = loss_lines + kl_lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False)
    _finish_training_figure(figure, axis, destination)


def save_evaluation_metrics(
    destination: Path, summary: dict[str, dict[str, float]]
) -> None:
    names = list(summary)
    metrics = [
        ("wer", "WER ↓"),
        ("number_normalized_wer", "Number-normalized WER ↓"),
        ("format_compliance", "Format compliance ↑"),
        ("harmful_edit_rate", "Reference-token miss rate ↓"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = [100 * summary[name][metric] for name in names]
        bars = axis.bar(
            np.arange(len(names)), values, color=[COLORS[name] for name in names]
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("Percent")
        axis.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0, max(values) * 1.18)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_components(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    punctuation = []
    digits = []
    no_disfluency = []
    for row in rows:
        text = row["prediction"]
        punctuation.append(
            "," in text or bool(re.search(r"[.!?][\"')\]]?$", text.strip()))
        )
        digits.append(bool(re.search(r"\b\d+(?:[.,]\d+)?\b", text)))
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        no_disfluency.append(
            not bool(re.search(r"\b(?:um|uh|erm|mm|you\s+know)\b", normalized))
        )
    return tuple(float(np.mean(values)) for values in (punctuation, digits, no_disfluency))


def save_format_diagnostics(
    destination: Path, predictions: dict[str, list[dict[str, Any]]]
) -> None:
    names = list(predictions)
    components = np.asarray([_format_components(predictions[name]) for name in names])
    base = predictions["Base"]
    hack_rates = []
    normalized_hack_rates = []
    base_normalized_wer = np.asarray(
        [
            wer_score(
                inverse_normalize_number_words(row["reference"]),
                inverse_normalize_number_words(row["prediction"]),
            )
            for row in base
        ]
    )
    for name in names:
        if name == "Base":
            hack_rates.append(0.0)
            normalized_hack_rates.append(0.0)
            continue
        rows = predictions[name]
        hack_rates.append(
            float(
                np.mean(
                    [
                        row["format_compliance"] > base_row["format_compliance"]
                        and row["wer"] > base_row["wer"]
                        for row, base_row in zip(rows, base)
                    ]
                )
            )
        )
        normalized_wer = np.asarray(
            [
                wer_score(
                    inverse_normalize_number_words(row["reference"]),
                    inverse_normalize_number_words(row["prediction"]),
                )
                for row in rows
            ]
        )
        format_improved = np.asarray(
            [
                row["format_compliance"] > base_row["format_compliance"]
                for row, base_row in zip(rows, base)
            ]
        )
        normalized_hack_rates.append(
            float(np.mean(format_improved & (normalized_wer > base_normalized_wer)))
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(names))
    width = 0.25
    labels = ("Punctuation", "Digit", "No disfluency")
    for index, label in enumerate(labels):
        axes[0].bar(x + (index - 1) * width, 100 * components[:, index], width, label=label)
    axes[0].set_title("Format heuristic components")
    axes[0].set_ylabel("Utterances passing component (%)")
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].legend(frameon=False)

    hack_width = 0.36
    bars = axes[1].bar(
        x - hack_width / 2,
        100 * np.asarray(hack_rates),
        hack_width,
        label="Standard WER",
        color="#6b7280",
    )
    normalized_bars = axes[1].bar(
        x + hack_width / 2,
        100 * np.asarray(normalized_hack_rates),
        hack_width,
        label="Number-normalized WER",
        color="#0891b2",
    )
    axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes[1].bar_label(normalized_bars, fmt="%.1f", padding=2, fontsize=8)
    axes[1].set_title("Format improves while WER worsens")
    axes[1].set_ylabel("Paired test utterances (%)")
    axes[1].set_xticks(x, names, rotation=25, ha="right")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _rl_epoch_diagnostics(model: str) -> dict[int, dict[str, float]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(ROOT / f"outputs/{model}_full_v3/training_samples.jsonl"):
        groups[int(row["epoch"])].append(row)
    output: dict[int, dict[str, float]] = {}
    for epoch, rows in groups.items():
        output[epoch] = {
            "acceptance": float(np.mean([row["accepted"] for row in rows])),
            "unique_rewards": float(
                np.mean([len(set(round(value, 10) for value in row["rewards"])) for row in rows])
            ),
            "reward_spread": float(
                np.mean([max(row["rewards"]) - min(row["rewards"]) for row in rows])
            ),
        }
    return output


def save_rl_diagnostics(destination: Path) -> None:
    dpo = _rl_epoch_diagnostics("dpo")
    grpo = _rl_epoch_diagnostics("grpo")
    epochs = sorted(dpo)
    grpo_logs = read_jsonl(ROOT / "outputs/grpo_full_v3/training_log.jsonl")
    kl_by_epoch = {
        epoch: float(np.mean([row["kl"] for row in grpo_logs if row["epoch"] == epoch]))
        for epoch in epochs
    }

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(epochs, [100 * dpo[e]["acceptance"] for e in epochs], "o-", label="DPO")
    axes[0].plot(epochs, [100 * grpo[e]["acceptance"] for e in epochs], "o-", label="GRPO")
    axes[0].set(title="Usable training samples", ylabel="Accepted (%)", xlabel="Epoch")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [dpo[e]["unique_rewards"] for e in epochs], "o-", label="DPO")
    axes[1].plot(epochs, [grpo[e]["unique_rewards"] for e in epochs], "o-", label="GRPO")
    axes[1].set(title="Reward resolution", ylabel="Unique rewards per group", xlabel="Epoch")
    axes[1].legend(frameon=False)

    axes[2].plot(epochs, [kl_by_epoch[e] for e in epochs], "o-", color=COLORS["GRPO"])
    axes[2].set(title="GRPO drift from reference", ylabel="Mean KL", xlabel="Epoch")
    for axis in axes:
        axis.set_xticks(epochs)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {name: read_jsonl(path) for name, path in prediction_specs()}
    summary = metric_summary(predictions)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_sft_training_curve(
        args.output_dir / "sft_raw_training_curve.png",
        "SFT (raw)",
        ROOT / "outputs/sft_full_v3/training_log.jsonl",
    )
    save_sft_training_curve(
        args.output_dir / "sft_preprocessed_training_curve.png",
        "SFT (preprocessed)",
        ROOT / "outputs/sft_preprocessed_full_v1/training_log.jsonl",
    )
    save_dpo_training_curve(args.output_dir / "dpo_training_curve.png")
    save_grpo_training_curve(args.output_dir / "grpo_training_curve.png")
    save_evaluation_metrics(args.output_dir / "evaluation_metrics.png", summary)
    save_format_diagnostics(args.output_dir / "format_diagnostics.png", predictions)
    save_rl_diagnostics(args.output_dir / "rl_diagnostics.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
