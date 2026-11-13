import torch

from evaluation.metrics import compute_format_reward, format_compliance
from experiments.asr_common import (
    aggregate_metrics,
    parse_common_config,
    resolve_training_steps,
)
from experiments.dpo import dpo_loss, select_preference_pair
from experiments.grpo import group_advantages
from data_prep.text_preprocessing import (
    format_sft_target_v1,
    inverse_normalize_number_words,
)


def test_dataset_is_selected_from_config() -> None:
    common = parse_common_config(
        {
            "dataset": {
                "path": "example/speech-dataset",
                "subset": "english",
                "audio_column": "waveform",
                "text_column": "sentence",
            }
        }
    )
    assert common.dataset.path == "example/speech-dataset"
    assert common.dataset.subset == "english"
    assert common.dataset.audio_column == "waveform"
    assert common.dataset.text_column == "sentence"


def test_group_advantages_are_centered_and_ordered() -> None:
    advantages = group_advantages([-1.0, 0.0, 2.0])
    assert abs(sum(advantages)) < 1.0e-5
    assert advantages[0] < advantages[1] < advantages[2]


def test_group_advantages_handle_identical_rewards() -> None:
    assert group_advantages([0.5, 0.5]) == [0.0, 0.0]


def test_epochs_resolve_to_complete_dataset_passes() -> None:
    assert resolve_training_steps(steps=None, epochs=3, dataset_size=2602) == 7806


def test_dpo_loss_decreases_when_policy_preference_margin_improves() -> None:
    reference = torch.tensor([-2.0, -2.0])
    baseline_loss, _ = dpo_loss(torch.tensor([-2.0, -2.0]), reference, beta=0.1)
    improved_loss, _ = dpo_loss(torch.tensor([-1.0, -3.0]), reference, beta=0.1)
    assert improved_loss < baseline_loss


def test_select_preference_pair_uses_strict_reward_extremes() -> None:
    pair = select_preference_pair(["middle", "best", "worst"], [0.5, 1.0, -1.0])
    assert pair == ("best", "worst")


def test_select_preference_pair_rejects_reward_tie() -> None:
    assert select_preference_pair(["first", "second"], [0.5, 0.5]) is None


def test_select_preference_pair_applies_margin_and_wer_gates() -> None:
    hypotheses = ["best", "worst"]
    assert (
        select_preference_pair(
            hypotheses,
            [0.6, 0.55],
            wers=[0.1, 0.2],
            min_reward_margin=0.1,
            max_chosen_wer=0.15,
        )
        is None
    )
    assert (
        select_preference_pair(
            hypotheses,
            [0.8, 0.5],
            wers=[0.2, 0.3],
            min_reward_margin=0.1,
            max_chosen_wer=0.15,
        )
        is None
    )


def test_format_compliance_observes_original_punctuation() -> None:
    assert format_compliance("Hello world 2.") > format_compliance("hello world two")


def test_format_compliance_ignores_initial_capitalization() -> None:
    assert format_compliance("Hello world.") == format_compliance("hello world.")


def test_format_compliance_counts_comma_as_punctuation() -> None:
    assert format_compliance("hello, world") > format_compliance("hello world")
    assert format_compliance("hello, world.") == format_compliance("hello world.")


def test_wer_tolerance_is_a_free_window() -> None:
    reward_without_tolerance = compute_format_reward("hello world", "hello there")
    reward_with_tolerance = compute_format_reward(
        "hello world", "hello there", wer_tolerance=0.5
    )
    assert reward_with_tolerance > reward_without_tolerance


def test_aggregate_metrics() -> None:
    summary = aggregate_metrics(
        [
            {"wer": 0.0, "format_compliance": 1.0, "harmful_edit_rate": 0.0},
            {"wer": 1.0, "format_compliance": 0.0, "harmful_edit_rate": 1.0},
        ]
    )
    assert summary == {"wer": 0.5, "format_compliance": 0.5, "harmful_edit_rate": 0.5}


def test_sft_target_preprocessing_preserves_words_around_numbers() -> None:
    text = "prides include one to three males and as many as thirty females"
    assert format_sft_target_v1(text) == (
        "prides include 1 to 3 males and as many as 30 females."
    )


def test_sft_target_preprocessing_handles_years_and_ordinals() -> None:
    assert inverse_normalize_number_words("in the twentieth century") == (
        "in the 20th century"
    )
    assert inverse_normalize_number_words("on fifteen august nineteen forty") == (
        "on 15 august 1940"
    )


def test_sft_target_preprocessing_keeps_existing_numeric_entities() -> None:
    assert format_sft_target_v1("compatible with 802.11a and 32%") == (
        "compatible with 802.11a and 32%."
    )
