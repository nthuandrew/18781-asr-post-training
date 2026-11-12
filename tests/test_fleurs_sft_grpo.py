from evaluation.metrics import compute_format_reward, format_compliance
from experiments.fleurs_sft_grpo import aggregate_metrics, group_advantages


def test_group_advantages_are_centered_and_ordered() -> None:
    advantages = group_advantages([-1.0, 0.0, 2.0])
    assert abs(sum(advantages)) < 1.0e-5
    assert advantages[0] < advantages[1] < advantages[2]


def test_group_advantages_handle_identical_rewards() -> None:
    assert group_advantages([0.5, 0.5]) == [0.0, 0.0]


def test_format_compliance_observes_original_punctuation() -> None:
    assert format_compliance("Hello world 2.") > format_compliance("hello world two")


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
