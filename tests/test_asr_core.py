import math

from evaluation.metrics import compute_metrics, format_compliance, wer_score
from trainers.dpo_trainer import DPOConfig, DPOTrainer


def test_wer_score_zero_for_exact_match() -> None:
    assert wer_score("hello world", "hello world") == 0.0


def test_format_compliance_rejects_disfluency_and_accepts_clean_number_format() -> None:
    assert format_compliance("um hello one hundred") == 0.0
    assert format_compliance("hello 100") > 0.0


def test_dpo_loss_is_finite_and_monotonic_with_margin() -> None:
    trainer = DPOTrainer(policy_model=None, reference_model=None, config=DPOConfig(beta=0.5))
    loss1 = trainer._dpo_loss(1.0, 0.3, 0.2, 0.8)
    loss2 = trainer._dpo_loss(2.0, 0.3, 0.2, 0.8)
    assert math.isfinite(loss1)
    assert math.isfinite(loss2)
    assert loss2 < loss1
