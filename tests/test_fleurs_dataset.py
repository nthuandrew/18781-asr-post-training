from data_prep.download_open_datasets import format_transcript_for_fleurs, score_format_compliance


def test_format_transcript_removes_disfluencies_and_normalizes_numbers() -> None:
    text = "uh hello this is one hundred and five dollars"
    formatted = format_transcript_for_fleurs(text)
    assert "uh" not in formatted.lower()
    assert "one hundred and five" in formatted.lower()
    assert "105" in formatted


def test_format_score_rewards_clean_written_format() -> None:
    clean = "Hello world. The total is 105 dollars."
    dirty = "uh hello world the total is one hundred and five dollars"
    assert score_format_compliance(clean) > score_format_compliance(dirty)
