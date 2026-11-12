from data_prep.download_open_datasets import build_fleurs_preference_pairs


def test_fleurs_preference_pairs_have_expected_outputs() -> None:
    rows = [
        {
            "audio_path": "sample.wav",
            "transcript": "uh hello world the total is one hundred and five dollars",
            "formatted_transcript": "Hello world. The total is one hundred and five (105) dollars.",
        }
    ]
    pairs = build_fleurs_preference_pairs(rows, max_pairs=1)
    assert len(pairs) == 1
    assert "audio_path" in pairs[0]
    assert "chosen" in pairs[0]
    assert "rejected" in pairs[0]
    assert "reference" in pairs[0]
