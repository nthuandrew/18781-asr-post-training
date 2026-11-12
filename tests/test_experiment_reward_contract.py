from data_prep.download_open_datasets import build_fleurs_preference_pairs, build_fleurs_training_batch
from evaluation.metrics import compute_format_reward, wer_score
from models.espnet_wrapper import EspnetASRWrapper


def test_fleurs_training_batch_keeps_reward_contract() -> None:
    rows = [
        {
            "audio_path": "sample.wav",
            "transcript": "uh hello world",
            "formatted_transcript": "Hello world.",
        }
    ]
    batch = build_fleurs_training_batch(rows, max_examples=1)
    assert len(batch) == 1
    assert set(batch[0].keys()) >= {"audio_path", "reference", "chosen", "rejected", "reward"}
    assert batch[0]["reward"] >= 0.0


def test_format_reward_penalizes_wer_regression() -> None:
    reference = "hello world"
    clean = "Hello world."
    bad = "hello world blah"

    clean_reward = compute_format_reward(reference, clean)
    bad_reward = compute_format_reward(reference, bad)

    assert clean_reward > bad_reward
    assert bad_reward < 0.0


def test_espnet_wrapper_uses_speech2text_inference_contract(monkeypatch) -> None:
    class DummySpeech2Text:
        @classmethod
        def from_pretrained(cls, model_tag: str, **kwargs):
            assert model_tag == "espnet/owsm_v4_base_102M"
            return cls()

        def __call__(self, speech, text_prev=None, lang_sym=None, task_sym=None, predict_time=None):
            assert speech is not None
            return [[("hello world", ["hello", "world"], [1, 2], "hello world", None)]]

    monkeypatch.setattr("models.espnet_wrapper.Speech2Text", DummySpeech2Text)

    wrapper = EspnetASRWrapper("espnet/owsm_v4_base_102M")
    result = wrapper.generate([0.1, 0.2, 0.3])

    assert result == "hello world"
