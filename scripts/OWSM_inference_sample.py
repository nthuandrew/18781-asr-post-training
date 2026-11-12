import numpy as np
from datasets import load_dataset
from espnet2.bin.s2t_inference import Speech2Text

fleurs_language = "en_us"
ds = load_dataset("google/fleurs", fleurs_language, split="test")

row = ds[0]
audio = row["audio"]

if isinstance(audio, dict):
    waveform = np.asarray(audio["array"], dtype=np.float32)
else:
    samples = audio.get_all_samples()
    waveform = np.asarray(samples.data.squeeze(0).numpy(), dtype=np.float32)

# OWSM model
s2t = Speech2Text.from_pretrained(
    "espnet/owsm_v4_base_102M",
    lang_sym="<eng>",
    task_sym="<asr>",
)

# result format: [(text, tokens, token_ids, text_nospecial, hypothesis)]
result = s2t(waveform)
text = result[0][0]
print("ASR output:", text)