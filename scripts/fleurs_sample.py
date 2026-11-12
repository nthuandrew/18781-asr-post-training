import numpy as np
from datasets import load_dataset
from scipy.io import wavfile

fleurs_language = "en_us"
ds = load_dataset("google/fleurs", fleurs_language, split="test", trust_remote_code=True)

row = ds[0]
audio = row["audio"]

# dataset audio may be a dict or a torchcodec decoder
if isinstance(audio, dict):
    waveform = audio["array"]
    sr = audio["sampling_rate"]
else:
    samples = audio.get_all_samples()
    waveform = samples.data.squeeze(0).numpy()
    sr = audio.metadata.sample_rate

# Save as .wav
waveform = np.asarray(waveform, dtype=np.float32)
wavfile.write("fleurs_sample.wav", sr, waveform)

print("saved:", "fleurs_sample.wav")
print("shape:", waveform.shape, "sr:", sr)