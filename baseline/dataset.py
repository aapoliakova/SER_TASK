import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

"""
Return transformed wav file and labels
"""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class IEMOCAMP_AUDIO(Dataset):
    def __init__(self, dataset_path, transformation=None, num_samples=31100):
        self.data = pd.read_csv(dataset_path, index_col=0)
        self.transform = transformation.to(device)
        self.num_samples = num_samples
        self.audio_path = 'data/IEMOCAP_full_release/'
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wav_path = self.audio_path + self.data['path'][index]
        y = self.data['labels'][index]
        wav, sr = torchaudio.load(wav_path)

        wav = wav.to(device)
        assert sr == 16000, "Sample rate have to be 16_000"
        wav = self.resize(wav)
        mel = self.transform(wav)
        return mel, y

    def resize(self, wav):
        n_channels, length = wav.shape
        assert n_channels == 1, "Number of channels has to be 1"
        if length > self.num_samples:
            return wav[:, :self.num_samples]
        elif length < self.num_samples:
            padding = self.num_samples - length
            last_dim_padding = (0, padding)
            wav = torch.nn.functional.pad(wav, last_dim_padding)
            return wav


if __name__ == "__main__":
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = IEMOCAMP_AUDIO(dataset_path='preprocessed_data/full_dataset.csv',
                         transformation=mel_spectrogram,
                         num_samples=32500)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]  # (1, 64, 64)
