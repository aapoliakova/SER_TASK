import torchaudio
from torch.utils.data import DataLoader
import torch
from torch import nn

from baseline.dataset import IEMOCAMP_AUDIO
from baseline_model import CNNNetwork

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")

"""
Data prep block 
"""
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

train_data = IEMOCAMP_AUDIO(dataset_path='preprocessed_data/audio_train.csv',
                            transformation=mel_spectrogram,
                            num_samples=32500)

val_test_data = IEMOCAMP_AUDIO(dataset_path='preprocessed_data/audio_test.csv',
                               transformation=mel_spectrogram,
                               num_samples=32500)

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(val_test_data, batch_size=batch_size)

"""
Model setup 
"""

cnn = CNNNetwork().to(device)
print(cnn)

# initialise loss function + optimiser
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(),
                             lr=1e-3)

X, y = next(iter(train_dataloader))
print(X)



def train_single_epoch(model, data_loader, loss_fn, optimiser):
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # calculate loss
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # back-propagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
