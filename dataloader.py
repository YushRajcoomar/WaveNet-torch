import pandas as pd
import torch
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset, DataLoader

class CustomAudioDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform
        self.pred_seconds = 3
        self.set_sample_rate = 16000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        audio_path = self.data.iloc[idx]['mp3_path']
        waveform, sample_rate = torchaudio.load(audio_path)
        resample_transform = T.Resample(sample_rate,16000)
        waveform = resample_transform(waveform)
        split_samples = self.pred_seconds * self.set_sample_rate
        train_waveform = waveform[:,:-split_samples]
        test_waveform = waveform[:,-split_samples:]


        if self.transform:
            for tfm in self.transform:
                train_waveform = tfm(train_waveform)
                test_waveform = tfm(test_waveform)
        return train_waveform , test_waveform