import os
from pprint import pprint
from numpy.random import randint
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

def get_all_files_from(folder):
    files = []
    i = 0
    for (path, dirlist, filelist) in os.walk(folder):
        path = Path(path)
        for file in filelist:
            filepath = path.joinpath(Path(file))
            files.append(filepath)
    return files

class AudioDatasetAt(Dataset):
    def __init__(self, folder, srate=32000, transform=nn.Identity()):
        self.folder = folder
        self.transform = transform
        self.files = get_all_files_from(folder)
        self.len = len(self.files)
        self.srate = srate
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        audio, srate = torchaudio.load(self.files[idx])
        return self.transform(audio)
    
    def get_random(self):
        i = randint(self.len)
        print(f"Getting {i}th from the {len(self)} datasets.")
        return self[i]

def fixed_length_audio_yielder(waveform, numSamples=64000):
    idx = 0
    size = waveform.shape[-1]
    remaining = lambda : size - idx
    while remaining() > numSamples:
        yield waveform[:, idx: idx+numSamples]
        idx += numSamples
    
    if remaining():
        padding = (0, numSamples-remaining())
        yield F.pad(waveform[:, idx:], padding)

def batch_yielder(batch, numSamples=64000):
    for waveform in batch:
        yield from fixed_length_audio_yielder(waveform, numSamples)

def my_collater(batch, numSamples=64000):
    return torch.stack(list((batch_yielder(batch, numSamples))))
    

def train_():
    return None


