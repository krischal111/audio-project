from pathlib import Path
import pandas as pd
import torchaudio
from torch.nn import functional as F
import torch
from torch.utils.data import random_split

counting_tool = lambda  x, y : (x-1)//y + 1

def isAudioFile(filename):
    ''' 
    Returns True if given file is an audio file.

    filename: Anything that is like a file name.
    '''
    try:
        file = Path(filename)
        fileinfo = torchaudio.info(file)
        return (fileinfo.num_channels, fileinfo.num_frames)
    except:
        return False

def getAudioSplittingDataFrame(folder, numSamples, limit=0):
    audios = pd.DataFrame(columns=["name", "channel", "start"])
    i = 0
    for file in Path(folder).rglob("*"):
        if fileinfo := isAudioFile(file):
            ch, size = fileinfo
            counts = counting_tool(size, numSamples)
            how_many = ch*counts
            new_row = pd.DataFrame({
                "name":[file]*how_many, 
                "channel":[i for i in range(ch)] * counts, 
                "start":[i*numSamples for i in range(counts)] * ch
                })
            audios=pd.concat([audios,new_row])

            i += how_many
            if limit and i > limit:
                break
    return audios

def selector(file, numSamples) -> torch.Tensor:
    wf, rate = torchaudio.load(file["name"], frame_offset=file["start"], num_frames=numSamples)
    padding = numSamples - wf.shape[-1]
    return F.pad(wf[file["channel"]].unsqueeze(0), [0, padding])

def my_collater(df, numSamples):
    return torch.stack(
        [selector(file, numSamples) for file in df.itertuples()]
    )

from torch.utils.data import Dataset
from torch import nn

def get_random(dataset) -> torch.Tensor:
    from numpy.random import randint
    idx = randint(len(dataset))
    print(f"Getting {idx}th item from the {len(dataset)} datasets.")
    return dataset[idx]

class AudioDatasetAt(Dataset):
    def __init__(self, folder, srate=32000, split_duration=3,transform = nn.Identity(), limit=0):
        self.folder = folder
        self.transform =transform
        self.srate = srate
        self.numSamples = srate * split_duration
        self.files_df = getAudioSplittingDataFrame(folder, self.numSamples, limit=limit)
    
    def __len__(self):
        return len(self.files_df.index)
    
    def __getitem__(self, index) -> torch.Tensor:
        return selector(self.files_df.iloc[index], self.numSamples)
    
    def get_random(self) -> torch.Tensor:
        return get_random(self)

def split_dataset(dataset: Dataset, split_ratios, seed=None):
    seed = seed if seed is not None else 42
    n = len(dataset)
    splits = [int(k*n)+1 for k in split_ratios]
    excess = sum(splits) - n
    splits[splits.index(max(splits))] -= excess
    return random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
