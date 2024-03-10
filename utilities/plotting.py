import matplotlib.pyplot as plt
import torch

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

def plot_statistics(statistics):

    tl = statistics['TrainLosses']
    vl = statistics['ValidationLosses']
    time_axis = torch.arange(0,len(tl))/len(tl)

    waveform = [tl, vl]
    names = ["Train Loss", "Validation Loss"]

    figure, axes = plt.subplots(2, 1)
    num_channels = 2

    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"{names[c]}")
    figure.suptitle("Training and validation losses")
