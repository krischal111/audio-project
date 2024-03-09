import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio

class LazyCausalConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.causal_padding_left = (dilation) * (kernel_size-1)
        self.lazyLayer = nn.LazyConv1d(out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        return self.lazyLayer(F.pad(x, [self.causal_padding_left, 0])) #, self.weight, self.bias)


class ConvolutionalStack(nn.Module):
    def __init__(self, out_channels, k_size, stride=1, dilation=1):
        super().__init__()
        self.ELU1 = nn.ELU()
        self.Conv1 = LazyCausalConv1d(out_channels, kernel_size=k_size, stride=stride, dilation=dilation)
        self.BNorm1 = nn.LazyBatchNorm1d()
    def forward(self, x):
        x = self.ELU1(x)
        x = self.Conv1(x)
        x = self.BNorm1(x)
        return x

class Down(nn.Module):
    def __init__(self, out_channels, k_size, factor=2):
        super().__init__()
        self.InConv = nn.LazyConv1d(out_channels, 1)
        self.Conv1 = ConvolutionalStack(out_channels, k_size=k_size, dilation=1)
        self.Conv2 = ConvolutionalStack(out_channels, k_size=k_size, dilation=2)
        self.Conv3 = ConvolutionalStack(out_channels, k_size=k_size, dilation=4)
        self.Conv4 = ConvolutionalStack(out_channels, k_size=k_size, dilation=8)
        self.Pooling = nn.AvgPool1d(factor,factor)
        self.Residual = nn.LazyConv1d(out_channels, kernel_size=factor, stride=factor)
    
    def forward(self, x):
        oldx = x
        # print(1, x.shape)
        x = self.InConv(x)
        # print(2, x.shape)
        x = self.Conv1(x)
        # print(3, x.shape)
        x = self.Conv2(x)
        # print(4, x.shape)
        x = self.Conv3(x)
        # print(5, x.shape)
        x = self.Conv4(x)
        # print(6, x.shape)
        x = self.Pooling(x)
        # print(7, x.shape)
        newx = self.Residual(oldx)
        x += newx
        return x

        # try:
        # # except:
        #     print("At Down")
        #     print(x.shape, newx.shape)
        #     some_shit = "error"
        #     raise some_shit
    
class Up(nn.Module):
    def __init__(self, out_channels, k_size, factor=2):
        super().__init__()
        self.Upsampler = nn.Upsample(scale_factor=factor)
        self.InConv = nn.LazyConv1d(out_channels, 1)
        self.Conv1 = ConvolutionalStack(out_channels, k_size, dilation=1)
        self.Conv2 = ConvolutionalStack(out_channels, k_size, dilation=2)
        self.Conv3 = ConvolutionalStack(out_channels, k_size, dilation=4)
        self.Conv4 = ConvolutionalStack(out_channels, k_size, dilation=8)
        self.Residual = nn.LazyConv1d(out_channels, 1)
    
    def forward(self, x):
        # print("Before upsampling", x.shape)
        x = self.Upsampler(x)
        # print("After upsampling", x.shape)
        oldx = x
        x = self.InConv(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        newx = self.Residual(oldx)
        x += newx
        return x

class Encoder(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.encoder = nn.Sequential(
            Down(out_channels=4, k_size=11, factor=2),
            Down(out_channels=16, k_size=11, factor=2),
            Down(out_channels=64, k_size=11, factor=2),
            Down(out_channels=256, k_size=11, factor=2),
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.decoder = nn.Sequential(
            Up(out_channels=64, k_size=11, factor=2),
            Up(out_channels=16, k_size=11, factor=2),
            Up(out_channels=4, k_size=11, factor=2),
            Up(out_channels=1, k_size=11, factor=2),
        )
    def forward(self, x):
        return self.decoder(x)


class LossFunction(nn.Module):
    def __init__(self, srate=32000):
        super().__init__()
        self.tf1 = torchaudio.transforms.Spectrogram(srate)
        self.tf2 = torchaudio.transforms.Spectrogram(srate)
        self.spectral_loss = nn.SmoothL1Loss()
        self.waveform_loss = nn.MSELoss()

    def forward(self, x, recons):
        device = 'cpu'
        x = x.to(device)
        recons = recons.to(device)
        loss = torch.Tensor([0])
        # loss = .0001*self.spectral_loss(self.tf1(x), self.tf2(recons))
        loss += self.waveform_loss(recons, x)
        return loss


        



