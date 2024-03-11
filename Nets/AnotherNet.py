from importlib import reload
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import vector_quantize_pytorch
import auraloss

import Nets.causalConv as causalConv
reload(causalConv)
from Nets.causalConv import LazyCausalConv1d, CausalPoolingDoubleFactor

class ConvolutionalStack(nn.Module):
    def __init__(self, out_channels, k_size, dilation=1):
        super().__init__()
        # self.BNorm1 = nn.LazyBatchNorm1d()
        self.ELU1 = nn.ELU()
        self.Conv1 = LazyCausalConv1d(out_channels, kernel_size=k_size, dilation=dilation)
    def forward(self, x):
        # x = self.BNorm1(x)
        x = self.ELU1(x)
        x = self.Conv1(x)
        return x

class ResidualDown(nn.Module):
    def __init__(self, out_channels, k_size, factor=2):
        super().__init__()
        self.Conv1 = ConvolutionalStack(out_channels, k_size=k_size, dilation=1)
        self.Conv2 = ConvolutionalStack(out_channels, k_size=k_size, dilation=3)
        self.Conv3 = ConvolutionalStack(out_channels, k_size=k_size, dilation=9)
        self.StridedConvLayer = LazyCausalConv1d(out_channels, kernel_size=2*factor, stride=factor)
    
    def forward(self, x):
        x = x0 = x
        x = x1 = self.Conv1(x)
        x = x2 = self.Conv2(torch.cat([x0, x1], dim=-2))
        x = x3 = self.Conv3(torch.cat([x0, x1, x2], dim=-2))
        x = x4 = self.StridedConvLayer(torch.cat([x0, x1, x2, x3], dim=-2))
        return x4
    
class ResidualUp(nn.Module):
    def __init__(self, out_channels, k_size, factor=2):
        super().__init__()
        self.Upsampler = nn.Upsample(scale_factor=factor, mode='linear')
        self.Conv1 = ConvolutionalStack(out_channels, k_size, dilation=1)
        self.Conv2 = ConvolutionalStack(out_channels, k_size, dilation=3)
        self.Conv3 = ConvolutionalStack(out_channels, k_size, dilation=9)
        self.Residual = nn.LazyConv1d(out_channels, 1)
    
    def forward(self, x):
        x = x0 = self.Upsampler(x)
        x = x1 = self.Conv1(x)
        x = x2 = self.Conv2(torch.cat([x0, x1], dim = -2))
        x = x3 = self.Conv3(torch.cat([x0, x1, x2], dim=-2))
        x = x4 = self.Residual(torch.cat([x0, x1, x2, x3], dim=-2))
        return x4

class Encoder(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualDown(out_channels=2, k_size=11, factor=2),
            ResidualDown(out_channels=4, k_size=11, factor=2),
            ResidualDown(out_channels=8, k_size=11, factor=2),
            ResidualDown(out_channels=16, k_size=11, factor=5),
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.decoder = nn.Sequential(
            ResidualUp(out_channels=8, k_size=11, factor=5),
            ResidualUp(out_channels=4, k_size=11, factor=2),
            ResidualUp(out_channels=2, k_size=11, factor=2),
            ResidualUp(out_channels=1, k_size=11, factor=2),
        )
    def forward(self, x):
        return self.decoder(x)

class Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = vector_quantize_pytorch.VectorQuantize(16, 65536)
    
    def forward(self, x):
        x, _, loss =  self.quantizer(x.transpose(-1, -2))
        return x.transpose(-2, -1), loss


class LossFunction(nn.Module):
    def __init__(self, srate=32000):
        super().__init__()
        # self.tf1 = torchaudio.transforms.Spectrogram(srate, normalized=True)
        # self.tf2 = torchaudio.transforms.Spectrogram(srate, normalized=True)
        # self.spectral_loss = nn.SmoothL1Loss()
        # self.waveform_loss = nn.MSELoss()
        self.stft_loss = auraloss.freq.STFTLoss()
        # self.esrloss = auraloss.time.ESRLoss()
        self.dcloss = auraloss.time.DCLoss()
        self.recons_loss = self.waveform_loss = nn.SmoothL1Loss()

    def forward(self, x, recons):

        x = x[:, :, 100*150:]
        recons = recons[:, :, 100*150:]

        loss = torch.Tensor([0]).to(x.device)
        loss += self.stft_loss(x, recons)
        # loss += self.esrloss(x, recons)
        loss += self.dcloss(x, recons)
        loss += self.recons_loss(x, recons)

        return loss

        # loss = .001*self.spectral_loss(self.tf1(x), self.tf2(recons))
        # loss += self.waveform_loss(recons, x)


        



