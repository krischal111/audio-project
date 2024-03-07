import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvLayer(nn.Module):
    def __init__(self, out_channels, k_size, stride=1, padding=0,  dilation=1):
        '''
        '''

        self.Elu1 = nn.ELU()
        self.Conv1 = nn.LazyConv1d(out_channels, k_size, stride, padding, dilation)
        self.Bnorm1 = nn.BatchNorm1d()
    
    def forward(self, x):
        x = self.Elu1(x)
        x = self.Conv1(x)
        x = self.Bnorm1(x)
        return x

class Down(nn.Module):
    def __init__(self, out_channels, k_size, stride):
        self.Conv1 = ConvLayer(out_channels, k_size, dilation=1)
        self.Conv2 = ConvLayer(out_channels, k_size, dilation=2)
        self.Conv3 = ConvLayer(out_channels, k_size, dilation=4)
        self.Conv4 = ConvLayer(out_channels, k_size, dilation=8)
        self.Conv5 = ConvLayer(out_channels, k_size, dilation=16)
        self.Conv6 = ConvLayer(out_channels, k_size, dilation=32)
        self.Pooling = nn.AvgPool1d(k_size, k_size)
        # I want something different
        # I want maxpooling in each dilation, 
        # That will go into next layer kind of situation.
        # Like a frequency. Idk what I am talking. Or, I can't explain in words
        # I have to code it, i guess.
    
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)
        x = self.Pooling(x)
        return x

class Quantizer(nn.Module):
    def __init__(self):
        self.placeholder = nn.Identity()

    def forward(self, x):
        return self.placeholder(x)

class Up(nn.Module):
    def __init__(self, out_channels, k_size, stride):
        self.Upsample = nn.Upsample(scale_factor=k_size, mode='linear')
        self.Conv6 = ConvLayer(out_channels, k_size, dilation=32)
        self.Conv5 = ConvLayer(out_channels, k_size, dilation=16)
        self.Conv4 = ConvLayer(out_channels, k_size, dilation=8)
        self.Conv3 = ConvLayer(out_channels, k_size, dilation=4)
        self.Conv2 = ConvLayer(out_channels, k_size, dilation=2)
        self.Conv1 = ConvLayer(out_channels, k_size, dilation=1)
        # I want something different
        # I want maxpooling in each dilation, 
        # That will go into next layer kind of situation.
        # Like a frequency. Idk what I am talking. Or, I can't explain in words
        # I have to code it, i guess.
    
    def forward(self, x):
        x = self.Upsample(x)
        x = self.Conv6(x)
        x = self.Conv5(x)
        x = self.Conv4(x)
        x = self.Conv3(x)
        x = self.Conv2(x)
        x = self.Conv1(x)
        return x

class Encoder(nn.Module):
    def __init__(self, steps, oChannels):
        self.encoder = nn.Sequential(*list(Up(min(4*2**step, 32), 16) for step in range(steps) ))
        # Up(4, 16)
        # Up(8, 16)
        # Up(16, 16)
        # Up(32, 16)
        self.final = nn.LazyConv1d(oChannels, 1)
        #
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.final(x)
        return x

class Decoder(nn.Module):
    def __init__(self, steps, oChannels):
        self.decoder = nn.Sequential(*list(Down(32/min(2*2**step, 32), 16) for step in range(steps) ))
        # Up(16, 16)
        # Up(8, 16)
        # Up(4, 16)
        # Up(1, 16)
        self.final = nn.LazyConv1d(oChannels, 1)
    
    def forward(self, x):
        self.decoder(x)
        x = self.final(x)
        return x








