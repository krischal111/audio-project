import torch
import torch.nn as nn
from torch.nn import functional as F

from Nets import simpleNet

class allNet(nn.Module):
    def __init__(self, encoder:nn.Module, quantizer:nn.Module, decoder:nn.Module):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        # print("Before quantiziing", x.shape)
        x = self.quantizer(x)
        # print("After quantizing", x.shape)
        x = self.decoder(x)
        return x