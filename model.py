import torch
import torch.nn as nn
from torch.nn import functional as F

from Nets import simpleNet

class allNet(nn.Module):
    def __init__(self, encoder:nn.Module, quantizer:nn.Module, decoder:nn.Module):
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.quantizer(x)
        x = self.decoder(x)
        return x