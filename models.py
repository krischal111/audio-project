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
    
    def train_forward(self, x):
        x = self.encoder(x)
        x , q_loss = self.quantizer(x)
        x = self.decoder(x)
        return x, q_loss
    
    def forward(self, x):
        x = self.encoder(x)
        # print("Before quantiziing", x.shape)
        x , _ = self.quantizer(x)
        # print("After quantizing", x.shape)
        x = self.decoder(x)
        return x

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=3 * 1e-4, betas=(0.5, 0.9))
