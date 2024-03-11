import torch.nn as nn
from torch.nn import functional as F

# Causal Equation:
# Padding + Stride = (kernel_size -1) * dilation + 1
class LazyCausalConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, stride = 1, dilation=1):
        super().__init__()
        self.causal_padding_left = (dilation) * (kernel_size-1) + 1 - stride
        self.lazyLayer = nn.LazyConv1d(out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        return self.lazyLayer(F.pad(x, [self.causal_padding_left, 0])) #, self.weight, self.bias)

class CausalPoolingDoubleFactor(nn.Module):
    def __init__(self, factor):
        self.poolingLayer = nn.AvgPool1d(kernel_size=2*factor, stride=factor)
        self.causal_padding_right = factor
    def forward(self, x):
        return self.poolingLayer(F.pad(x, [self.causal_padding, 0]))