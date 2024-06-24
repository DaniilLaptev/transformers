
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()

        pe = torch.zeros(maxlen, d)
        position = torch.arange(maxlen).view(-1, 1)
        wk = 1 / math.log(10000.0) ** (torch.arange(0, d, 2) / d)

        pe[:, 0::2] = torch.sin(position * wk)
        pe[:, 1::2] = torch.cos(position * wk)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(-1)]
    
class SegmentEmbedding:
    pass
