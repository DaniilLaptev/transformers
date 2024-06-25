
import torch
import torch.nn as nn
import math
    
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_length):
        super(SinusoidalPositionEmbedding, self).__init__()

        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(max_length).view(-1, 1)
        wk = 1 / math.log(10000.0) ** (torch.arange(0, hidden_dim, 2) / hidden_dim)

        pe[:, 0::2] = torch.sin(position * wk)
        pe[:, 1::2] = torch.cos(position * wk)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(-1)]
    
class SegmentEmbedding:
    pass
