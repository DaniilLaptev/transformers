
import torch
import torch.nn as nn

from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d, h, dropout = 0.5, mask = None):
        super(DecoderLayer, self).__init__()
        
        self.d = d
        self.h = h
        self.attention = MultiHeadAttention(d, h)
        self.layernorm1 = nn.LayerNorm(d)
        self.layernorm2 = nn.LayerNorm(d)
        self.linear = nn.Linear(d, d)
        self.dropout = nn.Dropout1d(dropout)
        self.mask = mask
        
    def forward(self, x):
        x = self.layernorm1(x + self.attention(x, x, x, self.mask))
        x = self.layernorm2(x + self.dropout(self.linear(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, layers, vocab_size, d, h):
        super(Decoder, self).__init__()
        
        self.d = d
        self.h = h
        self.linear = nn.Linear(d, vocab_size)
        self.layers = nn.ModuleList([
            DecoderLayer(d, h) for _ in range(layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        logits = self.linear(x)
        return logits