
import torch
import torch.nn as nn

from .embeddings import SinusoidalPositionEmbedding, SegmentEmbedding
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d, h, dropout = 0.5):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d, h)
        self.linear = nn.Linear(d, d)
        self.layernorm = nn.LayerNorm(d)
        self.dropout = nn.Dropout1d(dropout)
        
    def forward(self, x):
        x = self.layernorm(x + self.attention(x, x, x))
        x = self.layernorm(x + self.dropout(self.linear(x)))
        return x

class Encoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        num_layers = 1, 
        attn_heads = 4,
        hidden_dim = 64, 
        max_length = 32,
        dropout = 0.5):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.te = nn.Embedding(vocab_size, hidden_dim)
        self.pe = SinusoidalPositionEmbedding(hidden_dim)
        # self.se = SegmentEmbedding(hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, attn_heads, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.te(x) + self.pe(x)# + self.se(x)
        for layer in self.layers:
            x = layer(x)
        return x