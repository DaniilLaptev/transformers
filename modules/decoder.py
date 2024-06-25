
import torch
import torch.nn as nn

from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, attn_heads, context, dropout):
        super(DecoderLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.attention = MultiHeadAttention(hidden_dim, attn_heads)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout1d(dropout)
        
        mask = torch.triu(torch.ones(context, context)).bool()
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        x = self.layernorm1(x + self.attention(x, x, x, self.mask))
        x = self.layernorm2(x + self.dropout(self.linear(x)))
        return x

class Decoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        num_layers = 1, 
        attn_heads = 4, 
        hidden_dim = 64,
        dropout = 0.5,
        context = 128
        ):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.te = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, attn_heads, context, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        
        return_squeezed = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            return_squeezed = True
            
        x = self.te(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.linear(x)
        return logits.squeeze(0) if return_squeezed else logits