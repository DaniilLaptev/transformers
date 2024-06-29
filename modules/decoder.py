
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .embeddings import SinusoidalPositionEmbedding

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, attn_heads, ff = 128, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.attention = MultiHeadAttention(hidden_dim, attn_heads)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ff, hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, e, x, mask):
        x = self.layernorm1(x + self.attention(e, e, x, mask))
        x = self.layernorm2(x + self.linear2(self.gelu(self.linear1(x))))
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        num_layers = 1, 
        attn_heads = 4, 
        hidden_dim = 64,
        dropout = 0.5,
        context = 128,
        mask_type = 'causal'
        ):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.te = nn.Embedding(vocab_size, hidden_dim)
        self.pe = SinusoidalPositionEmbedding(hidden_dim, context)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, attn_heads, context, dropout) 
            for _ in range(num_layers)
        ])
        
        self.mask_type = mask_type
        self.mask = self._get_mask(mask_type)
        
    def _get_mask(self, mask_type):
        
        mask = torch.ones(self.context, self.context)
        
        if mask_type == 'none':
            return mask
        if mask_type == 'causal':
            return torch.triu(mask).bool()
        
    def forward(self, x, e = None):
        
        return_squeezed = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            return_squeezed = True
            
        if e is None:
            e = x
        
        x = self.te(x)
        for layer in self.layers:
            x = layer(x, e, self.causal_mask)
        logits = self.linear(x)
        
        return logits.squeeze(0) if return_squeezed else logits