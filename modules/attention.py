
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, attn_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % attn_heads == 0
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.r = hidden_dim // attn_heads
        self.wqs = nn.Parameter(torch.rand((attn_heads, 1, hidden_dim, self.r)))
        self.wks = nn.Parameter(torch.rand((attn_heads, 1, hidden_dim, self.r)))
        self.wvs = nn.Parameter(torch.rand((attn_heads, 1, hidden_dim, self.r)))
        self.wo  = nn.Parameter(torch.rand((hidden_dim, hidden_dim)))

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()

        for w in [self.wqs, self.wks, self.wvs, self.wo]:
            nn.init.xavier_uniform_(w)
        
    def forward(self, q, k, v, mask = None):
        
        batch_size = q.size(0)
        
        Q = q @ self.wqs
        K = k @ self.wks
        V = v @ self.wvs
        
        logits = Q @ K.transpose(2, 3) / math.sqrt(self.r)
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, 1e-10)
        
        score = torch.softmax(logits, dim = -1)
        values = (score @ V).reshape(batch_size, -1, self.attn_heads * self.r)
        output = self.layernorm(self.gelu(values @ self.wo))
        
        return output