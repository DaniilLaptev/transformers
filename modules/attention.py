
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, attn_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % attn_heads == 0
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.r = hidden_dim // attn_heads
        
        self.wqs = nn.Linear(hidden_dim, attn_heads)
        self.wks = nn.Linear(hidden_dim, attn_heads)
        self.wvs = nn.Linear(hidden_dim, hidden_dim)
        self.wo  = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask = None):
        
        batch_size = q.size(0)
        
        Q = self.wqs(q)
        K = self.wks(k)
        V = self.wvs(k)
    
        logits = Q @ K.transpose(-2, -1) / math.sqrt(self.r)
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, 1e-10)
        
        score = torch.softmax(logits, dim = -1)
        values = (score @ V)
        values = values.reshape(batch_size, -1, self.attn_heads * self.r)
        output = self.dropout(self.wo(values))
        
        return output