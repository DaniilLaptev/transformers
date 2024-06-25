
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super(MultiHeadAttention, self).__init__()
        
        assert d % h == 0
        
        self.d = d
        self.h = h
        self.r = d // h
        self.wqs = nn.Parameter(torch.rand((h, 1, d, self.r)))
        self.wks = nn.Parameter(torch.rand((h, 1, d, self.r)))
        self.wvs = nn.Parameter(torch.rand((h, 1, d, self.r)))
        self.wo  = nn.Parameter(torch.rand((d, d)))
        
    def forward(self, q, k, v, mask = None):
        
        batch_size = q.size(0)
        
        Q = q @ self.wqs
        K = k @ self.wks
        V = v @ self.wvs
        
        logits = Q @ K.transpose(2, 3) / math.sqrt(self.r)
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, 1e-10)
        
        score = torch.softmax(logits, dim = -1)
        values = (score @ V).reshape(batch_size, -1, self.h * self.r)
        output = values @ self.wo
        
        return output