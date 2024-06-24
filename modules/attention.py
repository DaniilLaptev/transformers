
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
        self.wqs = nn.Parameter(torch.rand((h, d, self.r)))
        self.wks = nn.Parameter(torch.rand((h, d, self.r)))
        self.wvs = nn.Parameter(torch.rand((h, d, self.r)))
        self.wo  = nn.Parameter(torch.rand((d, d)))
        
    def forward(self, q, k, v, mask = None):
        
        batch_size, n_tokens, _ = q.shape
        
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        
        Q = torch.matmul(q, self.wqs)
        K = torch.matmul(k, self.wks)
        V = torch.matmul(v, self.wvs)
        
        logits = Q @ K.transpose(-2, -1) / math.sqrt(self.r)
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, 1e-10)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.r)
        score = torch.softmax(logits, dim = -1)
        values = torch.matmul(score, V).reshape(batch_size, n_tokens, self.h * self.r)
        output = torch.matmul(values, self.wo)
        
        return output