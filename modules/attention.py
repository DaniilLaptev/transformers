import torch
import torch.nn as nn
import math

class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.attn_heads == 0
        
        self.attn = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.hidden_dim = config.hidden_dim
        self.attn_heads = config.attn_heads
        
        self.register_buffer('bias', self._get_mask(config))
         
    def _get_mask(self, config):
        context = config.context
        if config.mask_type == 'causal':
            return torch.tril(torch.ones(context, context)).view(1, 1, context, context)
        else:
            raise NotImplementedError(f'Mask type {config.mask_type} is not implemented.')
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=2)
        # We transform q, k, v to have shape [B, num heads, T, head size]:
        q = q.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2)
        k = k.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2) 
        v = v.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2)
        
        # Logits will have shape [B, num heads, T, T]
        logits = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))
        logits = logits.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        out = logits.softmax(dim=-1) @ v # [B, num heads, T, head size]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.linear(out)
        
        return out