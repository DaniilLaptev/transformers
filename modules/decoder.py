import torch
import torch.nn as nn

from collections import OrderedDict

from .attention import MultiheadSelfAttention

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.mlp_hidden)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.mlp_hidden, config.hidden_dim)
    
    def forward(self, x):
        x = self.linear2(self.gelu(self.linear1(x)))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiheadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eos_token_id = config.eos_token_id
        
        if config.lora_embed:
            self.te = nn.Sequential(OrderedDict(
                emb = nn.Embedding(config.vocab_size, config.reduced_dim),
                prj = nn.Linear(config.reduced_dim, config.hidden_dim, bias = False)
            ))
        else:
            self.te = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.pe = nn.Embedding(config.context, config.hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.hidden_dim)
        
        if config.lora_embed:
            self.lmhead = nn.Sequential(OrderedDict(
                prj = nn.Linear(config.hidden_dim, config.reduced_dim, bias = False),
                emb = nn.Linear(config.reduced_dim, config.vocab_size, bias = False),
            ))
            self.te.emb.weight = self.lmhead.emb.weight
            # self.te.prj.weight = self.lmhead.prj.weight
        else:
            self.lmhead = nn.Linear(config.hidden_dim, config.vocab_size, bias = False)
            self.te.weight = self.lmhead.weight
        
        self.is_lora = config.lora_embed
        
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.context
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe = self.pe(pos)
        te = self.te(idx)
        x = pe + te
        
        for block in self.layers:
            x = block(x)
        x = self.ln(x)
        
        logits = self.lmhead(x)
        return logits
    
    def generate(self, inp, max_new_tokens = 10, topk = 50):
        answ = inp.clone()
        B, T = answ.size()

        while answ.size(1) < T + max_new_tokens:
            with torch.no_grad():
                logits = self.forward(answ)[:,-1,:]
                probs = logits.softmax(dim=-1)
                topk_p, topk_idx = torch.topk(probs, topk, dim=-1)
                idx = torch.multinomial(topk_p, 1)
                xcol = torch.gather(topk_idx, -1, idx)
                answ = torch.cat([answ, xcol], dim=-1)
                
                if sum(answ[:,-1] == self.eos_token_id) == B:
                    break
        
        return {
            'input': inp,
            'answer': answ[:, T:]
        }