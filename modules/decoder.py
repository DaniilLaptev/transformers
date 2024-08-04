import torch
import torch.nn as nn

from dataclasses import dataclass
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

@dataclass
class DecoderConfig:
    context:      int = 128
    vocab_size:   int = 50257
    num_layers:   int = 1
    attn_heads:   int = 4
    hidden_dim:   int = 64
    mlp_hidden:   int = 256
    mask_type:    str = 'causal'
    eos_token_id: int = 50256
    reduced_dim:  int = -1

class Decoder(nn.Module):
    def __init__(
        self, 
        context:      int = 128,
        vocab_size:   int = 50257,
        num_layers:   int = 1,
        attn_heads:   int = 4,
        hidden_dim:   int = 64,
        mlp_hidden:   int = 256,
        mask_type:    str = 'causal',
        eos_token_id: int = 50256,
        reduced_dim:  int = -1,
        config:       DecoderConfig = None,
        ):
        """GPT-like decoder module.
        
        To build up the model, you should provide either configuration as
        instance of DecoderConfig module or necessary parameters to build
        config internally.

        Parameters
        ----------
        context : int, optional
            Maximum length of input sequence, by default 128
        vocab_size : int, optional
            Number of tokens in tokenizer vocabulary, by default 50257
        num_layers : int, optional
            Number of stacked decoder layers, by default 1
        attn_heads : int, optional
            Number of attention heads in each decoder layer, by default 4
        hidden_dim : int, optional
            Embedding dimension, by default 64
        mlp_hidden : int, optional
            Dimensions of hidden layer in MLP (see DecoderLayer), by default 256
        mask_type : str, optional
            Type of mask to generate for each attention layer, by default 'causal'
        eos_token_id : int, optional
            Number of EOS token in tokenizer, by default 50256
        reduced_dim : int, optional
            If > -1, then embeddings first projected to reduced dimension, then to hidden dimension, which reduces number of parameters up to 25%, by default -1
        config : DecoderConfig, optional
            Configuration of decoder model, by default None
        """
        super().__init__()
        
        self.config = config
        if config is None:
            self.config = DecoderConfig(
                context=context,
                vocab_size=vocab_size,
                num_layers=num_layers,
                attn_heads=attn_heads,
                hidden_dim=hidden_dim,
                mlp_hidden=mlp_hidden,
                mask_type=mask_type,
                eos_token_id=eos_token_id,
                reduced_dim=reduced_dim
            )
        
        self.reduced = self.config.reduced_dim > -1
        
        if self.reduced:
            self.te = nn.Sequential(OrderedDict(
                emb = nn.Embedding(self.config.vocab_size, self.config.reduced_dim),
                prj = nn.Linear(self.config.reduced_dim, self.config.hidden_dim, bias = False)
            ))
        else:
            self.te = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)
        
        self.pe = nn.Embedding(self.config.context, self.config.hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.num_layers)])
        self.ln = nn.LayerNorm(self.config.hidden_dim)
        
        if self.reduced:
            self.lmhead = nn.Sequential(OrderedDict(
                prj = nn.Linear(self.config.hidden_dim, self.config.reduced_dim, bias = False),
                emb = nn.Linear(self.config.reduced_dim, self.config.vocab_size, bias = False),
            ))
            self.te.emb.weight = self.lmhead.emb.weight
        else:
            self.lmhead = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias = False)
            self.te.weight = self.lmhead.weight
        
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.context
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.pe(pos) + self.te(idx)
        
        for block in self.layers:
            x = block(x)
        x = self.ln(x)
        
        logits = self.lmhead(x)
        return logits
    
    def generate(self, inp, max_new_tokens = 10, topk = 50):
        """Generates output from inputs.

        Parameters
        ----------
        inp : torch.Tensor,
            Input tokens represented as tensors.
        max_new_tokens : int, optional
            Maximum number of generated tokens, by default 10
        topk : int, optional
            Top-k for topk generation algorithm, by default 50

        Returns
        -------
        dict('input', 'answer')
            Dictionary with tensors as values.
        """
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
                
                if sum(answ[:,-1] == self.config.eos_token_id) == B:
                    break
        
        return {
            'input': inp,
            'answer': answ[:, T:]
        }
        
