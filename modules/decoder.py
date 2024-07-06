
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
        
    def forward(self, x, memory, mask):
        x = self.layernorm1(x + self.attention(memory, memory, x, mask))
        x = self.layernorm2(x + self.linear2(self.gelu(self.linear1(x))))
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(
        self, 
        tokenizer,
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
        self.tokenizer = tokenizer
        self.te = nn.Embedding(tokenizer.vocab_size, hidden_dim)
        self.pe = SinusoidalPositionEmbedding(context, hidden_dim)
        self.linear = nn.Linear(hidden_dim, tokenizer.vocab_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, attn_heads, context, dropout) 
            for _ in range(num_layers)
        ])
        self.context = context
        self.mask_type = mask_type
        
    def _get_mask(self, x, attn):
        
        seqlen = x.shape[1]
        
        mask = torch.ones(seqlen, seqlen)
        
        if attn is not None:
            attn = torch.repeat_interleave(attn, seqlen, dim = -1).reshape(-1, seqlen, seqlen)
        else:
            attn = 1
        
        if self.mask_type == 'none':
            return attn * mask
        if self.mask_type == 'causal':
            return attn * (torch.tril(mask) == 1)
        
    def encode(self, texts):
        encoded = self.tokenizer(texts, padding = True, return_tensors='pt')
        return encoded['input_ids'], encoded['attention_mask']
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def generate(self, prompt, max_new_tokens=10 , p : float = 0.7):

        answer = ''
        
        for i in range(max_new_tokens):
            x, attn = self.encode(prompt + answer)

            next_token_probs = self.forward(x, attn = attn)[0, -1, :]

            sorted_logits, sorted_indices = torch.sort(next_token_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > p
            
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            new_logits = torch.clone(next_token_probs)
            new_logits[indices_to_remove] = float('-inf')

            scores = torch.softmax(new_logits, dim=-1)
            id = torch.multinomial(scores, num_samples=1).item()
            
            # convert to token and add new token to text
            answer += self.decode(id)
            
            if self.tokenizer.eos_token_id == id:
                break

        return answer
        
    def forward(self, x, attn = None, memory = None, mask = None):
        
        if isinstance(x, str):
            x, attn = self.encode(x)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            attn = attn.unsqueeze(0)
            
        if mask is None:
            mask = self._get_mask(x, attn)
        
        x = self.te(x)
        
        if memory is None:
            memory = x
            
        for layer in self.layers:
            x = layer(x, memory, mask)
        
        return x