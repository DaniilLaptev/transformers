
import torch
import torch.nn as nn

from .embeddings import SinusoidalPositionEmbedding
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, attn_heads, dropout = 0.5):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, attn_heads)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout1d(dropout)
        
    def forward(self, x):
        x = self.layernorm(x + self.dropout(self.attention(x, x, x)))
        x = self.layernorm(x + self.linear2(self.gelu(self.linear1(x))))
        return x

class Encoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        num_layers = 1,
        attn_heads = 4,
        hidden_dim = 64,
        dropout = 0.5,
        context = 128,
        ):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.te = nn.Embedding(vocab_size, hidden_dim)
        self.pe = SinusoidalPositionEmbedding(hidden_dim, context)
        # self.se = SegmentEmbedding(hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, attn_heads, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.context = context
        
    def forward(self, x):
        
        return_squeezed = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            return_squeezed = True
        
        x = self.layernorm(self.te(x) + self.pe(x))# + self.se(x)
        for layer in self.layers:
            x = layer(x)
            
        return x.squeeze(0) if return_squeezed else x
    
class EncoderForSequenceClassification(nn.Module):
    def __init__(self, encoder, num_classes, learn='all'):
        super(EncoderForSequenceClassification, self).__init__()

        self.encoder = encoder
        self.linear = nn.Linear(encoder.hidden_dim, num_classes)

        if learn == 'classifier':
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        x = self.encoder(x)[:, 0]
        x = self.linear(x)
        
        return torch.softmax(x, dim = -1)