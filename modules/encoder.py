
import torch
import torch.nn as nn

from .embeddings import SinusoidalPositionEmbedding
from .attention import MultiheadSelfAttention

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, attn_heads, ff = 128, dropout = 0.1):
        """Default encoder layer.

        Parameters
        ----------
        hidden_dim : int
            Number of dimensions of encoder embeddings.
        attn_heads : int
            Number of attention heads in MultiHeadAttention layer.
        ff : int, optional
            Hidden dimension of feed-forward layer, by default 128
        dropout : float, optional
            Probability of dropout applied to output, by default 0.1
        """
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, attn_heads)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ff, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layernorm(x + self.attention(x, x, x))
        x = self.layernorm(x + self.linear2(self.gelu(self.linear1(x))))
        return self.dropout(x)
        
class Encoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        num_layers = 1,
        attn_heads = 4,
        hidden_dim = 64,
        context = 128,
        ):
        """Text encoder without pre-trained weights. 

        Parameters
        ----------
        vocab_size : int
            Size of vocabulary of tokenizer.
        num_layers : int, optional
            Number of encoder layers, by default 1
        attn_heads : int, optional
            Number of attention head in each encoder layer, by default 4
        hidden_dim : int, optional
            Dimensionality of model embeddings, by default 64
        context : int, optional
            Maximum number of tokens that model can work with, by default 128
        """
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.te = nn.Embedding(vocab_size, hidden_dim)
        self.pe = SinusoidalPositionEmbedding(hidden_dim, context)
        # self.se = SegmentEmbedding(hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, attn_heads) 
            for _ in range(num_layers)
        ])
        self.context = context
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_normal_(p)
        
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
    def __init__(
        self,
        num_classes,
        encoder = None,
        vocab_size = None,
        num_layers = 1,
        attn_heads = 4,
        hidden_dim = 64,
        context = 128,
        learn='all',
        ):
        """Text encoder with classification head. By default, embedding 
        of first token is considered as embedding of input sequence.
        
        Returns softmax outputs.

        Parameters
        ----------
        num_classes : int
            Number of classifier outputs.
        encoder : Encoder, optional
            If not None, by default None
        vocab_size : int
            Size of vocabulary of tokenizer.
        num_layers : int, optional
            Number of encoder layers, by default 1
        attn_heads : int, optional
            Number of attention head in each encoder layer, by default 4
        hidden_dim : int, optional
            Dimensionality of model embeddings, by default 64
        context : int, optional
            Maximum number of tokens that model can work with, by default 128
        learn : str, optional
            If 'classifier', then encoder weights are frozen, by default 'all'
        """
        super(EncoderForSequenceClassification, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Encoder(
                vocab_size=vocab_size,
                num_layers=num_layers,
                attn_heads=attn_heads,
                hidden_dim=hidden_dim,
                context=context
            )
        
        self.linear = nn.Linear(self.encoder.hidden_dim, num_classes)

        if learn == 'classifier':
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        x = self.encoder(x)[:, 0]
        x = self.linear(x)
        
        return torch.softmax(x, dim = -1)