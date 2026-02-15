import torch
import numpy as np
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
       self,
       in_features: int,
       out_features: int,
       device: torch.device=None,
       dtype: torch.dtype=None, 
    ):
        super().__init__()
        self.in_dim=in_features
        self.d_model=out_features
        self.dtype = dtype or torch.float32
        self.device = device or 'cpu'
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), dtype=self.dtype))
        
        # initialize forward parameter according to given setting
        init_std =torch.sqrt(
            torch.tensor(
                [2 / (in_features + out_features)], 
                dtype=self.dtype)
        )
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=init_std,
            a=-(3 * init_std),
            b=(3 * init_std)
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')

class Embedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.dtype = dtype or torch.float32
        self.device = device or 'cpu'
        self.EmbeddingMatrix = torch.nn.Parameter(torch.empty((self.vocab_size, self.d_model), dtype=self.dtype))
        
        # Initialize embedding matrix according to given setting
        torch.nn.init.trunc_normal_(
            self.EmbeddingMatrix,
            mean=0,
            std=1,
            a=-3,
            b=3
        )
    
    def forward(
        self,
        token_ids: torch.Tensor # [bs, seq_len]
    ) -> torch.Tensor:
        return self.EmbeddingMatrix[token_ids] # pure look-up here