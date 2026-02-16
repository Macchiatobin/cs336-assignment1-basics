import torch
from einops import einsum, rearrange, repeat


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
        
    def load_param(
        self, 
        W: torch.Tensor
    ):
        assert W.shape == (self.d_model, self.in_dim), f"Expected shape {(self.d_model, self.in_dim)}, but got {W.shape}"
        with torch.no_grad():
            self.W = torch.nn.Parameter(W.to(self.dtype).to(self.device))
    
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
        
    def load_param(
        self,
        EmbeddingMatrix: torch.Tensor
    ):
        assert EmbeddingMatrix.shape == (self.vocab_size, self.d_model), f"Expected shape {(self.vocab_size, self.d_model)}, but got {EmbeddingMatrix.shape}"
        with torch.no_grad():
            self.EmbeddingMatrix = torch.nn.Parameter(EmbeddingMatrix.to(self.dtype).to(self.device)
    )
    
    def forward(
        self,
        token_ids: torch.Tensor # [bs, seq_len]
    ) -> torch.Tensor:
        return self.EmbeddingMatrix[token_ids] # pure look-up here
    
class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.device=device or 'cpu'
        self.dtype=dtype or torch.float32
        
        # initialize as ones, according to the given setting
        self.gain_param = torch.nn.Parameter(torch.ones((self.d_model,), dtype=self.dtype))
        
    def load_param(
        self,
        gain_param: torch.Tensor
    ):
        assert gain_param.shape == (self.d_model,), f"Expected shape {(self.d_model,)}, but got {gain_param.shape}"
        with torch.no_grad():
            self.gain_param = torch.nn.Parameter(gain_param.to(self.dtype).to(self.device))
        
    def forward(
        self,
        x: torch.Tensor # (... d_model)
    ):
        original_dtype = x.dtype
        x = x.to(torch.float32)
        
        RMS = torch.sqrt(self.eps + (einsum(x, x, '... i, ... i -> ...') / self.d_model)) # sum over a_i^2
        RMS = rearrange(RMS, '... -> ... 1') # unsqueeze for broadcasting
        result = einsum(x, self.gain_param, '... i, i -> ... i') / RMS
        
        return result.to(original_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.check_d_ff_d_model()
        self.device=device or 'cpu'
        self.dtype=dtype or torch.float32
        
        # initialize params according to the given setting
        self.W1_layer = Linear(
            in_features=self.d_model,
            out_features=self.d_ff,
        )
        self.W2_layer = Linear(
            in_features=self.d_ff,
            out_features=self.d_model,
        )
        self.W3_layer = Linear(
            in_features=self.d_model,
            out_features=self.d_ff,
        )
     
    def check_d_ff_d_model(
        self
    ):
        """Very Naive, might need improvement in the future"""
        # check whether d_ff and d_model is initialized
        assert self.d_ff and self.d_model, 'Params not initialized!'
        
        # check whether d_ff approximately equals to (8/3) * d_model
        expected_d_ff = (8 / 3) * self.d_model
        assert abs(self.d_ff - expected_d_ff) <= 1e2, f"Expected d_ff to be approximately {(8 / 3) * self.d_model}, but got {self.d_ff}"
            
        
    def load_param(
        self,
        W1: torch.Tensor | None,
        W2: torch.Tensor | None,
        W3: torch.Tensor | None,
    ):
        if W1 is not None:
            self.W1_layer.load_param(W1)
        if W2 is not None:
            self.W2_layer.load_param(W2)
        if W3 is not None:
            self.W3_layer.load_param(W3)
    
    def forward(
        self,
        x: torch.Tensor, # (... d_model)
    ) -> torch.Tensor:
        SiLU_input = self.W1_layer.forward(x) # ... d_ff
        SiLU_output = SiLU_input * torch.sigmoid(SiLU_input)
        gate_input = self.W3_layer.forward(x) # ... d_ff
        gate_output = einsum(SiLU_output, gate_input, '... d_model, ... d_model -> ... d_model')
        return self.W2_layer.forward(gate_output)
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ):
        super().__init__()
        
        self.Theta=theta
        self.d_k=d_k # d_model | d_head
        self.max_seq_len=max_seq_len
        self.device: torch.device=device or 'cpu'
        self.dtype: torch.dtype=dtype or torch.float32
        
        positions = torch.arange(max_seq_len) # (max_seq_len,), range: 0, 1, ..., max_seq_len - 1
        starting_dims_for_pairs = torch.arange(0, self.d_k, 2) # (d_k / 2,), range: 0, 2, ..., d_k - 2
        freqs = 1.0 / (self.Theta ** (starting_dims_for_pairs.float() / self.d_k)) # (d_k / 2,)
        angles = einsum(positions, freqs, 'max_seq_len, half_d_k -> max_seq_len half_d_k') # \theta_0, \theta_1...
        angles = angles.repeat_interleave(2, dim=-1) # (max_seq_len, d_k)
        # \theta_0, \theta_0, \theta_1, \theta_1...
        
        # register rotation matrix (cos) as unlearnable tensor
        self.register_buffer(
            name='cos',
            tensor=torch.cos(angles).to(self.dtype),
            persistent=False,
        )
        self.register_buffer(
            name='sin',
            tensor=torch.sin(angles).to(self.dtype),
            persistent=False,
        )
        
    def forward(
        self,
        x: torch.Tensor, # ..., seq_len, d_k
        token_positions: torch.Tensor, # .., seq_len
    ):
        # slice token positions, (..., seq_len, d_k)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        x1 = x[..., 0::2] # (..., seq_len, d_k/2), even dims=x0, x2...
        x2 = x[..., 1::2] # (..., seq_len, d_k/2), odd dims=x1, x3...
        sin_part = torch.stack([-x2, x1], dim=-1) # (..., seq_len, d_k/2, 2) [[-x1, x0], [-x3, x2]... ]
        sin_part = sin_part.flatten(start_dim=-2) # (..., seq_len, d_k) [-x1, x0, -x3, x2...]
        
        """
            [
                x0 * cos - x1 * sin,
                x1 * cos + x0 * sin,
                x2 * cos - x3 * sin,
                x3 * cos + x2 * sin,
                ...
            ]
        """
        return (x * cos) + (sin_part * sin)