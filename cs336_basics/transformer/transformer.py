import torch
from .basic_modules import MultiHeadSelfAttention, SwiGLU, RMSNorm

class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int=2048,
        theta: float=1e5,
        device: torch.device='cpu',
        dtype: torch.dtype=torch.float32,
    ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.device=device
        self.dtype=dtype
        
        # initialize sub-layers
        self.rms_norm1 = RMSNorm(
            d_model=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.causal_MHSA_with_rope = MultiHeadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            device=self.device,
            dtype=self.dtype
        )
        self.rms_norm2 = RMSNorm(
            d_model=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.swiglu = SwiGLU(
            d_model=self.d_model,
            d_ff=self.d_ff,
            device=self.device,
            dtype=self.dtype
        )
        
    def load_param(
        self,
        state_dict: dict,
    ):
        raise NotImplementedError # I will implement this later
    
    def load_param_for_test(
        self,
        state_dict: dict,
    ):
        """
        Hard-matching with pre-defined str names, for passing the test
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
    """
        self.rms_norm1.load_param(
            gain_param=state_dict['ln1.weight']
        )
        self.causal_MHSA_with_rope.load_param(
            W_Q=state_dict['attn.q_proj.weight'],
            W_K=state_dict['attn.k_proj.weight'],
            W_V=state_dict['attn.v_proj.weight'],
            W_O=state_dict['attn.output_proj.weight'],
        )
        self.rms_norm2.load_param(
            gain_param=state_dict['ln2.weight']
        )
        self.swiglu.load_param(
            W1=state_dict['ffn.w1.weight'],
            W2=state_dict['ffn.w2.weight'],
            W3=state_dict['ffn.w3.weight'],
        )
        
    def forward(
        self,
        x: torch.Tensor, # ... seq_len d_model
    ):
        seq_len = x.shape[-2]
        res_x = x + self.causal_MHSA_with_rope.forward(
            x=self.rms_norm1.forward(x),
            token_positions=torch.arange(seq_len), # tokens are in sequence's original order
        )
        return res_x + self.swiglu.forward(
            x=self.rms_norm2.forward(res_x),
        )