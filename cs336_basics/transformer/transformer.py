import torch
from .basic_modules import MultiHeadSelfAttention, SwiGLU, RMSNorm, Linear, Embedding

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

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int=2048,
        theta: float=1e5,
        device: torch.device='cpu',
        dtype: torch.dtype=torch.float32,
    ):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.device=device
        self.dtype=dtype
        
        # initialize sub-layers
        self.token_embedding_layer = Embedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.transformer_blocks = [Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            device=self.device,
            dtype=self.dtype,
        ) for _ in range(self.num_layers)]
        self.final_norm = RMSNorm(
            d_model=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.lm_head_layer = Linear(
            in_features=self.d_model,
            out_features=self.vocab_size,
            device=self.device,
            dtype=self.dtype,
        )
        
    def load_params(
        self,
        state_dict: dict,
    ):
        # TODO later for real use
        raise NotImplementedError
    
    def load_params_for_test(
        self,
        state_dict: dict,
    ):
        """
            weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        """
        self.token_embedding_layer.load_param(
            EmbeddingMatrix=state_dict['token_embeddings.weight']
        )
        for i in range(self.num_layers):
            self.transformer_blocks[i].load_param_for_test(
                state_dict={
                    'ln1.weight': state_dict[f'layers.{i}.ln1.weight'],
                    'attn.q_proj.weight': state_dict[f'layers.{i}.attn.q_proj.weight'],
                    'attn.k_proj.weight': state_dict[f'layers.{i}.attn.k_proj.weight'],
                    'attn.v_proj.weight': state_dict[f'layers.{i}.attn.v_proj.weight'],
                    'attn.output_proj.weight': state_dict[f'layers.{i}.attn.output_proj.weight'],
                    'ln2.weight': state_dict[f'layers.{i}.ln2.weight'],
                    'ffn.w1.weight': state_dict[f'layers.{i}.ffn.w1.weight'],
                    'ffn.w2.weight': state_dict[f'layers.{i}.ffn.w2.weight'],
                    'ffn.w3.weight': state_dict[f'layers.{i}.ffn.w3.weight'],
                }
            )
        self.final_norm.load_param(
            gain_param=state_dict['ln_final.weight']
        )
        self.lm_head_layer.load_param(
            W=state_dict['lm_head.weight']
        )
        
    def forward(
        self,
        token_ids: torch.LongTensor, # ... seq_len
    ):
        # ... seq_len d_model
        res = self.token_embedding_layer.forward(token_ids=token_ids) 
        for layer in self.transformer_blocks:
            res = layer.forward(res)
        res = self.final_norm.forward(res)
        
        # ... seq_len vocab_size
        return self.lm_head_layer.forward(res)