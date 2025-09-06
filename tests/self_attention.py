import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

from .rotary_positional_embedding import RotaryPositionalEmbedding

from .util_layers import softmax


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        """
        Construct a scaled dot product attention module. This function should accept the following parameters:
        num_heads: int Number of attention heads
        dropout: float Dropout probability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Given the keys, queries, and values, return the output of the scaled dot product attention.
        This function should accept the following parameters:
        queries: (batch_size, ..., seq_len, d_k)
        keys: (batch_size, ..., seq_len, d_k)
        values: (batch_size, ..., seq_len, d_v)
        mask: (seq_len, seq_len)  Mask for the attention weights; boolean
        =>
        (batch_size, ..., seq_len, d_v)
        """
        d_k = keys.shape[-1]
        Q = queries
        K = keys
        Q_K = einsum(Q, K, "b ... l1 k, b ... l2 k -> b ... l1 l2")
        Q_K = Q_K / np.sqrt(d_k)
        if mask is not None:
            mask = torch.where(mask, torch.tensor(0.0), torch.tensor(float("inf")))
            mask = mask.expand_as(Q_K)
            # subtract infinity from where mask is false => no attention to these keys
            Q_K = Q_K - mask  # (batch_size, ..., seq_len, seq_len)
        attention = softmax(Q_K, dim=-1)  # [batch_size, ..., seq_len, seq_len]
        return einsum(attention, values, "b ... l1 l2, b ... l2 d_v -> b ... l1 d_v")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        Construct a multi-head self-attention module. This function should accept the following parameters:
        num_heads: int Number of attention heads
        dropout: float Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        q_proj_weight = torch.empty(
            (self.d_model, self.d_model), dtype=dtype, device=device
        )
        k_proj_weight = torch.empty(
            (self.d_model, self.d_model), dtype=dtype, device=device
        )
        v_proj_weight = torch.empty(
            (self.d_model, self.d_model), dtype=dtype, device=device
        )
        o_proj_weight = torch.empty(
            (self.d_model, self.d_model), dtype=dtype, device=device
        )
        self.q_proj_weight = torch.nn.Parameter(q_proj_weight)
        self.k_proj_weight = torch.nn.Parameter(k_proj_weight)
        self.v_proj_weight = torch.nn.Parameter(v_proj_weight)
        self.o_proj_weight = torch.nn.Parameter(o_proj_weight)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Given the input x, return the output of the multi-head self-attention.
        This function should accept the following parameters:
        x: "... sequence_length d_in"
        =>
        "... sequence_length d_out"
        """
        # MultiHeadSelfAttention(x) = Wo MultiHead(Wq@x, Wk@x, Wv@x)
        # MultiHead(Q, K, V ) = Concat(head_1, . . . , head_h)
        # for head_i = Attention(Qi, Ki, Vi)
        # print("tlu7... x.shape", x.shape)
        # print("tlu7... self.q_proj_weight", self.q_proj_weight)
        # print("tlu7... self.k_proj_weight", self.k_proj_weight)
        # print("tlu7... self.v_proj_weight", self.v_proj_weight)
        # print("tlu7... self.o_proj_weight", self.o_proj_weight)
        # print("tlu7... self.d_k", self.d_k)
        # print("tlu7... self.d_v", self.d_v)
        # print("tlu7... self.num_heads", self.num_heads)
        # print("tlu7... self.d_model", self.d_model)
        Wq_x = einsum(
            self.q_proj_weight, x, "d_world d_in, ... l d_in -> ... l d_world"
        )
        Wk_x = einsum(
            self.k_proj_weight, x, "d_world d_in, ... l d_in -> ... l d_world"
        )
        Wv_x = einsum(
            self.v_proj_weight, x, "d_world d_in, ... l d_in ->  ... l d_world"
        )
        seq_len = x.shape[-2]
        # print("tlu7... seq_len", seq_len)
        attentions = []
        for i in range(self.num_heads):
            Q_i = Wq_x[..., i * self.d_k : (i + 1) * self.d_k]
            K_i = Wk_x[..., i * self.d_k : (i + 1) * self.d_k]
            V_i = Wv_x[..., i * self.d_v : (i + 1) * self.d_v]
            if rope is not None:
                Q_i = rope(Q_i, token_positions)
                K_i = rope(K_i, token_positions)
            mask = (1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).bool()
            attention_i = ScaledDotProductAttention()(
                queries=Q_i, keys=K_i, values=V_i, mask=mask
            )
            attentions.append(attention_i)
        attentions = torch.cat(
            attentions, dim=-1
        )  # (batch_size, ..., seq_len, d_model)
        return einsum(
            self.o_proj_weight, attentions, "d_model d_v, ... l d_v -> ... l d_model"
        )
