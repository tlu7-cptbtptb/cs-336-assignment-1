import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

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
