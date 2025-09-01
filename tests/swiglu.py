import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, repeat


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, dim_ff=None, device=None, dtype=None):
        """
        Construct a SwiGLU module. This function should accept the following parameters:
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.dim_ff = None
        if dim_ff is None:
            dim_ff = 8 * d_model / 3
            dim_ff = round(dim_ff / 64) * 64
            self.dim_ff = dim_ff
        else:
            self.dim_ff = dim_ff
        # print("tlu7... dim_ff=", self.dim_ff)
        w1_weight = torch.empty((self.dim_ff, d_model), dtype=dtype, device=device)
        w2_weight = torch.empty((d_model, self.dim_ff), dtype=dtype, device=device)
        w3_weight = torch.empty((self.dim_ff, d_model), dtype=dtype, device=device)
        self.w1_weight = torch.nn.Parameter(w1_weight)
        self.w2_weight = torch.nn.Parameter(w2_weight)
        self.w3_weight = torch.nn.Parameter(w3_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """
        w1_x = einsum(self.w1_weight, x, "d_ff d, ... d -> ... d_ff")  # [b, s, d_ff]
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = einsum(self.w3_weight, x, "d_ff d, ... d -> ... d_ff")  # [b, s, d_ff]
        silu_w3_x = silu * w3_x
        result = einsum(silu_w3_x, self.w2_weight, "... d_ff, d d_ff -> ... d")
        return result
