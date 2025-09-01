import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, repeat


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        gains = torch.ones(d_model, dtype=dtype, device=device)
        self.gains = torch.nn.Parameter(gains)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        norm_squared = einops.reduce(x * x, "b s d -> b s 1", "sum")
        rms = (norm_squared / self.d_model + self.eps).sqrt()  # [b, s, 1]
        rms_expanded = einops.repeat(rms, "b s 1 -> b s d", d=self.d_model)
        gains_expanded = einops.repeat(
            self.gains, "d -> b s d", b=x.shape[0], s=x.shape[1]
        )
        result = x / rms_expanded * gains_expanded
        # Return the result in the original dtype
        return result.to(in_dtype)
