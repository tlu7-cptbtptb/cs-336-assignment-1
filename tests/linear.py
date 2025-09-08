import numpy as np
import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weights = torch.empty((out_features, in_features), dtype=dtype, device=device)
        sigma = np.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            weights, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        self.weights = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in, out in-> ... out")
