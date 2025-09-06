import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax function along a given dimension.
    """
    x_max, _ = x.max(dim=dim, keepdim=True)
    x_max.expand_as(x)
    x = x - x_max # subtract maximum for numerical stability
    x_exp = torch.exp(x)
    x_exp_denom = x_exp.sum(dim=dim, keepdim=True).expand_as(x)
    return x_exp / x_exp_denom
