import einops
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
    x = x - x_max  # subtract maximum for numerical stability
    x_exp = torch.exp(x)
    x_exp_denom = x_exp.sum(dim=dim, keepdim=True).expand_as(x)
    return x_exp / x_exp_denom


def cross_entropy_loss(
    predicted_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Handle any additional batch dimensions and return the average across the batch.
    As with section 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.
    """
    # softmax_logits = softmax(predicted_logits, dim=-1) # [batch_size, seq_len, vocab_size]
    targets = einops.repeat(targets, "... -> ... 1")
    target_logits = torch.gather(
        predicted_logits, dim=-1, index=targets
    )  # the predicted logits for the targets; [batch_size, seq_len, 1]
    # the log sum exp trick
    max_logits = torch.max(
        predicted_logits, dim=-1, keepdim=True
    ).values  # [batch_size, seq_len, 1]
    denom = torch.log(
        torch.exp(predicted_logits - max_logits).sum(dim=-1, keepdim=True)
    )  # [batch_size, seq_len, 1]

    denom = denom + max_logits  # [batch_size, seq_len, 1]
    loss = target_logits - denom  # [batch_size, seq_len, 1]
    loss *= -1  # [batch_size, seq_len, 1]
    return loss.mean()
