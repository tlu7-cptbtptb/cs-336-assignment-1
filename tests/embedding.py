import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        embedding_lookup = torch.empty(
            (num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        torch.nn.init.trunc_normal_(embedding_lookup, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.embedding_lookup = torch.nn.Parameter(embedding_lookup)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [batch_size, seq_len] Tensor of token ids
        =>
        embeddings: [batch_size, seq_len, embedding_dim] Tensor of embeddings
        """
        one_hot = F.one_hot(token_ids, num_classes=self.num_embeddings).float()
        return einsum(one_hot, self.embedding_lookup, "... n, n d -> ... d")
