import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

from .embedding import Embedding

from .linear import Linear
from .rmsnorm import RMSNorm

from .rotary_positional_embedding import RotaryPositionalEmbedding
from .self_attention import MultiHeadSelfAttention, ScaledDotProductAttention
from .swiglu import SwiGLU
from .util_layers import softmax


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a Transformer block. This function should accept the following parameters:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            d_k=self.d_k, max_seq_len=max_seq_len, theta=theta, device=device
        )

        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.mha = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, device=device, dtype=dtype
        )

        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """
        # first half: y = x + MultiHeadSelfAttention(RMSNorm(x))
        x_norm = self.rms1(x)
        x = x + self.mha.forward(x_norm, self.rope)
        print("tlu7 ... x.shape after 1st half of transformerblock", x.shape)
        # second half: y = x + ffn(RMSNorm(x))
        x_norm_2 = self.rms2(x)
        x = x + self.swiglu(x_norm_2)
        print("tlu7 ... x.shape after 2nd half of transformerblock", x.shape)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a Transformer model. This function should accept the following parameters:
        vocab_size: The size of the vocabulary, necessary for determining the dimensionality of the token
        context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.

        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=max_seq_len,
                    device=device,
                    dtype=dtype,
                )
            )
        self.rms_norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.llm_output = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length)
        =>
        (batch_size, sequence_length, vocab_size) for next token prediction probability
        """
        # embedding lookup
        x = self.embedding(x)
        print("tlu7 ... x.shape after embedding", x.shape)
        # transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        print("tlu7 ... x.shape after transformer", x.shape)
        # final rms norm
        x = self.rms_norm_final(x)
        print("tlu7 ... x.shape after rms_norm_final", x.shape)
        # final linear layer
        x = self.llm_output(x)
        # x = softmax(x, dim=-1)
        print("tlu7 ... x.shape after entire transformer", x.shape)
        return x
