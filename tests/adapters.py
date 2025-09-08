from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import reduce
from multiprocessing import Manager, Process, Queue

from pathlib import Path
from typing import Any, BinaryIO, IO

import numpy.typing as npt

import regex as re
import torch
from jaxtyping import Float, Int

from torch import Tensor

from .embedding import Embedding

from .linear import Linear
from .rmsnorm import RMSNorm
from .rotary_positional_embedding import RotaryPositionalEmbedding
from .self_attention import MultiHeadSelfAttention, ScaledDotProductAttention
from .swiglu import SwiGLU
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerBlock
from .util_layers import cross_entropy_loss, softmax


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear_module = Linear(in_features=d_in, out_features=d_out)
    linear_module.load_state_dict({"weights": weights})
    return linear_module(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    embedding.load_state_dict({"embedding_lookup": weights})
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model=d_model, dim_ff=d_ff)

    swiglu.load_state_dict(
        {"w1_weight": w1_weight, "w2_weight": w2_weight, "w3_weight": w3_weight}
    )
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    attention_layer = ScaledDotProductAttention()
    return attention_layer(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    mha.load_state_dict(
        {
            "q_proj_weight": q_proj_weight,
            "k_proj_weight": k_proj_weight,
            "v_proj_weight": v_proj_weight,
            "o_proj_weight": o_proj_weight,
        }
    )
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope = RotaryPositionalEmbedding(
        theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len
    )
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    mha.load_state_dict(
        {
            "q_proj_weight": q_proj_weight,
            "k_proj_weight": k_proj_weight,
            "v_proj_weight": v_proj_weight,
            "o_proj_weight": o_proj_weight,
        }
    )
    print("tlu7.. token_positions, ", token_positions)
    return mha(in_features, rope, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
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
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    transformer = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        max_seq_len=max_seq_len,
    )

    transformer.rms1.load_state_dict({"gains": weights["ln1.weight"]})

    transformer.mha.load_state_dict(
        {
            "q_proj_weight": weights["attn.q_proj.weight"],
            "k_proj_weight": weights["attn.k_proj.weight"],
            "v_proj_weight": weights["attn.v_proj.weight"],
            "o_proj_weight": weights["attn.output_proj.weight"],
        }
    )

    transformer.rms2.load_state_dict({"gains": weights["ln2.weight"]})

    transformer.swiglu.load_state_dict(
        {
            "w1_weight": weights["ffn.w1.weight"],
            "w2_weight": weights["ffn.w2.weight"],
            "w3_weight": weights["ffn.w3.weight"],
        }
    )

    return transformer(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
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
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
        max_seq_len=context_length,
    )
    transformer_lm.embedding.load_state_dict(
        {"embedding_lookup": weights["token_embeddings.weight"]}
    )

    for i in range(num_layers):
        transformer_lm.transformer_blocks[i].mha.load_state_dict(
            {
                "q_proj_weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "k_proj_weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "v_proj_weight": weights[f"layers.{i}.attn.v_proj.weight"],
                "o_proj_weight": weights[f"layers.{i}.attn.output_proj.weight"],
            }
        )
        transformer_lm.transformer_blocks[i].rms1.load_state_dict(
            {"gains": weights[f"layers.{i}.ln1.weight"]}
        )
        transformer_lm.transformer_blocks[i].rms2.load_state_dict(
            {"gains": weights[f"layers.{i}.ln2.weight"]}
        )
        transformer_lm.transformer_blocks[i].swiglu.load_state_dict(
            {
                "w1_weight": weights[f"layers.{i}.ffn.w1.weight"],
                "w2_weight": weights[f"layers.{i}.ffn.w2.weight"],
                "w3_weight": weights[f"layers.{i}.ffn.w3.weight"],
            }
        )
    transformer_lm.rms_norm_final.load_state_dict({"gains": weights["ln_final.weight"]})
    transformer_lm.llm_output.load_state_dict({"weights": weights["lm_head.weight"]})
    return transformer_lm(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model=d_model, eps=eps)
    rmsnorm.load_state_dict({"gains": weights})
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy_loss(predicted_logits=inputs, targets=targets)


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def break_pretoken_to_adjacent_byte_pairs(pretoken: bytes) -> list[bytes]:
    """
    "queue" => b'queue' => [(qu), (ue), (eu), (ue)]
    """
    byte_pairs = []
    l = len(pretoken)
    if l == 1:
        return [pretoken]
    if l >= 2:
        for i in range(l - 1):
            byte_pairs.append(pretoken[i : i + 2])
    return byte_pairs


def get_stats(vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Summarize mapping from adjenct bytes to frequency
    vocab: {('l', 'o', 'w', 'er'): 4, ('s', 'l', 'o', 'w', 'er'): 3}
    =>
    {('l','o'): 4, ('o','w'):4, ('w', 'er'): 7, ...}
    """
    pairs = defaultdict(int)
    for symbols, freq in vocab.items():

        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def get_stats_fast(
    vocab: dict[tuple[bytes, ...], int],
    prev_stats: dict[tuple[bytes, bytes], int],
    new_pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, bytes], int]:
    """
    Summarize mapping from adjenct bytes to frequency
    vocab: {('l', 'o', 'w', 'er'): 4, ('s', 'l', 'o', 'w', 'er'): 3, ('b', 'e', 'e') : 10}
    new_pair: ('e', 'r')
    prev_stat: {('s', 'l'): 3, ('l','o'): 7, ('o','w'):7, ('w', 'e'): 7, ('e', 'r'): 7,
        ('b', 'e'): 10, ('e', 'e'): 10}
    =>
    {('l','o'): 4, ('o','w'):4, ('w', 'er'): 7, ...}
    """
    if not prev_stats:
        return get_stats(vocab)
    for symbols, freq in vocab.items():
        if new_pair[0] + new_pair[1] in symbols:  # worth updating
            prev_symbols: list[bytes] = []
            for symbol in symbols:
                if symbol == new_pair[0] + new_pair[1]:
                    prev_symbols.append(new_pair[0])
                    prev_symbols.append(new_pair[1])
                else:
                    prev_symbols.append(symbol)
            for i in range(len(prev_symbols) - 1):
                prev_stats[(prev_symbols[i], prev_symbols[i + 1])] -= freq
            for i in range(len(symbols) - 1):
                prev_stats[(symbols[i], symbols[i + 1])] += freq
    return prev_stats


def merge_vocab(
    pair: tuple[bytes, bytes], vocab_in: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, ...], int]:
    """
    pair = ('low')
    vocab_in = {('lo', 'w', 'er'): 4}
    =>
    vocab_out = {('low', 'er'): 4}
    """
    vocab_out: dict[tuple[bytes, ...], int] = {}

    for word, freq in vocab_in.items():
        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if (
                i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]
            ):  # need update
                new_word.append(pair[0] + pair[1])
                i += 2  # skip by 2 so no double counting
            else:
                new_word.append(word[i])
                i += 1
        vocab_out[tuple(new_word)] = freq
    return vocab_out


def pretokenize_on_text(
    text: str, special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    "lower low"
    =>
    {(l o w e r): 1, (l o w): 1}
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_count: dict[tuple[bytes, ...], int] = {}

    pattern = (
        r"("
        + "|".join(re.escape(special_token) for special_token in special_tokens)
        + r")"
    )[1:-1]
    parts = re.split(pattern, text)
    for part in parts:
        re_iter = re.finditer(PAT, part)
        for s in re_iter:
            pretoken = s.group(0).encode("utf-8")  # str to bytes
            bytes_tuple: tuple[bytes, ...] = tuple(bytes([a]) for a in pretoken)  #
            pretoken_count[bytes_tuple] = pretoken_count.get(bytes_tuple, 0) + 1
    return pretoken_count


def pretokenize_on_text_add_to_multiprocess_queue(
    text: str, special_tokens: list[str], queue: Queue
) -> None:
    queue.put(pretokenize_on_text(text, special_tokens))


def pretokenize_on_text_add_to_multiprocess_list(
    chunk: str, special_tokens: list[str], results_list
):
    """Top-level function for multiprocessing."""
    results_list.append(pretokenize_on_text(chunk, special_tokens))


def do_pretokenization_single_process(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "r", encoding="utf-8") as f:
        text: str = f.read()
        print("tlu7... do_pretokenization_single_process for len(text) ", len(text))
        return pretokenize_on_text(text=text, special_tokens=special_tokens)


def do_pretokenization(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    do_multiprocess: bool = False,
) -> dict[tuple[bytes, ...], int]:
    SPECIAL_TOKEN = b"<|endoftext|>"
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # pretoken_count: dict[tuple[bytes, ...], int] = {}

    if not do_multiprocess:
        return do_pretokenization_single_process(
            input_path=input_path, special_tokens=special_tokens
        )
    else:
        manager = Manager()
        results_list = manager.list()  # shared list
        processes = []
        num_processes = 4
        chunks: list[str] = []

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, SPECIAL_TOKEN)

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk: str = f.read(end - start).decode(
                    "utf-8", errors="ignore"
                )  # back to string
                chunks.append(chunk)

        if len(chunks) < num_processes:  # fall back to single process
            return do_pretokenization_single_process(
                input_path=input_path, special_tokens=special_tokens
            )

        for i in range(num_processes):
            p = Process(
                target=pretokenize_on_text_add_to_multiprocess_list,
                args=(chunks[i], special_tokens, results_list),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # merge
        final = Counter()
        for pretoken_count_result in results_list:
            final.update(pretoken_count_result)
        print("tlu7...len(final)", len(final))
        return final


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    final_vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for special_token in special_tokens:  # e.g. <|endoftext}>
        final_vocab[len(final_vocab)] = special_token.encode("utf-8")
    # initialize 256 bytes:
    for i in range(256):
        final_vocab[len(final_vocab)] = bytes([i])

    pretoken_count: dict[tuple[bytes, ...], int] = do_pretokenization(
        input_path=input_path, special_tokens=special_tokens, do_multiprocess=True
    )
    print("tlu7... all pretoken_count finished")

    vocab = {k: v for k, v in pretoken_count.items()}

    remain_vocab_size = vocab_size - len(final_vocab)

    # in a single loop:
    # 1. get frequency of adjacent pairs of tokens
    # 2. take the lexically largest (b1, b2) as the new merge;  add it to the merge output
    # 3. for every pretoken, update its frequency:  ab1b2c -> ab1b2, b1b2c
    stats: dict[tuple[bytes, bytes], int] = {}
    best_pair: tuple[bytes, bytes] = None
    for i in range(remain_vocab_size):
        if len(stats) == 0:
            stats: dict[tuple[bytes, bytes], int] = get_stats(vocab)
        else:
            stats: dict[tuple[bytes, bytes], int] = get_stats_fast(
                vocab, stats, best_pair
            )
        # break tie by choosing lexically largest key
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        n = len(final_vocab)
        final_vocab[n] = best_pair[0] + best_pair[1]  # still a bytes
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)

    return (final_vocab, merges)
