import numpy
import torch
import torch.nn.functional as F
from einops import rearrange

from .adapters import (
    get_tokenizer,
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_silu,
    run_swiglu,
    run_train_bpe,
    run_transformer_block,
    run_transformer_lm,
)
from .tokenizer import *
from .data_loader import data_loading
from .generate_text_util import generate_text
from .optimizer import AdamW, gradient_clipping, learning_rate_schedule, SGD
from .transformer import Transformer, TransformerBlock
from .util_layers import cross_entropy_loss, softmax


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [
        ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]
    ]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (
        rearrange(x, "(batch head) seq d -> batch head seq d", head=2)
        for x in (q, k, v)
    )
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    ts_state_dict,
    n_keys,
    theta,
    pos_ids,
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(
    numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta
):
    block_weights = {
        k.replace("layers.0.", ""): v
        for k, v in ts_state_dict[0].items()
        if "layers.0." in k
    }

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings
    )

    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    output = run_rope(
        d_model,
        theta=theta,
        max_seq_len=n_queries,
        in_query_or_key=in_embeddings,
        token_positions=pos_ids,
    )
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_load_one_batch(
    dataset: numpy.ndarray, tokenizer: Any, context_length: int = 16
) -> None:
    batch = data_loading(
        dataset=dataset, batch_size=4, context_length=context_length, device="cpu"
    )
    print("batch, ", batch[0])
    for i in range(4):
        print(i, tokenizer.decode(batch[0][i].tolist()))
        print("---")
        print(i, tokenizer.decode(batch[1][i].tolist()))
        print("---")


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    input: torch.Tensor,
    target: torch.Tensor,
):
    model.train()
    optimizer.zero_grad()
    predicted_logits = model(input)
    loss = cross_entropy_loss(predicted_logits, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def get_optimizer(
    model: Transformer,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    return AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=eps,
    )


def generate_text_for_test_input(
    test_input: str | None,
    transformer: Transformer,
    tokenizer: Any,
    end_of_text_token_id: int,
    context_length: int = 64,
    max_gen_len: int = 64,
    temperature: float = 1.0,
) -> None:
    """
    Sanity check the model by generating text from a given input sentence.
    """
    default_test_input = "Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.\
                          Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but"
    test_input = test_input or default_test_input
    test_input = tokenizer.encode(test_input)
    test_input = test_input[:context_length]
    test_input = torch.tensor(test_input, dtype=torch.long, device="cpu")
    test_input = test_input.view(1, -1)
    test_output = generate_text(
        prompt=test_input,
        model=transformer,
        end_of_text_token=end_of_text_token_id,
        max_gen_len=max_gen_len,
        temperature=temperature,
    )
    test_output = tokenizer.decode(test_output.tolist())
    print("test_output, ", test_output)


def test_main(
    vocab_size: int = 10000,
    context_length: int = 64,
    d_model: int = 512,
    d_ff: int = 1344,
    rope_theta: float = 10000.0,
    num_layers: int = 4,
    num_heads: int = 16,
):
    # tokenizer training
    input_path = "/Users/tlu7/git_proj/stanford_336/cs-336-assignment-1/data/TinyStoriesV2-GPT4-train_100.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path, vocab_size=vocab_size, special_tokens=["<|endoftext|>"]
    )
    tokenizer = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    end_of_text_token_id = None
    for i in range(len(vocab)):
        if vocab[i] == "<|endoftext|>".encode("utf-8"):
            end_of_text_token_id = i
            break

    with open(input_path, "r") as f:
        corpus = f.read()
    token_ids = tokenizer.encode(corpus)
    print("len token_ids, ", len(token_ids))

    dataset = numpy.array(token_ids, dtype=numpy.int32)
    # Save to disk in .npy format (efficient for memmap)
    numpy.save("tokens.npy", dataset)

    dataset = numpy.load("tokens.npy", mmap_mode="r")  # read-only

    # sanity check
    # test_load_one_batch(dataset, tokenizer, context_length=context_length)

    # model and optimizer initialization
    transformer = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        theta=rope_theta,
        max_seq_len=context_length,
    )
    optimizer = get_optimizer(model=transformer)

    # sanity check
    test_input = "Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.\
                            Tom asked his friend, Sam, to help him search for the ball. "
    test_input = tokenizer.encode(test_input)

    # training loop
    for step in range(1000):
        batch = data_loading(
            dataset=dataset, batch_size=4, context_length=context_length, device="cpu"
        )
        input = batch[0]
        target = batch[1]
        loss = train_step(
            model=transformer, optimizer=optimizer, input=input, target=target
        )
        print("loss, ", loss)

        if step % 50 == 0:
            generate_text_for_test_input(
                test_input=None,
                transformer=transformer,
                tokenizer=tokenizer,
                end_of_text_token_id=end_of_text_token_id,
                max_gen_len=64,
            )
