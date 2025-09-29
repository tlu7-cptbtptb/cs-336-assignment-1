import csv

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
from .checkpoint_util import load_checkpoint, save_checkpoint
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


def reload_model_and_optimizer(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
    rope_theta: float,
    max_seq_len: int,
    src: str,
):
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
    transformer, optimizer, iteration = load_checkpoint(
        src=src, model=transformer, optimizer=optimizer, return_model_and_optimizer=True
    )
    return transformer, optimizer, iteration


def train_bpe_and_save(
    input_path: str, vocab_size: int, skip_train: bool = True
) -> tuple[str, str]:
    prefix = "/Users/tlu7/git_proj/stanford_336/cs-336-assignment-1/"
    vocab_path = f"{prefix}/vocab.pkl"
    merges_path = f"{prefix}/merges.pkl"
    if not skip_train:
        # tokenizer training
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )

        # save vocab
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        # save merges
        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)
    return vocab_path, merges_path


def load_tokenizer_from_saved_vocab_merges(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None = None,
):
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def prepare_train_or_valid_data(
    corpus_path: str,
    tokenizer: Any,
    train_or_valid: str = "train",
):
    with open(corpus_path, "r") as f:
        corpus = f.read()
    token_ids = tokenizer.encode(corpus)
    print("len token_ids, ", len(token_ids))
    if train_or_valid == "train":
        train_dataset = numpy.array(token_ids, dtype=numpy.int32)
        # Save to disk in .npy format (efficient for memmap)
        numpy.save("train_tokens.npy", train_dataset)
        train_dataset = numpy.load("train_tokens.npy", mmap_mode="r")  # read-only
        return train_dataset
    else:
        validation_dataset = numpy.array(token_ids, dtype=numpy.int32)
        # Save to disk in .npy format (efficient for memmap)
        numpy.save("validation_tokens.npy", validation_dataset)
        validation_dataset = numpy.load(
            "validation_tokens.npy", mmap_mode="r"
        )  # read-only
        return validation_dataset


def prepare_valid_data_for_loss(
    validation_dataset: numpy.ndarray,
    context_length: int,
    batches: int = 100,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    result = []
    for b in range(batches):
        batch = data_loading(
            dataset=validation_dataset,
            batch_size=1,
            context_length=context_length,
            device="cpu",
        )
        result.append(batch)
    return result


def calculate_validation_loss(
    model: Transformer, validation_data: list[tuple[torch.Tensor, torch.Tensor]]
):
    model.eval()
    loss_per_batch = []
    with torch.no_grad():
        for batch in validation_data:
            input = batch[0]
            target = batch[1]
            predicted_logits = model(input)
            loss = cross_entropy_loss(predicted_logits, target)
            loss_per_batch.append(loss.item())
    return torch.mean(torch.tensor(loss_per_batch))


def test_main(
    vocab_size: int = 10000,
    context_length: int = 64,
    d_model: int = 512,
    d_ff: int = 1344,
    rope_theta: float = 10000.0,
    num_layers: int = 4,
    num_heads: int = 16,
    data_already_tokenized: bool = True,
):
    prefix = "/Users/tlu7/git_proj/stanford_336/cs-336-assignment-1/data"
    train_path = f"{prefix}/TinyStoriesV2-GPT4-train_10000.txt"
    validation_path = f"{prefix}/TinyStoriesV2-GPT4-valid.txt"
    # tokenizer = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    vocab_path, merges_path = train_bpe_and_save(
        input_path=train_path, vocab_size=vocab_size, skip_train=True
    )
    tokenizer = load_tokenizer_from_saved_vocab_merges(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=["<|endoftext|>"],
    )
    vocab = tokenizer.vocab

    end_of_text_token_id = None
    for i in range(len(vocab)):
        if vocab[i] == "<|endoftext|>".encode("utf-8"):
            end_of_text_token_id = i
            break
    if data_already_tokenized:
        train_dataset = numpy.load("train_tokens.npy", mmap_mode="r")  # read-only
        validation_dataset = numpy.load(
            "validation_tokens.npy", mmap_mode="r"
        )  # read-only
    else:
        train_dataset = prepare_train_or_valid_data(
            corpus_path=train_path, tokenizer=tokenizer
        )
        validation_dataset = prepare_train_or_valid_data(
            corpus_path=validation_path, tokenizer=tokenizer, train_or_valid="valid"
        )

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
    compiled_model = torch.compile(transformer)
    # prepare validation data (sample a large number of batches from the full validation dataset)
    valid_data = prepare_valid_data_for_loss(
        validation_dataset=validation_dataset,
        context_length=context_length,
        batches=100,
    )

    # training loop
    valid_loss_per_step = []

    for step in range(2000):
        batch = data_loading(
            dataset=train_dataset,
            batch_size=4,
            context_length=context_length,
            device="cpu",
        )
        input = batch[0]
        target = batch[1]
        loss = train_step(
            model=compiled_model, optimizer=optimizer, input=input, target=target
        )
        if step % 5 == 0:
            print("step, ", step, "loss, ", loss)

        if step % 20 == 0:
            validation_loss = calculate_validation_loss(
                model=transformer, validation_data=valid_data
            )
            print("-------------------")
            print("step, ", step, "validation_loss, ", validation_loss)
            valid_loss_per_step.append((step, validation_loss.item()))

        if step % 50 == 0:
            generate_text_for_test_input(
                test_input=None,
                transformer=transformer,
                tokenizer=tokenizer,
                end_of_text_token_id=end_of_text_token_id,
                max_gen_len=64,
            )

    with open("valid_loss_lr=1e-3_post_norm.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])  # header
        writer.writerows(valid_loss_per_step)

    step = 50
    save_path = (
        f"""/Users/tlu7/git_proj/stanford_336/cs-336-assignment-1/ckpt/{step}.ckpt"""
    )
    save_checkpoint(
        model=transformer, optimizer=optimizer, iteration=step, out=save_path
    )
    print("---------saving model DONE ----------")
