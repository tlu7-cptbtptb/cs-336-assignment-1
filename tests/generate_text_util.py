import math

import torch

from .transformer import Transformer
from .util_layers import softmax


def generate_text(
    model: Transformer,
    prompt: torch.Tensor,  # the prompt after tokenization; can assume 1D
    end_of_text_token: int,
    max_gen_len: int | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Generate text from a prompt using a Transformer model.
    """
    output_seq = []

    max_gen_len = max_gen_len or 1000

    has_completed = False

    while not has_completed:
        model_output = model(prompt)
        next_token_logits = model_output[0, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token_probs = softmax(next_token_logits, dim=-1)
        sampled_next_token = torch.multinomial(next_token_probs, num_samples=1)[
            0
        ]  # tensor of shape (1, )

        output_seq.append(sampled_next_token)
        prompt = torch.cat(
            [
                prompt,
                sampled_next_token.view(
                    1,
                ),
            ]
        )
        if len(output_seq) >= max_gen_len or sampled_next_token == end_of_text_token:
            has_completed = True

    return torch.Tensor(output_seq)
