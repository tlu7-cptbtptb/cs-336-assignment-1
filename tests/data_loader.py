
import numpy as np
import torch
from typing import Tuple
import random

def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dataset: 1D numpy array of integer token IDs in the dataset.

    e.g. batch size = 2, context length = 3
    return:
    ([[0, 1, 2], [2, 3, 4]], [[9, 10, 11], [10, 11, 12]])
    """

    n_seq = dataset.shape[0] - context_length

    # build sliding windows

    # take batch_size sequences for input and next-shifted for target


    start_idx = [random.randint(0, n_seq - 1) for _ in range(batch_size)]

    x = [dataset[idx:idx+context_length] for idx in start_idx]
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = x + 1

    return (x, y)
