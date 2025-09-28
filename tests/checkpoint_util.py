import os
import re
import typing
from typing import Any, Dict, List, Optional, Tuple

import torch


"""
In addition to loading data, we will also need to save models as we train. When running jobs, we often
want to be able to resume a training run that for some reason stopped midway (e.g., due to your job timing
out, machine failure, etc). Even when all goes well, we might also want to later have access to intermediate
models (e.g., to study training dynamics post-hoc, take samples from models at different stages of training,
etc).

A checkpoint should have all the states that we need to resume training. We of course want to be able
to restore model weights at a minimum. If using a stateful optimizer (such as AdamW), we will also need
to save the optimizer’s state (e.g., in the case of AdamW, the moment estimates). Finally, to resume the
learning rate schedule, we will need to know the iteration number we stopped at. PyTorch makes it easy to
save all of these: every nn.Module has a state_dict() method that returns a dictionary with all learnable
weights; we can restore these weights later with the sister method load_state_dict(). The same goes
for any nn.optim.Optimizer. Finally, torch.save(obj, dest) can dump an object (e.g., a dictionary
containing tensors in some values, but also regular Python objects like integers) to a file (path) or file-like
object, which can then be loaded back into memory with torch.load(src)
"""


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    dict_to_save = {
        "model": model_state,
        "optimizer": optimizer_state,
        "iteration": iteration,
    }
    torch.save(dict_to_save, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    return_model_and_optimizer: bool = False,
) -> int | tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    if not return_model_and_optimizer:
        return iteration
    else:
        return (model, optimizer, iteration)
