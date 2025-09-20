import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch

from regex import B


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-8,
        weight_decay=0.01,
    ):
        hyperparams = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, hyperparams)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # state["step"] = 0
                state["1st_momentum"] = torch.zeros_like(p.data)
                state["2nd_momentum"] = torch.zeros_like(p.data)
                state["t"] = 1

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p
                grad = p.grad.data
                state["1st_momentum"] = (
                    beta1 * state["1st_momentum"] + (1 - beta1) * grad
                )
                state["2nd_momentum"] = (
                    beta2 * state["2nd_momentum"] + (1 - beta2) * grad * grad
                )
                t = state.get(
                    "t", 1
                )  # Get iteration number from the state, or initial value.
                lr_adjusted = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= (
                    lr_adjusted
                    * state["1st_momentum"]
                    / (torch.sqrt(state["2nd_momentum"]) + eps)
                )
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1  # Increment iteration number.
        return loss
