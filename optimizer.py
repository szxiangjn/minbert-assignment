from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                state = self.state[p]
                m = state["m"] if "m" in state else 0
                v = state["v"] if "v" in state else 0
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                beta1_t = (state["beta1_t"] if "beta1_t" in state else 1) * beta1
                state["beta1_t"] = beta1_t
                beta2_t = (state["beta2_t"] if "beta2_t" in state else 1) * beta2
                state["beta2_t"] = beta2_t
                # Update first and second moments of the gradients
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m
                v = beta2 * v + (1 - beta2) * grad ** 2
                state["v"] = v
                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                alpha_t = alpha * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                # Add weight decay
                p.mul_(1 - alpha_t * weight_decay)
                # Update parameters
                p.add_(m / torch.sqrt(v) + eps, alpha=-alpha_t)
                # Please note that the learning rate should be incorporated into this update.
        return loss
