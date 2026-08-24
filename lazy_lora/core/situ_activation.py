"""
SiTU (Sinusoidal Tanh Unit) and SiTU-GLU Activation Functions for Kimi K3.
Provides both PyTorch (GPU/CPU autograd) and NumPy/C analytical reference implementations.
"""

import math
from typing import Union, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = object

import numpy as np


class SiTUFunction(torch.autograd.Function if HAS_TORCH else object):
    """
    Custom Autograd Function for SiTU activation with fused forward and backward.
    situ(x, beta) = x * tanh(beta * x)
    d/dx = tanh(beta * x) + beta * x * (1 - tanh^2(beta * x))
    """

    @staticmethod
    def forward(ctx, x: "torch.Tensor", beta: float = 4.0) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for SiTU autograd function")
        tanh_val = torch.tanh(beta * x)
        ctx.save_for_backward(x, tanh_val)
        ctx.beta = beta
        return x * tanh_val

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor") -> Tuple["torch.Tensor", None]:
        x, tanh_val = ctx.saved_tensors
        beta = ctx.beta
        # sech^2(u) = 1 - tanh^2(u)
        grad_x = (tanh_val + beta * x * (1.0 - tanh_val * tanh_val)) * grad_output
        return grad_x, None


def situ_forward(x: Union["torch.Tensor", np.ndarray], beta: float = 4.0) -> Union["torch.Tensor", np.ndarray]:
    """Forward pass for SiTU activation."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return SiTUFunction.apply(x, beta)
    else:
        tanh_val = np.tanh(beta * x)
        return x * tanh_val


def situ_glu_forward(
    gate: Union["torch.Tensor", np.ndarray],
    up: Union["torch.Tensor", np.ndarray],
    beta: float = 4.0
) -> Union["torch.Tensor", np.ndarray]:
    """
    SiTU-Gated Linear Unit (SiTU-GLU) used in Kimi K3 MoE expert FFNs.
    Output = situ(gate, beta) * up
    """
    return situ_forward(gate, beta) * up


def situ_glu_backward(
    grad_output: Union["torch.Tensor", np.ndarray],
    gate: Union["torch.Tensor", np.ndarray],
    up: Union["torch.Tensor", np.ndarray],
    beta: float = 4.0
) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
    """
    Analytical backward pass for SiTU-GLU.
    Returns (grad_gate, grad_up).
    """
    if HAS_TORCH and isinstance(grad_output, torch.Tensor):
        tanh_gate = torch.tanh(beta * gate)
        situ_gate = gate * tanh_gate
        d_situ = tanh_gate + beta * gate * (1.0 - tanh_gate * tanh_gate)

        grad_gate = grad_output * up * d_situ
        grad_up = grad_output * situ_gate
        return grad_gate, grad_up
    else:
        tanh_gate = np.tanh(beta * gate)
        situ_gate = gate * tanh_gate
        d_situ = tanh_gate + beta * gate * (1.0 - tanh_gate * tanh_gate)

        grad_gate = grad_output * up * d_situ
        grad_up = grad_output * situ_gate
        return grad_gate, grad_up


if HAS_TORCH:
    class SiTUGLU(nn.Module):
        """PyTorch Module for SiTU-GLU."""
        def __init__(self, beta: float = 4.0):
            super().__init__()
            self.beta = beta

        def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return situ_glu_forward(gate, up, self.beta)
else:
    class SiTUGLU:
        def __init__(self, beta: float = 4.0):
            self.beta = beta
