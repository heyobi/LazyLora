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


SITU_BETA = 4.0            # activation_situ_beta
SITU_LINEAR_BETA = 25.0    # activation_situ_linear_beta


def situ_forward(
    x: Union["torch.Tensor", np.ndarray],
    beta: float = SITU_BETA,
) -> Union["torch.Tensor", np.ndarray]:
    """
    The gate branch of Kimi K3's SiTU activation:

        situ(g) = beta * tanh(g / beta) * sigmoid(g)

    Both factors are bounded, so the branch saturates at +-beta instead of growing with
    the input, and negative gates are suppressed by the sigmoid rather than being
    rectified into positive values.
    """
    if HAS_TORCH and isinstance(x, torch.Tensor):
        xf = x.to(torch.float32)
        out = beta * torch.tanh(xf / beta) * torch.sigmoid(xf)
        return out.to(x.dtype)
    xf = np.asarray(x, dtype=np.float32)
    return beta * np.tanh(xf / beta) / (1.0 + np.exp(-xf))


def situ_glu_forward(
    gate: Union["torch.Tensor", np.ndarray],
    up: Union["torch.Tensor", np.ndarray],
    beta: float = SITU_BETA,
    linear_beta: Optional[float] = SITU_LINEAR_BETA,
) -> Union["torch.Tensor", np.ndarray]:
    """
    SiTU-GLU as used in every Kimi K3 FFN (dense MLP, shared expert, routed experts):

        out = [beta * tanh(gate / beta) * sigmoid(gate)] * [linear_beta * tanh(up / linear_beta)]

    The up branch is soft-clipped at +-linear_beta. Computed in float32 like the
    reference implementation, then cast back.
    """
    if HAS_TORCH and isinstance(gate, torch.Tensor):
        dtype = gate.dtype
        g = gate.to(torch.float32)
        u = up.to(torch.float32)
        situ_a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        return (situ_a * u).to(dtype)

    g = np.asarray(gate, dtype=np.float32)
    u = np.asarray(up, dtype=np.float32)
    situ_a = beta * np.tanh(g / beta) / (1.0 + np.exp(-g))
    if linear_beta is not None:
        u = linear_beta * np.tanh(u / linear_beta)
    return situ_a * u


def situ_glu_backward(
    grad_output: Union["torch.Tensor", np.ndarray],
    gate: Union["torch.Tensor", np.ndarray],
    up: Union["torch.Tensor", np.ndarray],
    beta: float = SITU_BETA,
    linear_beta: Optional[float] = SITU_LINEAR_BETA,
) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
    """
    Analytical backward for SiTU-GLU.

    With t = tanh(g / beta) and s = sigmoid(g):
        situ(g)      = beta * t * s
        d situ / dg  = (1 - t^2) * s + beta * t * s * (1 - s)
    and for the soft-clipped up branch u' = linear_beta * tanh(u / linear_beta):
        d u' / du    = 1 - tanh^2(u / linear_beta)

    Returns (grad_gate, grad_up).
    """
    if HAS_TORCH and isinstance(grad_output, torch.Tensor):
        dtype = grad_output.dtype
        go = grad_output.to(torch.float32)
        g = gate.to(torch.float32)
        u = up.to(torch.float32)

        t = torch.tanh(g / beta)
        s = torch.sigmoid(g)
        situ_a = beta * t * s
        d_situ = (1.0 - t * t) * s + beta * t * s * (1.0 - s)

        if linear_beta is not None:
            tu = torch.tanh(u / linear_beta)
            u_out = linear_beta * tu
            d_up = 1.0 - tu * tu
        else:
            u_out = u
            d_up = torch.ones_like(u)

        return (go * u_out * d_situ).to(dtype), (go * situ_a * d_up).to(dtype)

    go = np.asarray(grad_output, dtype=np.float32)
    g = np.asarray(gate, dtype=np.float32)
    u = np.asarray(up, dtype=np.float32)

    t = np.tanh(g / beta)
    s = 1.0 / (1.0 + np.exp(-g))
    situ_a = beta * t * s
    d_situ = (1.0 - t * t) * s + beta * t * s * (1.0 - s)

    if linear_beta is not None:
        tu = np.tanh(u / linear_beta)
        u_out = linear_beta * tu
        d_up = 1.0 - tu * tu
    else:
        u_out = u
        d_up = np.ones_like(u)

    return go * u_out * d_situ, go * situ_a * d_up


if HAS_TORCH:
    class SiTUGLU(nn.Module):
        """PyTorch Module for SiTU-GLU."""
        def __init__(self, beta: float = SITU_BETA, linear_beta: Optional[float] = SITU_LINEAR_BETA):
            super().__init__()
            self.beta = beta
            self.linear_beta = linear_beta

        def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return situ_glu_forward(gate, up, self.beta, self.linear_beta)
else:
    class SiTUGLU:
        def __init__(self, beta: float = SITU_BETA, linear_beta: Optional[float] = SITU_LINEAR_BETA):
            self.beta = beta
            self.linear_beta = linear_beta
