"""
LoRA Linear Layer Adapter with Out-of-Core Base Weight Decoupling.
Provides:
- Parameter-efficient low-rank adaptation: y = x @ W_0.T + (alpha / r) * (x @ A.T) @ B.T
- Decoupled base weight streaming (base weight W_0 is loaded on demand and not retained)
- Direct analytical gradients for LoRA matrices A and B
"""

import math
from typing import Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = object


class LoRAParameter:
    """Wrapper for NumPy arrays to provide .data and .grad attributes."""
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, value):
        self.data[item] = value

    def __array__(self):
        return self.data


class LazyLoRALinear(nn.Module if HAS_TORCH else object):
    """
    LoRA Linear Layer designed for Lazy Out-of-Core MoE Training.
    Base weight W_0 is kept on disk / streaming buffer, only low-rank matrices A & B are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.dropout_rate = lora_dropout

        if HAS_TORCH:
            super().__init__()
            torch_dtype = torch.bfloat16 if dtype == "bfloat16" else (
                torch.float16 if dtype == "float16" else torch.float32
            )
            
            if r > 0:
                self.lora_A = nn.Parameter(torch.empty((r, in_features), dtype=torch_dtype, device=device))
                self.lora_B = nn.Parameter(torch.zeros((out_features, r), dtype=torch_dtype, device=device))
                
                # Initialize A with Kaiming uniform, B with zeros
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            else:
                self.lora_A = None
                self.lora_B = None

            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            if r > 0:
                # NumPy initialization
                bound = 1.0 / math.sqrt(in_features)
                self.lora_A = LoRAParameter(np.random.uniform(-bound, bound, (r, in_features)).astype(np.float32))
                self.lora_B = LoRAParameter(np.zeros((out_features, r), dtype=np.float32))
            else:
                self.lora_A = None
                self.lora_B = None
            self.dropout = None

    def forward_lora_only(self, x: Union["torch.Tensor", np.ndarray]) -> Union["torch.Tensor", np.ndarray]:
        """Compute only the low-rank delta: (alpha / r) * (dropout(x) @ A.T) @ B.T"""
        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            if HAS_TORCH and isinstance(x, torch.Tensor):
                return torch.zeros((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
            else:
                return np.zeros((*x.shape[:-1], self.out_features), dtype=x.dtype)

        if HAS_TORCH and isinstance(x, torch.Tensor):
            x_dropped = self.dropout(x)
            # x @ A.T -> [..., r]
            intermediate = F.linear(x_dropped, self.lora_A)
            # intermediate @ B.T -> [..., out_features]
            delta = F.linear(intermediate, self.lora_B) * self.scaling
            return delta
        else:
            A_data = self.lora_A.data if hasattr(self.lora_A, "data") else self.lora_A
            B_data = self.lora_B.data if hasattr(self.lora_B, "data") else self.lora_B
            intermediate = np.matmul(x, A_data.T)
            delta = np.matmul(intermediate, B_data.T) * self.scaling
            return delta

    def forward_with_base(
        self,
        x: Union["torch.Tensor", np.ndarray],
        base_weight: Union["torch.Tensor", np.ndarray],
        base_bias: Optional[Union["torch.Tensor", np.ndarray]] = None,
    ) -> Union["torch.Tensor", np.ndarray]:
        """
        Fused forward combining streaming base weight with low-rank adaptation:
        y = x @ base_weight.T + (base_bias) + forward_lora_only(x)
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            # Base linear projection
            y_base = F.linear(x, base_weight, base_bias)
            y_lora = self.forward_lora_only(x)
            return y_base + y_lora
        else:
            y_base = np.matmul(x, base_weight.T)
            if base_bias is not None:
                y_base = y_base + base_bias
            y_lora = self.forward_lora_only(x)
            return y_base + y_lora

    def compute_lora_gradients(
        self,
        grad_output: Union["torch.Tensor", np.ndarray],
        input_activation: Union["torch.Tensor", np.ndarray],
        base_weight: Optional[Union["torch.Tensor", np.ndarray]] = None,
    ) -> Tuple[
        Union["torch.Tensor", np.ndarray],
        Union["torch.Tensor", np.ndarray],
        Optional[Union["torch.Tensor", np.ndarray]],
    ]:
        """
        Compute explicit analytical gradients for LoRA A, LoRA B, and input activations.
        Returns:
            grad_A: Gradient w.r.t lora_A [r, in_features]
            grad_B: Gradient w.r.t lora_B [out_features, r]
            grad_input: Gradient w.r.t input_activation [..., in_features]
        """
        if HAS_TORCH and isinstance(grad_output, torch.Tensor):
            x_flat = input_activation.view(-1, self.in_features)  # [N, d_in]
            dy_flat = grad_output.view(-1, self.out_features)     # [N, d_out]

            # Intermediate h = x @ A.T  [N, r]
            h = F.linear(x_flat, self.lora_A)

            # dL / dB = scaling * dy.T @ h  [d_out, r]
            grad_B = self.scaling * torch.matmul(dy_flat.t(), h)

            # Upstream to intermediate: dh = scaling * dy @ B  [N, r]
            dh = self.scaling * F.linear(dy_flat, self.lora_B.t())

            # dL / dA = dh.T @ x  [r, d_in]
            grad_A = torch.matmul(dh.t(), x_flat)

            # Downstream grad to input: dx = dy @ W_0 + dh @ A
            grad_input = None
            if base_weight is not None:
                dx_base = F.linear(dy_flat, base_weight.t())
                dx_lora = F.linear(dh, self.lora_A.t())
                grad_input = (dx_base + dx_lora).view_as(input_activation)

            return grad_A, grad_B, grad_input
        else:
            x_flat = input_activation.reshape(-1, self.in_features)
            dy_flat = grad_output.reshape(-1, self.out_features)

            A_data = self.lora_A.data if hasattr(self.lora_A, "data") else self.lora_A
            B_data = self.lora_B.data if hasattr(self.lora_B, "data") else self.lora_B

            h = np.matmul(x_flat, A_data.T)
            grad_B = self.scaling * np.matmul(dy_flat.T, h)

            dh = self.scaling * np.matmul(dy_flat, B_data)
            grad_A = np.matmul(dh.T, x_flat)

            grad_input = None
            if base_weight is not None:
                dx_base = np.matmul(dy_flat, base_weight)
                dx_lora = np.matmul(dh, A_data)
                grad_input = (dx_base + dx_lora).reshape(input_activation.shape)

            return grad_A, grad_B, grad_input

    def accumulate_grad(
        self,
        grad_A: Union["torch.Tensor", np.ndarray],
        grad_B: Union["torch.Tensor", np.ndarray],
    ) -> None:
        """Accumulate analytical gradients into trainable LoRA parameters."""
        if self.lora_A is None or self.lora_B is None:
            return

        if HAS_TORCH and isinstance(self.lora_A, torch.Tensor):
            t_grad_A = grad_A if isinstance(grad_A, torch.Tensor) else torch.from_numpy(grad_A)
            t_grad_B = grad_B if isinstance(grad_B, torch.Tensor) else torch.from_numpy(grad_B)

            if self.lora_A.grad is None:
                self.lora_A.grad = t_grad_A.clone().to(self.lora_A.device, dtype=self.lora_A.dtype)
            else:
                self.lora_A.grad.add_(t_grad_A.to(self.lora_A.device, dtype=self.lora_A.dtype))

            if self.lora_B.grad is None:
                self.lora_B.grad = t_grad_B.clone().to(self.lora_B.device, dtype=self.lora_B.dtype)
            else:
                self.lora_B.grad.add_(t_grad_B.to(self.lora_B.device, dtype=self.lora_B.dtype))
        else:
            gA = np.asarray(grad_A, dtype=np.float32)
            gB = np.asarray(grad_B, dtype=np.float32)
            if self.lora_A.grad is None:
                self.lora_A.grad = gA.copy()
            else:
                self.lora_A.grad += gA

            if self.lora_B.grad is None:
                self.lora_B.grad = gB.copy()
            else:
                self.lora_B.grad += gB
