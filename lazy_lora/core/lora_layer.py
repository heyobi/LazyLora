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
        dtype: str = "float32",
    ):
        # The adapters are float32 (report finding K2): a bf16 parameter carries ~3
        # significant digits, so an Adam step of 2e-4 on a 0.05 weight sits at its
        # rounding threshold and most later updates round to zero. The forward casts the
        # low-rank delta back to the activation dtype, so the base path stays bf16.
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
                return np.zeros((*x.shape[:-1], self.out_features), dtype=getattr(x, "dtype", np.float32))

        if HAS_TORCH and isinstance(x, torch.Tensor):
            x_dropped = self.dropout(x)
            orig_dtype = x.dtype
            if self.lora_A is not None and x_dropped.dtype != self.lora_A.dtype:
                x_dropped = x_dropped.to(self.lora_A.dtype)
            # x @ A.T -> [..., r]
            intermediate = F.linear(x_dropped, self.lora_A)
            # intermediate @ B.T -> [..., out_features]
            delta = F.linear(intermediate, self.lora_B) * self.scaling
            if delta.dtype != orig_dtype:
                delta = delta.to(orig_dtype)
            return delta
        else:
            def _to_np(p):
                if p is None:
                    return None
                if HAS_TORCH and isinstance(p, torch.Tensor):
                    return p.detach().to(torch.float32).cpu().numpy()
                if hasattr(p, "data"):
                    return _to_np(p.data)
                return np.asarray(p, dtype=np.float32)

            A_data = _to_np(self.lora_A)
            B_data = _to_np(self.lora_B)
            x_np = np.asarray(x, dtype=np.float32)
            intermediate = np.matmul(x_np, A_data.T)
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
            if base_weight is not None and base_weight.dtype != x.dtype:
                base_weight = base_weight.to(x.dtype)
            if base_bias is not None and base_bias.dtype != x.dtype:
                base_bias = base_bias.to(x.dtype)
            y_base = F.linear(x, base_weight, base_bias)
            y_lora = self.forward_lora_only(x)
            return y_base + y_lora
        else:
            def _to_np(p):
                if p is None:
                    return None
                if HAS_TORCH and isinstance(p, torch.Tensor):
                    return p.detach().to(torch.float32).cpu().numpy()
                if hasattr(p, "data"):
                    return _to_np(p.data)
                return np.asarray(p, dtype=np.float32)

            x_np = np.asarray(x, dtype=np.float32)
            bw_np = _to_np(base_weight)
            bb_np = _to_np(base_bias)
            y_base = np.matmul(x_np, bw_np.T)
            if bb_np is not None:
                y_base = y_base + bb_np
            y_lora = self.forward_lora_only(x_np)
            return y_base + y_lora

    def compute_lora_gradients(
        self,
        grad_output: Union["torch.Tensor", np.ndarray],
        input_activation: Union["torch.Tensor", np.ndarray],
        base_weight: Optional[Union["torch.Tensor", np.ndarray]] = None,
        dx_base: Optional["torch.Tensor"] = None,
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
            target_dtype = self.lora_A.dtype if self.lora_A is not None else grad_output.dtype
            x_flat = input_activation.view(-1, self.in_features).to(target_dtype)
            dy_flat = grad_output.view(-1, self.out_features).to(target_dtype)

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
            if dx_base is None and base_weight is not None:
                # dy @ W_0 in the base weight's own dtype: widening a 3072x3584 expert
                # matrix to fp32 for every expert would cost more than the matmul itself.
                dx_base = F.linear(grad_output.view(-1, self.out_features).to(base_weight.dtype), base_weight.t())
            if dx_base is not None:
                # dx_base = dy @ W_0 computed by the caller (e.g. the native MXFP4 kernel)
                dx_lora = F.linear(dh, self.lora_A.t())
                grad_input = (dx_base.view(-1, self.in_features).to(target_dtype) + dx_lora).view_as(input_activation)
                if grad_input.dtype != input_activation.dtype:
                    grad_input = grad_input.to(input_activation.dtype)

            return grad_A, grad_B, grad_input
        else:
            def _to_np(p):
                if p is None:
                    return None
                if HAS_TORCH and isinstance(p, torch.Tensor):
                    return p.detach().to(torch.float32).cpu().numpy()
                if hasattr(p, "data"):
                    return _to_np(p.data)
                return np.asarray(p, dtype=np.float32)

            x_flat = np.asarray(input_activation, dtype=np.float32).reshape(-1, self.in_features)
            dy_flat = np.asarray(grad_output, dtype=np.float32).reshape(-1, self.out_features)

            A_data = _to_np(self.lora_A)
            B_data = _to_np(self.lora_B)

            h = np.matmul(x_flat, A_data.T)
            grad_B = self.scaling * np.matmul(dy_flat.T, h)

            dh = self.scaling * np.matmul(dy_flat, B_data)
            grad_A = np.matmul(dh.T, x_flat)

            grad_input = None
            if base_weight is not None:
                bw_np = _to_np(base_weight)
                dx_base = np.matmul(dy_flat, bw_np)
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
            if isinstance(grad_A, torch.Tensor):
                t_grad_A = grad_A.to(self.lora_A.device, dtype=self.lora_A.dtype)
            else:
                t_grad_A = torch.from_numpy(np.asarray(grad_A, dtype=np.float32)).to(self.lora_A.device, dtype=self.lora_A.dtype)

            if isinstance(grad_B, torch.Tensor):
                t_grad_B = grad_B.to(self.lora_B.device, dtype=self.lora_B.dtype)
            else:
                t_grad_B = torch.from_numpy(np.asarray(grad_B, dtype=np.float32)).to(self.lora_B.device, dtype=self.lora_B.dtype)

            if self.lora_A.grad is None:
                self.lora_A.grad = t_grad_A.clone()
            else:
                self.lora_A.grad.add_(t_grad_A)

            if self.lora_B.grad is None:
                self.lora_B.grad = t_grad_B.clone()
            else:
                self.lora_B.grad.add_(t_grad_B)
        else:
            def _to_np(p):
                if p is None:
                    return None
                if HAS_TORCH and isinstance(p, torch.Tensor):
                    return p.detach().to(torch.float32).cpu().numpy()
                if hasattr(p, "data"):
                    return _to_np(p.data)
                return np.asarray(p, dtype=np.float32)

            gA = _to_np(grad_A)
            gB = _to_np(grad_B)

            if self.lora_A.grad is None:
                self.lora_A.grad = gA.copy()
            else:
                self.lora_A.grad += gA

            if self.lora_B.grad is None:
                self.lora_B.grad = gB.copy()
            else:
                self.lora_B.grad += gB
