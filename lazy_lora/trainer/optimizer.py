"""
Low-Overhead AdamW Optimizer and Learning Rate Scheduler for LazyLoRA.
Optimizes only the trainable low-rank adaptation matrices (A & B) with gradient clipping.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import torch
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class CosineWarmupLRScheduler:
    """Cosine Annealing with Linear Warmup."""
    def __init__(
        self,
        base_lr: float = 2e-4,
        min_lr: float = 1e-5,
        warmup_steps: int = 50,
        max_steps: int = 1000,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(warmup_steps + 1, max_steps)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step / float(self.warmup_steps))
        progress = (step - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


class LazyLoRAOptimizer:
    """
    AdamW Optimizer customized for LazyLoRA parameters.
    Maintains first and second moment buffers (m_t, v_t) with weight decay and gradient clipping.
    """

    def __init__(
        self,
        parameters: List[Any],
        lr: float = 2e-4,
        min_lr: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 50,
        max_steps: int = 1000,
        grad_clip_norm: float = 1.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.step_count = 0
        self.scheduler = CosineWarmupLRScheduler(lr, min_lr, warmup_steps, max_steps)

        # Optimizer states: m (first moment), v (second moment)
        self.m_states: Dict[int, Any] = {}
        self.v_states: Dict[int, Any] = {}

    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to resume: moments, step counter, schedule and hyperparameters."""
        return {
            "step_count": self.step_count,
            "m_states": {int(k): (v.detach().cpu().clone() if HAS_TORCH and isinstance(v, torch.Tensor) else np.array(v))
                         for k, v in self.m_states.items()},
            "v_states": {int(k): (v.detach().cpu().clone() if HAS_TORCH and isinstance(v, torch.Tensor) else np.array(v))
                         for k, v in self.v_states.items()},
            "lr": self.lr, "betas": (self.beta1, self.beta2), "eps": self.eps,
            "weight_decay": self.weight_decay, "grad_clip_norm": self.grad_clip_norm,
            "scheduler": {"base_lr": self.scheduler.base_lr, "min_lr": self.scheduler.min_lr,
                          "warmup_steps": self.scheduler.warmup_steps, "max_steps": self.scheduler.max_steps},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore moments and the step counter. Without the moments a resumed Adam restarts
        its bias correction at zero and the first steps take an effective lr far above the
        schedule."""
        self.step_count = int(state["step_count"])
        n = len(self.parameters)
        self.m_states = {}
        self.v_states = {}
        for k, v in state["m_states"].items():
            if int(k) < n:
                self.m_states[int(k)] = v.clone() if HAS_TORCH and isinstance(v, torch.Tensor) else np.array(v)
        for k, v in state["v_states"].items():
            if int(k) < n:
                self.v_states[int(k)] = v.clone() if HAS_TORCH and isinstance(v, torch.Tensor) else np.array(v)
        sch = state.get("scheduler")
        if sch:
            self.scheduler = CosineWarmupLRScheduler(sch["base_lr"], sch["min_lr"], sch["warmup_steps"], sch["max_steps"])

    def zero_grad(self) -> None:
        """Clear all gradients."""
        for p in self.parameters:
            if HAS_TORCH and isinstance(p, torch.Tensor):
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            elif hasattr(p, "grad"):
                p.grad = None

    def clip_grad_norm(self) -> float:
        """Clip gradients by global L2 norm."""
        total_norm_sq = 0.0
        for p in self.parameters:
            grad = p.grad if hasattr(p, "grad") else None
            if grad is not None:
                if HAS_TORCH and isinstance(grad, torch.Tensor):
                    total_norm_sq += grad.data.norm(2).item() ** 2
                else:
                    g_data = np.asarray(grad.data if hasattr(grad, "data") else grad, dtype=np.float32)
                    total_norm_sq += float(np.sum(g_data ** 2))

        total_norm = math.sqrt(total_norm_sq)
        if total_norm > self.grad_clip_norm and total_norm > 0:
            scale = self.grad_clip_norm / (total_norm + 1e-6)
            for p in self.parameters:
                grad = p.grad if hasattr(p, "grad") else None
                if grad is not None:
                    if HAS_TORCH and isinstance(grad, torch.Tensor):
                        grad.data.mul_(scale)
                    elif hasattr(grad, "data"):
                        grad.data = np.asarray(grad.data) * scale
                    else:
                        p.grad = np.asarray(p.grad) * scale
        return total_norm

    def step(self) -> float:
        """Perform a single AdamW optimization step across all parameters."""
        self.step_count += 1
        current_lr = self.scheduler.get_lr(self.step_count)
        self.clip_grad_norm()

        for idx, p in enumerate(self.parameters):
            grad = p.grad if hasattr(p, "grad") else None
            if grad is None:
                continue

            if HAS_TORCH and isinstance(p, torch.Tensor):
                data = p.data
                # The moments are kept in float32 even when the parameters are bfloat16:
                # bfloat16 carries ~3 significant digits, which is not enough to
                # accumulate a second moment or to make `eps` mean anything.
                g = grad.data.to(torch.float32)

                # Weight decay
                if self.weight_decay != 0:
                    data.mul_(1.0 - current_lr * self.weight_decay)

                # Initialize states
                if idx not in self.m_states:
                    self.m_states[idx] = torch.zeros(data.shape, dtype=torch.float32, device=data.device)
                    self.v_states[idx] = torch.zeros(data.shape, dtype=torch.float32, device=data.device)

                m = self.m_states[idx]
                v = self.v_states[idx]

                # Update moments
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                # Bias correction
                bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
                bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
                step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameter: data -= step_size * m / (sqrt(v) + eps)
                update = (m / (v.sqrt() + self.eps)).to(data.dtype)
                data.add_(update, alpha=-step_size)
            else:
                # NumPy / LoRAParameter implementation
                data = np.asarray(p.data if hasattr(p, "data") else p, dtype=np.float32)
                g = np.asarray(grad.data if hasattr(grad, "data") else grad, dtype=np.float32)

                if self.weight_decay != 0:
                    data = data * (1.0 - current_lr * self.weight_decay)

                if idx not in self.m_states:
                    self.m_states[idx] = np.zeros_like(data)
                    self.v_states[idx] = np.zeros_like(data)

                m = self.m_states[idx]
                v = self.v_states[idx]

                m = self.beta1 * m + (1.0 - self.beta1) * g
                v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)
                self.m_states[idx] = m
                self.v_states[idx] = v

                bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
                bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
                step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1

                data = data - step_size * m / (np.sqrt(v) + self.eps)
                if hasattr(p, "data"):
                    p.data = data
                else:
                    p = data

        return current_lr
