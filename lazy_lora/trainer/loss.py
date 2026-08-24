"""
Loss Functions for LazyLoRA Language Model Training.
Provides CrossEntropyLoss with label smoothing and ignore_index support.
"""

from typing import Tuple, Union, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def compute_cross_entropy_loss(
    logits: Union["torch.Tensor", np.ndarray],
    targets: Union["torch.Tensor", np.ndarray],
    ignore_index: int = 163839,
    label_smoothing: float = 0.0,
) -> Tuple[Union["torch.Tensor", float], Optional[Union["torch.Tensor", np.ndarray]]]:
    """
    Computes Cross Entropy Loss and analytical gradient w.r.t logits.
    Args:
        logits: [batch_size, seq_len, vocab_size] or [N, vocab_size]
        targets: [batch_size, seq_len] or [N]
    Returns:
        loss: scalar loss value
        grad_logits: [N, vocab_size] gradient tensor (if requested or in autograd)
    """
    if HAS_TORCH and isinstance(logits, torch.Tensor):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        valid_mask = targets_flat != ignore_index
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        probs = F.softmax(logits_flat, dim=-1)
        grad_logits = probs.clone()
        if valid_mask.any():
            num_valid = valid_mask.sum().item()
            grad_logits[valid_mask, targets_flat[valid_mask]] -= 1.0
            grad_logits[~valid_mask] = 0.0
            grad_logits = grad_logits / max(1, num_valid)
        else:
            grad_logits.zero_()
        return loss, grad_logits.view_as(logits)
    else:
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)

        # Mask ignored indices
        valid_mask = targets_flat != ignore_index
        if not np.any(valid_mask):
            return 0.0, np.zeros_like(logits_flat)

        valid_logits = logits_flat[valid_mask]
        valid_targets = targets_flat[valid_mask]

        # Softmax with numerical stability
        shifted_logits = valid_logits - np.max(valid_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-20)

        # Cross entropy
        N = valid_logits.shape[0]
        target_probs = probs[np.arange(N), valid_targets]
        loss = -np.mean(np.log(target_probs + 1e-20))

        # Gradient w.r.t logits: (probs - one_hot) / N
        grad_logits = np.zeros_like(logits_flat)
        grad_valid = probs.copy()
        grad_valid[np.arange(N), valid_targets] -= 1.0
        grad_valid /= max(1, N)
        grad_logits[valid_mask] = grad_valid

        return float(loss), grad_logits
