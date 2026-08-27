"""
Kimi K3 Mixture-of-Experts (MoE) Router and Gating Mechanism.
Implements:
- Top-16 expert selection out of 896 fine-grained experts + 2 shared experts
- Unbiased Sigmoid combining weights with routing bias selection steering
- Dynamic token-to-expert dispatch index calculation
"""

from typing import Tuple, Optional, Dict, List, Union
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


class KimiK3MoERouter(nn.Module if HAS_TORCH else object):
    """
    Router for Kimi K3 MoE layers.
    896 routed experts, top-16 routing with unbiased sigmoid combining weights.
    """

    def __init__(
        self,
        hidden_size: int = 7168,
        num_experts: int = 896,
        top_k: int = 16,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
        use_bias: bool = True,
    ):
        if HAS_TORCH:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
            self.bias = nn.Parameter(torch.zeros(num_experts)) if use_bias else None
            # Initialize with small variance
            nn.init.normal_(self.weight, std=0.02)
        else:
            self.weight = np.random.normal(0, 0.02, (num_experts, hidden_size)).astype(np.float32)
            self.bias = np.zeros(num_experts, dtype=np.float32) if use_bias else None

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.renormalize = renormalize
        self.use_bias = use_bias

    def forward(
        self,
        x: Union["torch.Tensor", np.ndarray],
        weight: Optional[Union["torch.Tensor", np.ndarray]] = None,
        bias: Optional[Union["torch.Tensor", np.ndarray]] = None,
    ) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
        """
        Forward pass for MoE routing.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] or [N, hidden_size]
            weight: this layer's gate matrix [num_experts, hidden_size]. Each layer has its
                own gate on disk, so it is streamed in and passed here; the module's own
                randomly initialised weight is only a fallback for the mock configurations.
            bias: this layer's e_score_correction_bias [num_experts], added to the sigmoid
                scores for *selection* only, exactly as the reference gate does.
        Returns:
            topk_indices: [N, top_k] int64 indices of selected experts
            topk_weights: [N, top_k] float32 normalized combining weights
        """
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias

        if HAS_TORCH and isinstance(x, torch.Tensor):
            x_flat = x.reshape(-1, self.hidden_size).to(torch.float32)
            w = weight.to(torch.float32) if isinstance(weight, torch.Tensor) else torch.as_tensor(weight, dtype=torch.float32)

            logits = F.linear(x_flat, w)                    # [N, num_experts]
            scores = torch.sigmoid(logits)                  # combining weights

            scores_for_choice = scores
            if bias is not None:
                b = bias.to(torch.float32) if isinstance(bias, torch.Tensor) else torch.as_tensor(bias, dtype=torch.float32)
                scores_for_choice = scores + b.unsqueeze(0)

            _, topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=True)
            topk_weights = torch.gather(scores, dim=-1, index=topk_indices)

            if self.top_k > 1 and self.renormalize:
                denom = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
                topk_weights = topk_weights / denom

            topk_weights = topk_weights * self.routed_scaling_factor
            return topk_indices, topk_weights
        else:
            # NumPy / Reference CPU implementation
            x_flat = np.asarray(x, dtype=np.float32).reshape(-1, self.hidden_size)
            weight_np = weight.detach().to(torch.float32).cpu().numpy() if (HAS_TORCH and isinstance(weight, torch.Tensor)) else np.asarray(weight, dtype=np.float32)
            bias_np = bias.detach().to(torch.float32).cpu().numpy() if (HAS_TORCH and isinstance(bias, torch.Tensor)) else (np.asarray(bias, dtype=np.float32) if bias is not None else None)

            logits = np.matmul(x_flat, weight_np.T)
            raw_scores = 1.0 / (1.0 + np.exp(-logits))

            # Bias steers selection only, and applies to the scores rather than the logits
            scores_for_choice = raw_scores + bias_np if bias_np is not None else raw_scores

            topk_indices = np.argsort(-scores_for_choice, axis=-1)[:, :self.top_k]

            rows = np.arange(x_flat.shape[0])[:, None]
            topk_weights = raw_scores[rows, topk_indices]

            if self.top_k > 1 and self.renormalize:
                denom = np.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
                topk_weights = topk_weights / denom
                
            topk_weights = topk_weights * self.routed_scaling_factor
            return topk_indices, topk_weights

    @staticmethod
    def get_active_expert_set(topk_indices: Union["torch.Tensor", np.ndarray]) -> List[int]:
        """
        Extract unique active expert IDs across the entire batch.
        Enables lazy-loading ONLY the experts needed for this forward pass!
        """
        if HAS_TORCH and isinstance(topk_indices, torch.Tensor):
            unique_experts = torch.unique(topk_indices).cpu().tolist()
        else:
            unique_experts = np.unique(topk_indices).tolist()
        return sorted([int(e) for e in unique_experts])
