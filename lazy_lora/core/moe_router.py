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
        self, x: Union["torch.Tensor", np.ndarray]
    ) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
        """
        Forward pass for MoE routing.
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] or [N, hidden_size]
        Returns:
            topk_indices: [N, top_k] int64 indices of selected experts
            topk_weights: [N, top_k] float32 normalized combining weights
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            x_flat = x.view(-1, self.hidden_size)
            if self.weight.dtype != x_flat.dtype:
                x_flat = x_flat.to(self.weight.dtype)

            # Linear projection: logits = x @ W.T
            logits = F.linear(x_flat, self.weight)  # [N, num_experts]
            
            # Unbiased sigmoid scores for combining
            raw_scores = torch.sigmoid(logits)  # [N, num_experts]
            
            # Selection logits (steered by bias if present)
            if self.bias is not None:
                bias_val = self.bias.to(logits.dtype) if isinstance(self.bias, torch.Tensor) else self.bias
                selection_logits = logits + bias_val
            else:
                selection_logits = logits
                
            # Top-k selection based on selection scores
            _, topk_indices = torch.topk(selection_logits, k=self.top_k, dim=-1, sorted=True)  # [N, top_k]
            
            # Gather unbiased scores for the selected top-k experts
            topk_weights = torch.gather(raw_scores, dim=-1, index=topk_indices)  # [N, top_k]
            
            # Renormalize weights
            if self.renormalize:
                denom = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
                topk_weights = topk_weights / denom
                
            topk_weights = topk_weights * self.routed_scaling_factor
            return topk_indices, topk_weights
        else:
            # NumPy / Reference CPU implementation
            x_flat = np.asarray(x, dtype=np.float32).reshape(-1, self.hidden_size)
            weight_np = self.weight.detach().to(torch.float32).cpu().numpy() if (HAS_TORCH and isinstance(self.weight, torch.Tensor)) else np.asarray(self.weight, dtype=np.float32)
            bias_np = self.bias.detach().to(torch.float32).cpu().numpy() if (HAS_TORCH and isinstance(self.bias, torch.Tensor)) else (np.asarray(self.bias, dtype=np.float32) if self.bias is not None else None)

            logits = np.matmul(x_flat, weight_np.T)
            raw_scores = 1.0 / (1.0 + np.exp(-logits))
            
            if bias_np is not None:
                selection_logits = logits + bias_np
            else:
                selection_logits = logits
                
            # Top-k selection
            topk_indices = np.argsort(-selection_logits, axis=-1)[:, :self.top_k]
            
            # Gather weights
            rows = np.arange(x_flat.shape[0])[:, None]
            topk_weights = raw_scores[rows, topk_indices]
            
            if self.renormalize:
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
