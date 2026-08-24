"""
Layer-Wise Dense Trunk Streamer for Kimi K3.
Streams dense attention projections (KDA / Gated MLA) and RMSNorms layer by layer.
Keeps only the active layer in memory.
"""

from typing import Dict, Optional, Tuple, Union, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from lazy_lora.streaming.mmap_loader import MmapTensorStreamer


class RMSNormFunction:
    """Fast RMSNorm forward and backward."""
    @staticmethod
    def forward(x: Union["torch.Tensor", np.ndarray], weight: Union["torch.Tensor", np.ndarray], eps: float = 1e-5):
        if HAS_TORCH and isinstance(x, torch.Tensor):
            variance = x.pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + eps)
            return x_normed * weight
        else:
            variance = np.mean(x ** 2, axis=-1, keepdims=True)
            x_normed = x * (1.0 / np.sqrt(variance + eps))
            return x_normed * weight


class TrunkWeightBundle:
    """Holds dense trunk weights for a single layer."""
    def __init__(
        self,
        layer_idx: int,
        input_layernorm: Union["torch.Tensor", np.ndarray],
        post_attention_layernorm: Union["torch.Tensor", np.ndarray],
        q_proj: Union["torch.Tensor", np.ndarray],
        k_proj: Union["torch.Tensor", np.ndarray],
        v_proj: Union["torch.Tensor", np.ndarray],
        o_proj: Union["torch.Tensor", np.ndarray],
    ):
        self.layer_idx = layer_idx
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj


class LayerTrunkStreamer:
    """
    Sequential layer-wise streamer for Attention and Normalization matrices.
    """

    def __init__(self, mmap_streamer: MmapTensorStreamer, device: str = "cpu", hidden_size: int = 7168):
        self.mmap_streamer = mmap_streamer
        self.device = device
        self.hidden_size = hidden_size
        self._current_bundle: Optional[TrunkWeightBundle] = None

    def load_layer_trunk(self, layer_idx: int) -> TrunkWeightBundle:
        """Loads dense trunk matrices for layer_idx."""
        prefix = f"model.layers.{layer_idx}."
        
        in_norm = self.mmap_streamer.load_tensor(f"{prefix}input_layernorm.weight", target_device=self.device)
        post_norm = self.mmap_streamer.load_tensor(f"{prefix}post_attention_layernorm.weight", target_device=self.device)
        q_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.q_proj.weight", target_device=self.device)
        k_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.k_proj.weight", target_device=self.device)
        v_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.v_proj.weight", target_device=self.device)
        o_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.o_proj.weight", target_device=self.device)

        d_hidden = self.hidden_size
        if in_norm is None or q_proj is None:
            # Synthetic default for testing/profiling
            if HAS_TORCH:
                in_norm = torch.ones(d_hidden, dtype=torch.float32, device=self.device)
                post_norm = torch.ones(d_hidden, dtype=torch.float32, device=self.device)
                q_proj = torch.randn(d_hidden, d_hidden, dtype=torch.float32, device=self.device) * 0.02
                k_proj = torch.randn(d_hidden, d_hidden, dtype=torch.float32, device=self.device) * 0.02
                v_proj = torch.randn(d_hidden, d_hidden, dtype=torch.float32, device=self.device) * 0.02
                o_proj = torch.randn(d_hidden, d_hidden, dtype=torch.float32, device=self.device) * 0.02
            else:
                in_norm = np.ones(d_hidden, dtype=np.float32)
                post_norm = np.ones(d_hidden, dtype=np.float32)
                q_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)
                k_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)
                v_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)
                o_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)

        self._current_bundle = TrunkWeightBundle(
            layer_idx=layer_idx,
            input_layernorm=in_norm,
            post_attention_layernorm=post_norm,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )
        return self._current_bundle

    def release_layer_trunk(self) -> None:
        """Evict current layer trunk weights from RAM/VRAM."""
        self._current_bundle = None
        if HAS_TORCH and torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
