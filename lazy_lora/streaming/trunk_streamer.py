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


KDA_TENSORS = [
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
    "f_a_proj.weight", "f_b_proj.weight", "dt_bias", "A_log",
    "b_proj.weight", "g_proj.weight", "o_norm.weight", "o_proj.weight",
]

MLA_TENSORS = [
    "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
    "g_proj.weight", "o_proj.weight",
]


class AttentionWeights:
    """Namespace of one layer's attention tensors, named as the math expects them."""

    def __init__(self, layer_idx: int, is_kda: bool):
        self.layer_idx = layer_idx
        self.is_kda = is_kda

    def __repr__(self) -> str:
        kind = "KDA" if self.is_kda else "MLA"
        return f"<AttentionWeights layer={self.layer_idx} {kind}>"


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

        # Block-residual gates (attn_res_block_size mechanism)
        res = {}
        for name in ("self_attention_res_norm", "self_attention_res_proj",
                     "mlp_res_norm", "mlp_res_proj"):
            res[name] = self.mmap_streamer.load_tensor(f"{prefix}{name}.weight", target_device=self.device)
        q_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.q_proj.weight", target_device=self.device)
        k_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.k_proj.weight", target_device=self.device)
        v_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.v_proj.weight", target_device=self.device)
        o_proj = self.mmap_streamer.load_tensor(f"{prefix}self_attn.o_proj.weight", target_device=self.device)

        d_hidden = self.hidden_size
        if in_norm is None or q_proj is None:
            # Synthetic default for testing/profiling
            if HAS_TORCH:
                in_norm = torch.ones(d_hidden, dtype=torch.bfloat16, device=self.device)
                post_norm = torch.ones(d_hidden, dtype=torch.bfloat16, device=self.device)
                q_proj = torch.randn(d_hidden, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
                k_proj = torch.randn(d_hidden, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
                v_proj = torch.randn(d_hidden, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
                o_proj = torch.randn(d_hidden, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
            else:
                in_norm = np.ones(d_hidden, dtype=np.float32)
                post_norm = np.ones(d_hidden, dtype=np.float32)
                q_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)
                k_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)
                v_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)
                o_proj = (np.random.randn(d_hidden, d_hidden) * 0.02).astype(np.float32)

        for name, value in list(res.items()):
            if value is not None:
                continue
            shape = (1, d_hidden) if name.endswith("_proj") else (d_hidden,)
            if HAS_TORCH:
                res[name] = torch.ones(shape, dtype=torch.bfloat16, device=self.device)
            else:
                res[name] = np.ones(shape, dtype=np.float32)

        self._current_bundle = TrunkWeightBundle(
            layer_idx=layer_idx,
            input_layernorm=in_norm,
            post_attention_layernorm=post_norm,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )
        for name, value in res.items():
            setattr(self._current_bundle, name, value)
        return self._current_bundle

    def load_attention_weights(
        self,
        layer_idx: int,
        is_kda: bool,
        num_heads: int = 96,
        head_dim: int = 128,
        conv_kernel: int = 4,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
    ) -> "AttentionWeights":
        """
        Stream the attention tensors of one layer, picking the set that matches its type.

        Missing tensors fall back to synthetic ones with the correct shapes so that the
        mock test configurations keep working without the 1.45 TB checkpoint.
        """
        w = AttentionWeights(layer_idx, is_kda)
        prefix = f"model.layers.{layer_idx}.self_attn."
        names = KDA_TENSORS if is_kda else MLA_TENSORS

        missing = []
        for name in names:
            attr = name.replace(".weight", "")
            tensor = self.mmap_streamer.load_tensor(prefix + name, target_device=self.device)
            if tensor is None:
                missing.append(attr)
            setattr(w, attr, tensor)

        if missing:
            self._fill_synthetic_attention(
                w, missing, num_heads, head_dim, conv_kernel, q_lora_rank, kv_lora_rank,
                qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            )
        return w

    def _fill_synthetic_attention(
        self, w, missing, num_heads, head_dim, conv_kernel, q_lora_rank, kv_lora_rank,
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
    ) -> None:
        """Synthetic attention tensors with the exact shapes of the real checkpoint."""
        d_hidden = self.hidden_size
        proj = num_heads * head_dim
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        shapes = {
            # KDA
            "q_proj": (proj, d_hidden), "k_proj": (proj, d_hidden), "v_proj": (proj, d_hidden),
            "q_conv1d": (proj, 1, conv_kernel), "k_conv1d": (proj, 1, conv_kernel),
            "v_conv1d": (proj, 1, conv_kernel),
            "f_a_proj": (head_dim, d_hidden), "f_b_proj": (proj, head_dim),
            "dt_bias": (proj,), "A_log": (head_dim,), "b_proj": (num_heads, d_hidden),
            "o_norm": (head_dim,),
            # MLA
            "q_a_proj": (q_lora_rank, d_hidden), "q_a_layernorm": (q_lora_rank,),
            "q_b_proj": (num_heads * q_head_dim, q_lora_rank),
            "kv_a_proj_with_mqa": (kv_lora_rank + qk_rope_head_dim, d_hidden),
            "kv_a_layernorm": (kv_lora_rank,),
            "kv_b_proj": (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank),
            # shared
            "g_proj": (proj, d_hidden), "o_proj": (d_hidden, proj),
        }

        for attr in missing:
            shape = shapes.get(attr)
            if shape is None:
                continue
            if attr in ("dt_bias", "A_log"):
                value = (torch.zeros(shape, dtype=torch.float32, device=self.device)
                         if HAS_TORCH else np.zeros(shape, dtype=np.float32))
            elif attr.endswith("norm") or attr.endswith("layernorm"):
                value = (torch.ones(shape, dtype=torch.bfloat16, device=self.device)
                         if HAS_TORCH else np.ones(shape, dtype=np.float32))
            elif HAS_TORCH:
                value = torch.randn(shape, dtype=torch.bfloat16, device=self.device) * 0.02
            else:
                value = (np.random.randn(*shape) * 0.02).astype(np.float32)
            setattr(w, attr, value)

    def release_layer_trunk(self) -> None:
        """Evict current layer trunk weights from RAM/VRAM."""
        self._current_bundle = None
        if HAS_TORCH and torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
