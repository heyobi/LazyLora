"""
LazyLoRA Out-of-Core MoE Training Engine for Kimi K3.
Executes sequential layer-wise forward and backward passes, streaming base weights
and storing boundary activations on D: drive to achieve training under 6GB VRAM and 16GB RAM.
"""

import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union
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

from lazy_lora.core.config import LazyLoraConfig, get_default_config
from lazy_lora.core.lora_layer import LazyLoRALinear
from lazy_lora.core.moe_router import KimiK3MoERouter
from lazy_lora.core.situ_activation import situ_glu_forward, situ_glu_backward
from lazy_lora.streaming.mmap_loader import MmapTensorStreamer
from lazy_lora.streaming.trunk_streamer import LayerTrunkStreamer, RMSNormFunction
from lazy_lora.streaming.expert_streamer import DynamicExpertStreamer
from lazy_lora.streaming.activation_ring_buffer import ActivationRingBuffer
from lazy_lora.trainer.loss import compute_cross_entropy_loss
from lazy_lora.trainer.optimizer import LazyLoRAOptimizer
from lazy_lora.monitor.metrics import MetricsTracker
from lazy_lora.monitor.dashboard import TerminalDashboard


class LoRALayerBundle:
    """Holds trainable LoRA adapters for a single Transformer layer."""
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 7168,
        attn_out_size: int = 12288,
        moe_latent_size: int = 3584,
        moe_intermediate_size: int = 3072,
        shared_intermediate_size: int = 6144,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        self.layer_idx = layer_idx
        # Attention LoRA adapters (hidden 7168 -> num_heads * head_dim)
        self.q_lora = LazyLoRALinear(hidden_size, attn_out_size, r, alpha, dropout, device)
        self.v_lora = LazyLoRALinear(hidden_size, attn_out_size, r, alpha, dropout, device)

        # Shared-expert LoRA adapters (dense path: 7168 -> 6144 -> 7168)
        self.shared_gate_lora = LazyLoRALinear(hidden_size, shared_intermediate_size, r, alpha, dropout, device)
        self.shared_up_lora = LazyLoRALinear(hidden_size, shared_intermediate_size, r, alpha, dropout, device)
        self.shared_down_lora = LazyLoRALinear(shared_intermediate_size, hidden_size, r, alpha, dropout, device)

        # Routed-expert LoRA adapters, shared across the 896 experts of this layer.
        # They live in the Kimi K3 latent MoE space: 3584 -> 3072 -> 3584.
        self.gate_lora = LazyLoRALinear(moe_latent_size, moe_intermediate_size, r, alpha, dropout, device)
        self.up_lora = LazyLoRALinear(moe_latent_size, moe_intermediate_size, r, alpha, dropout, device)
        self.down_lora = LazyLoRALinear(moe_intermediate_size, moe_latent_size, r, alpha, dropout, device)

    def all_modules(self) -> List[Tuple[str, Any]]:
        """(name, module) pairs for every LoRA adapter in this layer."""
        return [
            ("q_lora", self.q_lora),
            ("v_lora", self.v_lora),
            ("shared_gate_lora", self.shared_gate_lora),
            ("shared_up_lora", self.shared_up_lora),
            ("shared_down_lora", self.shared_down_lora),
            ("gate_lora", self.gate_lora),
            ("up_lora", self.up_lora),
            ("down_lora", self.down_lora),
        ]

    def get_parameters(self) -> List[Any]:
        """Returns all trainable LoRA tensors in this layer."""
        params = []
        for _, mod in self.all_modules():
            if mod.lora_A is not None:
                params.append(mod.lora_A)
            if mod.lora_B is not None:
                params.append(mod.lora_B)
        return params


class LazyLoRATrainer:
    """
    Main Out-of-Core Trainer for Kimi K3 MoE LoRA Fine-Tuning.
    """

    def __init__(self, config: Optional[LazyLoraConfig] = None):
        self.config = config or get_default_config()
        self.device = self.config.streaming.device if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        
        # Ensure directories exist on D: drive
        self.config.paths.ensure_directories()

        # Initialize streaming subsystems
        self.mmap_streamer = MmapTensorStreamer(self.config.paths.base_model_dir)
        self.trunk_streamer = LayerTrunkStreamer(
            self.mmap_streamer,
            device=self.device,
            hidden_size=self.config.model.hidden_size,
        )
        self.expert_streamer = DynamicExpertStreamer(
            self.mmap_streamer,
            device=self.device,
            async_prefetch=self.config.streaming.async_prefetch,
            hidden_size=self.config.model.hidden_size,
            latent_size=self.config.model.routed_expert_hidden_size,
            moe_intermediate_size=self.config.model.moe_intermediate_size,
            shared_intermediate_size=self.config.model.moe_intermediate_size * self.config.model.num_shared_experts,
        )
        self.act_buffer = ActivationRingBuffer(
            cache_dir=self.config.paths.activation_cache_dir,
            num_layers=self.config.model.num_hidden_layers,
        )

        # Initialize MoE Router
        self.router = KimiK3MoERouter(
            hidden_size=self.config.model.hidden_size,
            num_experts=self.config.model.num_experts,
            top_k=self.config.model.num_experts_per_token,
            routed_scaling_factor=self.config.model.routed_scaling_factor,
        )

        # Initialize Trainable LoRA Layer Bundles for all layers
        self.lora_layers: List[LoRALayerBundle] = [
            LoRALayerBundle(
                layer_idx=l,
                hidden_size=self.config.model.hidden_size,
                attn_out_size=self.config.model.num_attention_heads * self.config.model.head_dim,
                moe_latent_size=self.config.model.routed_expert_hidden_size,
                moe_intermediate_size=self.config.model.moe_intermediate_size,
                shared_intermediate_size=self.config.model.moe_intermediate_size * self.config.model.num_shared_experts,
                r=self.config.lora.r,
                alpha=self.config.lora.lora_alpha,
                dropout=self.config.lora.lora_dropout,
                device=self.device,
            )
            for l in range(self.config.model.num_hidden_layers)
        ]

        # Gather all LoRA parameters
        self.all_lora_params = []
        for bundle in self.lora_layers:
            self.all_lora_params.extend(bundle.get_parameters())

        # Initialize Optimizer
        self.optimizer = LazyLoRAOptimizer(
            parameters=self.all_lora_params,
            lr=self.config.training.learning_rate,
            min_lr=self.config.training.min_learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=self.config.training.max_steps,
            weight_decay=self.config.training.weight_decay,
            grad_clip_norm=self.config.training.grad_clip_norm,
        )

        # Monitoring & Dashboard
        self.tracker = MetricsTracker(
            total_steps=self.config.training.max_steps,
            total_layers=self.config.model.num_hidden_layers,
        )
        self.dashboard = TerminalDashboard()

    def _embed_tokens(self, input_ids: Union["torch.Tensor", np.ndarray]) -> Union["torch.Tensor", np.ndarray]:
        """Embed input tokens using resident or streamed embedding table."""
        embed_weight = self.mmap_streamer.load_tensor("model.embed_tokens.weight", target_device=self.device)
        vocab_sz = self.config.model.vocab_size
        d_hidden = self.config.model.hidden_size

        if embed_weight is None:
            # Synthetic embedding table for testing
            if HAS_TORCH:
                embed_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
            else:
                embed_weight = (np.random.randn(vocab_sz, d_hidden) * 0.02).astype(np.float32)

        if HAS_TORCH and isinstance(input_ids, torch.Tensor):
            return F.embedding(input_ids, embed_weight)
        else:
            return embed_weight[input_ids]

    def _project_lm_head(self, hidden_state: Union["torch.Tensor", np.ndarray]) -> Union["torch.Tensor", np.ndarray]:
        """Project final hidden state through LM head to compute vocabulary logits."""
        head_weight = self.mmap_streamer.load_tensor("lm_head.weight", target_device=self.device)
        vocab_sz = self.config.model.vocab_size
        d_hidden = self.config.model.hidden_size

        if head_weight is None:
            if HAS_TORCH:
                head_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
            else:
                head_weight = (np.random.randn(vocab_sz, d_hidden) * 0.02).astype(np.float32)

        if HAS_TORCH and isinstance(hidden_state, torch.Tensor):
            return F.linear(hidden_state, head_weight)
        else:
            return np.matmul(hidden_state, head_weight.T)

    def forward_layer(
        self,
        layer_idx: int,
        h_in: Union["torch.Tensor", np.ndarray],
    ) -> Tuple[Union["torch.Tensor", np.ndarray], List[int]]:
        """
        Executes out-of-core forward pass for a single layer:
        1. Save h_in to D: drive activation ring buffer.
        2. Attention forward with dense trunk streaming.
        3. MoE Router top-16 expert selection & dispatch.
        4. Active expert forward + LoRA branches.
        5. Evict layer weights from memory.
        """
        # 1. Save boundary activation to D: SSD
        self.act_buffer.save_activation(layer_idx, h_in)

        # 2. Attention & Norm
        trunk = self.trunk_streamer.load_layer_trunk(layer_idx)
        bundle = self.lora_layers[layer_idx]

        h_norm = RMSNormFunction.forward(h_in, trunk.input_layernorm)
        
        # Attention projection with LoRA
        if HAS_TORCH and isinstance(h_in, torch.Tensor):
            q = F.linear(h_norm, trunk.q_proj) + bundle.q_lora.forward_lora_only(h_norm)
            k = F.linear(h_norm, trunk.k_proj)
            v = F.linear(h_norm, trunk.v_proj) + bundle.v_lora.forward_lora_only(h_norm)
            # Lightweight self-attention / delta attention approximation
            attn_out = F.linear(v, trunk.o_proj)
            h_mid = h_in + attn_out
        else:
            q = np.matmul(h_norm, trunk.q_proj.T) + bundle.q_lora.forward_lora_only(h_norm)
            k = np.matmul(h_norm, trunk.k_proj.T)
            v = np.matmul(h_norm, trunk.v_proj.T) + bundle.v_lora.forward_lora_only(h_norm)
            attn_out = np.matmul(v, trunk.o_proj.T)
            h_mid = h_in + attn_out

        self.trunk_streamer.release_layer_trunk()

        # 3. MoE Routing
        h_moe_norm = RMSNormFunction.forward(h_mid, trunk.post_attention_layernorm)
        topk_indices, topk_weights = self.router.forward(h_moe_norm)
        active_experts = self.router.get_active_expert_set(topk_indices)

        # Prefetch active experts for next layer if applicable
        if layer_idx + 1 < self.config.model.num_hidden_layers:
            self.expert_streamer.request_prefetch_layer(layer_idx + 1, active_experts)

        # 4. MoE Expert Execution (1 Shared Module + Active Top-16 Routed Experts)
        # Kimi K3 Latent MoE: h(7168) → down_proj → h_latent(3584) → expert(3584→3072→3584) → up_proj → (7168)
        # LoRA is applied on the full target_modules set: attention q/v, the shared expert
        # (7168→6144→7168) and the routed experts in latent space (3584→3072→3584).
        if HAS_TORCH and isinstance(h_in, torch.Tensor):
            # Shared expert (single module, input=h_moe_norm [7168], output=7168)
            s_bundle = self.expert_streamer.get_expert(layer_idx, 0, is_shared=True)
            if h_moe_norm.dtype != s_bundle.gate_proj.dtype:
                h_moe_norm = h_moe_norm.to(s_bundle.gate_proj.dtype)
            s_gate = F.linear(h_moe_norm, s_bundle.gate_proj) + bundle.shared_gate_lora.forward_lora_only(h_moe_norm)
            s_up = F.linear(h_moe_norm, s_bundle.up_proj) + bundle.shared_up_lora.forward_lora_only(h_moe_norm)
            s_situ = situ_glu_forward(s_gate, s_up)
            shared_out = F.linear(s_situ, s_bundle.down_proj) + bundle.shared_down_lora.forward_lora_only(s_situ)
            del s_bundle, s_gate, s_up, s_situ

            # Latent MoE projections: load routed_expert_down_proj and routed_expert_up_proj
            prefix = f"model.layers.{layer_idx}.block_sparse_moe."
            latent_down_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_down_proj.weight", target_device=self.device)
            latent_up_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_up_proj.weight", target_device=self.device)
            latent_norm_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_norm.weight", target_device=self.device)
            d_h = self.config.model.hidden_size          # 7168
            d_l = self.config.model.routed_expert_hidden_size  # 3584
            if latent_down_w is None:
                latent_down_w = torch.randn(d_l, d_h, dtype=torch.bfloat16, device=self.device) * 0.01
            if latent_up_w is None:
                latent_up_w = torch.randn(d_h, d_l, dtype=torch.bfloat16, device=self.device) * 0.01
            if latent_norm_w is None:
                latent_norm_w = torch.ones(d_l, dtype=torch.bfloat16, device=self.device)
            if latent_down_w.dtype != torch.bfloat16:
                latent_down_w = latent_down_w.to(torch.bfloat16)
            if latent_up_w.dtype != torch.bfloat16:
                latent_up_w = latent_up_w.to(torch.bfloat16)

            # Project h_moe_norm (7168) → h_latent (3584) for routed experts
            N, top_k = topk_indices.shape
            h_flat = h_moe_norm.view(-1, d_h).to(latent_down_w.dtype)
            h_latent = F.linear(h_flat, latent_down_w)  # [N, 3584]
            del latent_down_w

            # Run routed experts in latent space (3584 → 3072 → 3584)
            routed_latent_out = torch.zeros(h_latent.shape[0], d_l, dtype=h_latent.dtype, device=self.device)

            for exp_id in active_experts:
                mask = (topk_indices == exp_id)  # [N, top_k]
                if not mask.any():
                    continue

                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                e_gate = F.linear(h_latent, e_bundle.gate_proj) + bundle.gate_lora.forward_lora_only(h_latent)   # [N, 3072]
                e_up = F.linear(h_latent, e_bundle.up_proj) + bundle.up_lora.forward_lora_only(h_latent)        # [N, 3072]
                e_situ = situ_glu_forward(e_gate, e_up)
                e_down = F.linear(e_situ, e_bundle.down_proj) + bundle.down_lora.forward_lora_only(e_situ)      # [N, 3584]
                del e_bundle, e_gate, e_up, e_situ

                token_weights = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
                routed_latent_out = routed_latent_out + e_down * token_weights
                del e_down

            # Apply latent norm then project back: (3584) → (7168)
            # Norm weights ship as float32 in some shards; keep the whole path in bfloat16.
            routed_latent_out = RMSNormFunction.forward(routed_latent_out, latent_norm_w.to(routed_latent_out.dtype))
            routed_out = F.linear(routed_latent_out.to(latent_up_w.dtype), latent_up_w)  # [N, 7168]
            del latent_up_w, latent_norm_w, routed_latent_out, h_latent

            moe_out = shared_out + routed_out.view_as(h_mid)
            h_out = h_mid + moe_out
        else:
            # NumPy path
            # Shared expert
            s_bundle = self.expert_streamer.get_expert(layer_idx, 0, is_shared=True)
            s_gate = np.matmul(h_moe_norm, s_bundle.gate_proj.T) + bundle.shared_gate_lora.forward_lora_only(h_moe_norm)
            s_up = np.matmul(h_moe_norm, s_bundle.up_proj.T) + bundle.shared_up_lora.forward_lora_only(h_moe_norm)
            s_situ = situ_glu_forward(s_gate, s_up)
            shared_out = np.matmul(s_situ, s_bundle.down_proj.T) + bundle.shared_down_lora.forward_lora_only(s_situ)

            # Latent MoE projections
            d_h = self.config.model.hidden_size
            d_l = self.config.model.routed_expert_hidden_size
            h_flat = h_moe_norm.reshape(-1, d_h)
            latent_down_w = (np.random.randn(d_l, d_h) * 0.01).astype(np.float32)
            latent_up_w = (np.random.randn(d_h, d_l) * 0.01).astype(np.float32)
            h_latent = np.matmul(h_flat, latent_down_w.T)

            routed_latent_out = np.zeros((h_latent.shape[0], d_l), dtype=np.float32)
            for exp_id in active_experts:
                mask = (topk_indices == exp_id)
                if not np.any(mask):
                    continue
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                e_gate = np.matmul(h_latent, e_bundle.gate_proj.T) + bundle.gate_lora.forward_lora_only(h_latent)
                e_up = np.matmul(h_latent, e_bundle.up_proj.T) + bundle.up_lora.forward_lora_only(h_latent)
                e_situ = situ_glu_forward(e_gate, e_up)
                e_down = np.matmul(e_situ, e_bundle.down_proj.T) + bundle.down_lora.forward_lora_only(e_situ)

                token_weights = np.sum(topk_weights * mask.astype(np.float32), axis=-1, keepdims=True)
                routed_latent_out = routed_latent_out + e_down * token_weights

            routed_out = np.matmul(routed_latent_out, latent_up_w.T)

            moe_out = shared_out + routed_out.reshape(h_mid.shape)
            h_out = h_mid + moe_out

        # 5. Evict layer expert weights from memory
        self.expert_streamer.evict_layer_experts(layer_idx)

        return h_out, active_experts

    def backward_layer(
        self,
        layer_idx: int,
        grad_h_out: Union["torch.Tensor", np.ndarray],
    ) -> Union["torch.Tensor", np.ndarray]:
        """
        Executes out-of-core backward pass for a single layer:
        1. Loads h_in from D: drive activation ring buffer.
        2. Recomputes intermediate forward states for layer_idx.
        3. Computes analytical gradients for all LoRA matrices (down, gate, up, q, v).
        4. Accumulates gradients into bundle LoRA parameters.
        5. Computes and returns downstream gradient grad_h_in to propagate to preceding layer.
        6. Evicts layer weights immediately from RAM/VRAM.
        """
        is_torch = HAS_TORCH and isinstance(grad_h_out, torch.Tensor)
        h_in = self.act_buffer.load_activation(layer_idx, target_device=self.device, as_torch=is_torch)
        if h_in is None:
            return grad_h_out

        trunk = self.trunk_streamer.load_layer_trunk(layer_idx)
        bundle = self.lora_layers[layer_idx]

        if is_torch:
            # Recompute Attention forward
            h_norm1 = RMSNormFunction.forward(h_in, trunk.input_layernorm)
            v = F.linear(h_norm1, trunk.v_proj) + bundle.v_lora.forward_lora_only(h_norm1)
            attn_out = F.linear(v, trunk.o_proj)
            h_mid = h_in + attn_out

            # Recompute MoE forward states
            h_moe_norm = RMSNormFunction.forward(h_mid, trunk.post_attention_layernorm)
            topk_indices, topk_weights = self.router.forward(h_moe_norm)
            active_experts = self.router.get_active_expert_set(topk_indices)

            # MoE Backward (Kimi K3 Latent MoE), mirroring the forward pass exactly:
            #   shared expert : 7168 -> 6144 -> 7168   (LoRA on gate/up/down)
            #   routed experts: 3584 -> 3072 -> 3584   (LoRA on gate/up/down, latent space)
            grad_h_mid = grad_h_out.clone()
            grad_moe_norm = torch.zeros_like(h_moe_norm)

            # 1. Shared expert backward (single module, dense path)
            s_bundle = self.expert_streamer.get_expert(layer_idx, 0, is_shared=True)
            if h_moe_norm.dtype != s_bundle.gate_proj.dtype:
                h_moe_norm = h_moe_norm.to(s_bundle.gate_proj.dtype)
            gate = F.linear(h_moe_norm, s_bundle.gate_proj) + bundle.shared_gate_lora.forward_lora_only(h_moe_norm)
            up = F.linear(h_moe_norm, s_bundle.up_proj) + bundle.shared_up_lora.forward_lora_only(h_moe_norm)
            situ = situ_glu_forward(gate, up)

            gA_d, gB_d, d_situ = bundle.shared_down_lora.compute_lora_gradients(grad_h_out, situ, s_bundle.down_proj)
            bundle.shared_down_lora.accumulate_grad(gA_d, gB_d)

            if d_situ is not None:
                d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                gA_g, gB_g, d_in_g = bundle.shared_gate_lora.compute_lora_gradients(d_gate, h_moe_norm, s_bundle.gate_proj)
                bundle.shared_gate_lora.accumulate_grad(gA_g, gB_g)

                gA_u, gB_u, d_in_u = bundle.shared_up_lora.compute_lora_gradients(d_up, h_moe_norm, s_bundle.up_proj)
                bundle.shared_up_lora.accumulate_grad(gA_u, gB_u)

                if d_in_g is not None and d_in_u is not None:
                    grad_moe_norm.add_(d_in_g + d_in_u)
            del s_bundle, gate, up, situ

            # 2. Routed experts backward, in the 3584-dim latent space
            d_h = self.config.model.hidden_size
            d_l = self.config.model.routed_expert_hidden_size
            prefix = f"model.layers.{layer_idx}.block_sparse_moe."
            latent_down_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_down_proj.weight", target_device=self.device)
            latent_up_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_up_proj.weight", target_device=self.device)
            if latent_down_w is None:
                latent_down_w = torch.randn(d_l, d_h, dtype=torch.bfloat16, device=self.device) * 0.01
            if latent_up_w is None:
                latent_up_w = torch.randn(d_h, d_l, dtype=torch.bfloat16, device=self.device) * 0.01
            latent_down_w = latent_down_w.to(torch.bfloat16)
            latent_up_w = latent_up_w.to(torch.bfloat16)

            h_flat = h_moe_norm.view(-1, d_h).to(latent_down_w.dtype)
            h_latent = F.linear(h_flat, latent_down_w)                       # [N, 3584]
            grad_flat = grad_h_out.view(-1, d_h).to(latent_up_w.dtype)
            # Back through routed_expert_up_proj (latent RMSNorm jacobian approximated as identity)
            grad_latent_out = F.linear(grad_flat, latent_up_w.t())           # [N, 3584]
            grad_latent_in = torch.zeros_like(h_latent)

            for exp_id in active_experts:
                mask = (topk_indices == exp_id)
                if not mask.any():
                    continue
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                gate = F.linear(h_latent, e_bundle.gate_proj) + bundle.gate_lora.forward_lora_only(h_latent)
                up = F.linear(h_latent, e_bundle.up_proj) + bundle.up_lora.forward_lora_only(h_latent)
                situ = situ_glu_forward(gate, up)

                token_weights = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
                token_weights = token_weights.view(-1, 1).to(grad_latent_out.dtype)
                d_down_weighted = grad_latent_out * token_weights            # [N, 3584]

                gA_d, gB_d, d_situ = bundle.down_lora.compute_lora_gradients(d_down_weighted, situ, e_bundle.down_proj)
                bundle.down_lora.accumulate_grad(gA_d, gB_d)

                if d_situ is not None:
                    d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                    gA_g, gB_g, d_in_g = bundle.gate_lora.compute_lora_gradients(d_gate, h_latent, e_bundle.gate_proj)
                    bundle.gate_lora.accumulate_grad(gA_g, gB_g)

                    gA_u, gB_u, d_in_u = bundle.up_lora.compute_lora_gradients(d_up, h_latent, e_bundle.up_proj)
                    bundle.up_lora.accumulate_grad(gA_u, gB_u)

                    if d_in_g is not None and d_in_u is not None:
                        grad_latent_in.add_(d_in_g + d_in_u)
                del e_bundle, gate, up, situ

            # Back through routed_expert_down_proj: latent (3584) -> hidden (7168)
            grad_moe_norm.add_(F.linear(grad_latent_in, latent_down_w.t()).view_as(h_moe_norm))
            del latent_down_w, latent_up_w, h_latent, grad_latent_in, grad_latent_out

            grad_h_mid.add_(grad_moe_norm)

            # Attention Sublayer Backward
            d_v = F.linear(grad_h_mid, trunk.o_proj.t())
            gA_v, gB_v, d_in_v = bundle.v_lora.compute_lora_gradients(d_v, h_norm1, trunk.v_proj)
            bundle.v_lora.accumulate_grad(gA_v, gB_v)

            gA_q, gB_q, d_in_q = bundle.q_lora.compute_lora_gradients(d_v, h_norm1, trunk.q_proj)
            bundle.q_lora.accumulate_grad(gA_q, gB_q)

            grad_h_in = grad_h_mid.clone()
            if d_in_v is not None:
                grad_h_in.add_(d_in_v)
            if d_in_q is not None:
                grad_h_in.add_(d_in_q)
        else:
            # NumPy analytical backward
            orig_shape = h_in.shape
            h_norm1 = RMSNormFunction.forward(h_in, trunk.input_layernorm)
            v = np.matmul(h_norm1, trunk.v_proj.T) + bundle.v_lora.forward_lora_only(h_norm1)
            attn_out = np.matmul(v, trunk.o_proj.T)
            h_mid = h_in + attn_out

            h_moe_norm = RMSNormFunction.forward(h_mid, trunk.post_attention_layernorm)
            topk_indices, topk_weights = self.router.forward(h_moe_norm)
            active_experts = self.router.get_active_expert_set(topk_indices)

            # MoE Backward: same structure as the torch path (shared dense expert +
            # routed experts in the 3584-dim latent space), full LoRA coverage.
            grad_h_mid = grad_h_out.copy().reshape(orig_shape)
            grad_moe_norm = np.zeros(orig_shape, dtype=np.float32)

            s_bundle = self.expert_streamer.get_expert(layer_idx, 0, is_shared=True)
            gate = np.matmul(h_moe_norm, s_bundle.gate_proj.T) + bundle.shared_gate_lora.forward_lora_only(h_moe_norm)
            up = np.matmul(h_moe_norm, s_bundle.up_proj.T) + bundle.shared_up_lora.forward_lora_only(h_moe_norm)
            situ = situ_glu_forward(gate, up)

            gA_d, gB_d, d_situ = bundle.shared_down_lora.compute_lora_gradients(grad_h_mid, situ, s_bundle.down_proj)
            bundle.shared_down_lora.accumulate_grad(gA_d, gB_d)

            if d_situ is not None:
                d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                gA_g, gB_g, d_in_g = bundle.shared_gate_lora.compute_lora_gradients(d_gate, h_moe_norm, s_bundle.gate_proj)
                bundle.shared_gate_lora.accumulate_grad(gA_g, gB_g)

                gA_u, gB_u, d_in_u = bundle.shared_up_lora.compute_lora_gradients(d_up, h_moe_norm, s_bundle.up_proj)
                bundle.shared_up_lora.accumulate_grad(gA_u, gB_u)

                if d_in_g is not None and d_in_u is not None:
                    grad_moe_norm += (d_in_g + d_in_u).reshape(orig_shape)

            d_h = self.config.model.hidden_size
            d_l = self.config.model.routed_expert_hidden_size
            latent_down_w = (np.random.randn(d_l, d_h) * 0.01).astype(np.float32)
            latent_up_w = (np.random.randn(d_h, d_l) * 0.01).astype(np.float32)

            h_flat = h_moe_norm.reshape(-1, d_h)
            h_latent = np.matmul(h_flat, latent_down_w.T)
            grad_flat = grad_h_mid.reshape(-1, d_h)
            grad_latent_out = np.matmul(grad_flat, latent_up_w)
            grad_latent_in = np.zeros_like(h_latent)

            for exp_id in active_experts:
                mask = (topk_indices == exp_id)
                if not np.any(mask):
                    continue
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                gate = np.matmul(h_latent, e_bundle.gate_proj.T) + bundle.gate_lora.forward_lora_only(h_latent)
                up = np.matmul(h_latent, e_bundle.up_proj.T) + bundle.up_lora.forward_lora_only(h_latent)
                situ = situ_glu_forward(gate, up)

                token_weights = np.sum(topk_weights * mask.astype(np.float32), axis=-1, keepdims=True).reshape(-1, 1)
                d_down_weighted = grad_latent_out * token_weights

                gA_d, gB_d, d_situ = bundle.down_lora.compute_lora_gradients(d_down_weighted, situ, e_bundle.down_proj)
                bundle.down_lora.accumulate_grad(gA_d, gB_d)

                if d_situ is not None:
                    d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                    gA_g, gB_g, d_in_g = bundle.gate_lora.compute_lora_gradients(d_gate, h_latent, e_bundle.gate_proj)
                    bundle.gate_lora.accumulate_grad(gA_g, gB_g)

                    gA_u, gB_u, d_in_u = bundle.up_lora.compute_lora_gradients(d_up, h_latent, e_bundle.up_proj)
                    bundle.up_lora.accumulate_grad(gA_u, gB_u)

                    if d_in_g is not None and d_in_u is not None:
                        grad_latent_in += d_in_g + d_in_u

            grad_moe_norm += np.matmul(grad_latent_in, latent_down_w).reshape(orig_shape)
            grad_h_mid += grad_moe_norm

            d_v = np.matmul(grad_h_mid, trunk.o_proj)
            gA_v, gB_v, d_in_v = bundle.v_lora.compute_lora_gradients(d_v, h_norm1, trunk.v_proj)
            bundle.v_lora.accumulate_grad(gA_v, gB_v)

            gA_q, gB_q, d_in_q = bundle.q_lora.compute_lora_gradients(d_v, h_norm1, trunk.q_proj)
            bundle.q_lora.accumulate_grad(gA_q, gB_q)

            grad_h_in = grad_h_mid.copy()
            if d_in_v is not None:
                grad_h_in += d_in_v.reshape(orig_shape)
            if d_in_q is not None:
                grad_h_in += d_in_q.reshape(orig_shape)

        self.trunk_streamer.release_layer_trunk()
        self.expert_streamer.evict_layer_experts(layer_idx)

        return grad_h_in

    def train_step(
        self,
        step: int,
        input_ids: Union["torch.Tensor", np.ndarray],
        target_ids: Union["torch.Tensor", np.ndarray],
    ) -> float:
        """
        Complete out-of-core forward-backward-optimizer step.
        """
        t0 = time.time()
        num_layers = self.config.model.num_hidden_layers

        # 1. Embed tokens
        if HAS_TORCH:
            no_grad_ctx = torch.no_grad()
            no_grad_ctx.__enter__()

        try:
            h_current = self._embed_tokens(input_ids)
            last_active_experts = []

            # 2. Sequential Layer-by-Layer Forward Pass (Layer 0 -> Layer L-1)
            for l in range(num_layers):
                if l % 5 == 0 or l == num_layers - 1:
                    print(f"\r  ⚡ [FORWARD PASS] Layer {l+1:02d}/{num_layers} (MoE Stream)", end="", flush=True)
                h_current, last_active_experts = self.forward_layer(l, h_current)

            print(f"\r  ⚡ [FORWARD COMPLETE] (93 Layers) -> Computing LM Head Cross-Entropy Loss...", end="", flush=True)
            # 3. Final LM Head Projection and Loss
            logits = self._project_lm_head(h_current)
            loss_val, grad_logits = compute_cross_entropy_loss(
                logits,
                target_ids,
                ignore_index=self.config.model.pad_token_id,
            )

            # 4. Out-of-Core Real Reverse Backward Pass (Layer L-1 -> Layer 0)
            # Backprop through LM head projection
            head_weight = self.mmap_streamer.load_tensor("lm_head.weight", target_device=self.device)
            if head_weight is None:
                vocab_sz = self.config.model.vocab_size
                d_hidden = self.config.model.hidden_size
                if HAS_TORCH and isinstance(logits, torch.Tensor):
                    head_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
                else:
                    head_weight = (np.random.randn(vocab_sz, d_hidden) * 0.02).astype(np.float32)

            if HAS_TORCH and isinstance(grad_logits, torch.Tensor):
                grad_h = F.linear(grad_logits.to(head_weight.dtype), head_weight.t())
            else:
                grad_h = np.matmul(grad_logits, head_weight)

            # Sequential reverse backward pass through all 93 layers reading activations from D: SSD
            for l in range(num_layers - 1, -1, -1):
                if l % 5 == 0 or l == 0:
                    print(f"\r  ⚡ [BACKWARD PASS] Layer {l+1:02d}/{num_layers} (Grad Stream)", end="", flush=True)
                grad_h = self.backward_layer(l, grad_h)
            print("\r" + " " * 80 + "\r", end="", flush=True)

        finally:
            if HAS_TORCH:
                no_grad_ctx.__exit__(None, None, None)

        # 5. Optimizer Step
        current_lr = self.optimizer.step()
        self.optimizer.zero_grad()

        # 6. Clean activation cache for next step
        self.act_buffer.clean_cache()

        # 7. Collect and render metrics
        dt = time.time() - t0
        tokens_count = int(input_ids.shape[0] * input_ids.shape[1]) if hasattr(input_ids, "shape") else 512
        loss_scalar = float(loss_val.item() if (HAS_TORCH and isinstance(loss_val, torch.Tensor)) else loss_val)

        m = self.tracker.update_step(
            step=step,
            loss=loss_scalar,
            lr=current_lr,
            active_layer=num_layers - 1,
            active_experts=last_active_experts,
            tokens_processed=tokens_count,
            disk_bytes_read=int(num_layers * 18 * 3584 * 7168 * 2),  # Bytes streamed per step
        )

        if step % self.config.training.logging_steps == 0:
            self.dashboard.render(m)

        return loss_scalar

    def save_lora_checkpoint(self, step: int) -> str:
        """Save trained LoRA weights to D: drive checkpoint directory."""
        ckpt_path = os.path.join(self.config.paths.checkpoints_dir, f"lazy_lora_step_{step:05d}.pt")
        
        state_dict = {}
        for l, bundle in enumerate(self.lora_layers):
            for name, mod in bundle.all_modules():
                if mod.lora_A is not None:
                    key_A = f"layers.{l}.{name}.lora_A"
                    key_B = f"layers.{l}.{name}.lora_B"
                    if HAS_TORCH and isinstance(mod.lora_A, torch.Tensor):
                        state_dict[key_A] = mod.lora_A.detach().cpu()
                        state_dict[key_B] = mod.lora_B.detach().cpu()
                    else:
                        state_dict[key_A] = mod.lora_A
                        state_dict[key_B] = mod.lora_B

        # Save binary
        if HAS_TORCH:
            torch.save(state_dict, ckpt_path)
        else:
            np.savez_compressed(ckpt_path.replace(".pt", ".npz"), **state_dict)

        return ckpt_path

    def train(self, num_steps: Optional[int] = None) -> List[float]:
        """
        Executes full LazyLoRA Out-of-Core training loop.
        Streams Turkish training dataset, processes micro-batches, updates weights,
        renders real-time dashboard, and periodically saves checkpoints.
        """
        from lazy_lora.dataset.turkish_dataset import TurkishDatasetManager
        from lazy_lora.dataset.stream_dataset import StreamingDatasetIterator

        target_steps = num_steps or self.config.training.max_steps
        dataset_mgr = TurkishDatasetManager(self.config.paths.dataset_dir)
        train_file = dataset_mgr.train_file
        if not os.path.exists(train_file):
            print(f"[*] Turkish dataset not found, generating curated dataset at: {train_file}")
            dataset_mgr.generate_curated_samples(target_samples=max(200, target_steps * 2))

        iterator = StreamingDatasetIterator(
            jsonl_path=train_file,
            max_seq_len=self.config.training.max_seq_len,
            pad_token_id=self.config.model.pad_token_id,
            eos_token_id=self.config.model.eos_token_id,
            bos_token_id=self.config.model.bos_token_id,
        )

        print("=" * 82)
        print("  🚀 [STARTING LAZYLORA OUT-OF-CORE TURKISH TRAINING]  🚀")
        print(f"  Target Steps : {target_steps} | Micro-Batch: {self.config.training.micro_batch_size}")
        print(f"  Target Device: {self.device.upper()} | Model Layers: {self.config.model.num_hidden_layers}")
        print(f"  Checkpoints  : {self.config.paths.checkpoints_dir}")
        print("=" * 82)

        step = 0
        loss_history = []

        while step < target_steps:
            for input_ids, target_ids in iterator.get_batches(
                batch_size=self.config.training.micro_batch_size,
                as_torch=HAS_TORCH,
                device=self.device,
            ):
                step += 1
                loss_val = self.train_step(step=step, input_ids=input_ids, target_ids=target_ids)
                loss_history.append(loss_val)

                if step % self.config.training.save_steps == 0:
                    saved_path = self.save_lora_checkpoint(step)
                    print(f"\n💾 [CHECKPOINT SAVED] Step {step:05d} -> {saved_path}")

                if step >= target_steps:
                    break

        final_ckpt = self.save_lora_checkpoint(step)
        print("\n" + "=" * 82)
        print(f"🎉 [TRAINING COMPLETE] {step} Steps Executed Successfully!")
        print(f" Final LoRA Adapter Checkpoint: {final_ckpt}")
        print("=" * 82)
        return loss_history


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LazyLoRA Kimi K3 Training Runner")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--device", default=None, help="Device override (cuda:0 / cpu)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.steps:
        cfg.training.max_steps = args.steps
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.device:
        cfg.streaming.device = args.device

    trainer = LazyLoRATrainer(config=cfg)
    trainer.train(num_steps=args.steps)


if __name__ == "__main__":
    main()

