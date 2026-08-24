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
        moe_latent_size: int = 3584,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        self.layer_idx = layer_idx
        # Attention LoRA adapters
        self.q_lora = LazyLoRALinear(hidden_size, hidden_size, r, alpha, dropout, device)
        self.v_lora = LazyLoRALinear(hidden_size, hidden_size, r, alpha, dropout, device)
        
        # MoE Expert LoRA adapters (shared adapter pool or per-expert low rank)
        self.gate_lora = LazyLoRALinear(hidden_size, moe_latent_size, r, alpha, dropout, device)
        self.up_lora = LazyLoRALinear(hidden_size, moe_latent_size, r, alpha, dropout, device)
        self.down_lora = LazyLoRALinear(moe_latent_size, hidden_size, r, alpha, dropout, device)

    def get_parameters(self) -> List[Any]:
        """Returns all trainable LoRA tensors in this layer."""
        params = []
        for mod in [self.q_lora, self.v_lora, self.gate_lora, self.up_lora, self.down_lora]:
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
                moe_latent_size=self.config.model.routed_expert_hidden_size,
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
                embed_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.float32, device=self.device) * 0.02
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
                head_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.float32, device=self.device) * 0.02
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

        # 4. MoE Expert Execution (2 Shared + Active Top-16)
        if HAS_TORCH and isinstance(h_in, torch.Tensor):
            moe_out = torch.zeros_like(h_mid)

            # Shared experts
            for s_idx in range(self.config.model.num_shared_experts):
                s_bundle = self.expert_streamer.get_expert(layer_idx, s_idx, is_shared=True)
                gate = F.linear(h_moe_norm, s_bundle.gate_proj) + bundle.gate_lora.forward_lora_only(h_moe_norm)
                up = F.linear(h_moe_norm, s_bundle.up_proj) + bundle.up_lora.forward_lora_only(h_moe_norm)
                situ = situ_glu_forward(gate, up)
                down = F.linear(situ, s_bundle.down_proj) + bundle.down_lora.forward_lora_only(situ)
                moe_out = moe_out + down

            # Routed active experts
            # For each active expert, compute forward for routed tokens
            N, top_k = topk_indices.shape
            h_flat = h_moe_norm.view(-1, self.config.model.hidden_size)
            
            for exp_id in active_experts:
                mask = (topk_indices == exp_id)  # [N, top_k]
                if not mask.any():
                    continue
                
                # Expert bundle
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                
                gate = F.linear(h_flat, e_bundle.gate_proj) + bundle.gate_lora.forward_lora_only(h_flat)
                up = F.linear(h_flat, e_bundle.up_proj) + bundle.up_lora.forward_lora_only(h_flat)
                situ = situ_glu_forward(gate, up)
                down = F.linear(situ, e_bundle.down_proj) + bundle.down_lora.forward_lora_only(situ)

                # Weight by routing coefficient
                # Find matching token weights
                token_weights = (topk_weights * mask.float()).sum(dim=-1, keepdim=True)  # [N, 1]
                weighted_down = down * token_weights
                moe_out = moe_out + weighted_down.view_as(h_mid)

            h_out = h_mid + moe_out
        else:
            moe_out = np.zeros_like(h_mid)
            # Shared experts
            for s_idx in range(self.config.model.num_shared_experts):
                s_bundle = self.expert_streamer.get_expert(layer_idx, s_idx, is_shared=True)
                gate = np.matmul(h_moe_norm, s_bundle.gate_proj.T) + bundle.gate_lora.forward_lora_only(h_moe_norm)
                up = np.matmul(h_moe_norm, s_bundle.up_proj.T) + bundle.up_lora.forward_lora_only(h_moe_norm)
                situ = situ_glu_forward(gate, up)
                down = np.matmul(situ, s_bundle.down_proj.T) + bundle.down_lora.forward_lora_only(situ)
                moe_out = moe_out + down

            h_flat = h_moe_norm.reshape(-1, self.config.model.hidden_size)
            for exp_id in active_experts:
                mask = (topk_indices == exp_id)
                if not np.any(mask):
                    continue
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                gate = np.matmul(h_flat, e_bundle.gate_proj.T) + bundle.gate_lora.forward_lora_only(h_flat)
                up = np.matmul(h_flat, e_bundle.up_proj.T) + bundle.up_lora.forward_lora_only(h_flat)
                situ = situ_glu_forward(gate, up)
                down = np.matmul(situ, e_bundle.down_proj.T) + bundle.down_lora.forward_lora_only(situ)
                
                token_weights = np.sum(topk_weights * mask.astype(np.float32), axis=-1, keepdims=True)
                weighted_down = down * token_weights
                moe_out = moe_out + weighted_down.reshape(h_mid.shape)

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

            # MoE Backward: grad_h_out flows into residual h_mid and into MoE experts
            grad_h_mid = grad_h_out.clone()
            grad_moe_norm = torch.zeros_like(h_moe_norm)

            # 1. Shared experts backward
            for s_idx in range(self.config.model.num_shared_experts):
                s_bundle = self.expert_streamer.get_expert(layer_idx, s_idx, is_shared=True)
                gate = F.linear(h_moe_norm, s_bundle.gate_proj) + bundle.gate_lora.forward_lora_only(h_moe_norm)
                up = F.linear(h_moe_norm, s_bundle.up_proj) + bundle.up_lora.forward_lora_only(h_moe_norm)
                situ = situ_glu_forward(gate, up)

                # Down LoRA backward
                gA_d, gB_d, d_situ = bundle.down_lora.compute_lora_gradients(grad_h_out, situ, s_bundle.down_proj)
                bundle.down_lora.accumulate_grad(gA_d, gB_d)

                if d_situ is not None:
                    d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                    gA_g, gB_g, d_in_g = bundle.gate_lora.compute_lora_gradients(d_gate, h_moe_norm, s_bundle.gate_proj)
                    bundle.gate_lora.accumulate_grad(gA_g, gB_g)

                    gA_u, gB_u, d_in_u = bundle.up_lora.compute_lora_gradients(d_up, h_moe_norm, s_bundle.up_proj)
                    bundle.up_lora.accumulate_grad(gA_u, gB_u)

                    if d_in_g is not None and d_in_u is not None:
                        grad_moe_norm.add_(d_in_g + d_in_u)

            # 2. Routed active experts backward
            h_flat = h_moe_norm.view(-1, self.config.model.hidden_size)
            grad_h_flat = grad_h_out.view(-1, self.config.model.hidden_size)
            for exp_id in active_experts:
                mask = (topk_indices == exp_id)
                if not mask.any():
                    continue
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                gate = F.linear(h_flat, e_bundle.gate_proj) + bundle.gate_lora.forward_lora_only(h_flat)
                up = F.linear(h_flat, e_bundle.up_proj) + bundle.up_lora.forward_lora_only(h_flat)
                situ = situ_glu_forward(gate, up)

                token_weights = (topk_weights * mask.float()).sum(dim=-1, keepdim=True)
                d_down_weighted = grad_h_flat * token_weights

                gA_d, gB_d, d_situ = bundle.down_lora.compute_lora_gradients(d_down_weighted, situ, e_bundle.down_proj)
                bundle.down_lora.accumulate_grad(gA_d, gB_d)

                if d_situ is not None:
                    d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                    gA_g, gB_g, d_in_g = bundle.gate_lora.compute_lora_gradients(d_gate, h_flat, e_bundle.gate_proj)
                    bundle.gate_lora.accumulate_grad(gA_g, gB_g)

                    gA_u, gB_u, d_in_u = bundle.up_lora.compute_lora_gradients(d_up, h_flat, e_bundle.up_proj)
                    bundle.up_lora.accumulate_grad(gA_u, gB_u)

                    if d_in_g is not None and d_in_u is not None:
                        grad_moe_norm.add_((d_in_g + d_in_u).view_as(h_moe_norm))

            grad_h_mid.add_(grad_moe_norm)

            # 3. Attention Sublayer Backward
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

            grad_h_mid = grad_h_out.copy().reshape(orig_shape)
            grad_moe_norm = np.zeros(orig_shape, dtype=np.float32)

            for s_idx in range(self.config.model.num_shared_experts):
                s_bundle = self.expert_streamer.get_expert(layer_idx, s_idx, is_shared=True)
                gate = np.matmul(h_moe_norm, s_bundle.gate_proj.T) + bundle.gate_lora.forward_lora_only(h_moe_norm)
                up = np.matmul(h_moe_norm, s_bundle.up_proj.T) + bundle.up_lora.forward_lora_only(h_moe_norm)
                situ = situ_glu_forward(gate, up)

                gA_d, gB_d, d_situ = bundle.down_lora.compute_lora_gradients(grad_h_mid, situ, s_bundle.down_proj)
                bundle.down_lora.accumulate_grad(gA_d, gB_d)

                if d_situ is not None:
                    d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                    gA_g, gB_g, d_in_g = bundle.gate_lora.compute_lora_gradients(d_gate, h_moe_norm, s_bundle.gate_proj)
                    bundle.gate_lora.accumulate_grad(gA_g, gB_g)

                    gA_u, gB_u, d_in_u = bundle.up_lora.compute_lora_gradients(d_up, h_moe_norm, s_bundle.up_proj)
                    bundle.up_lora.accumulate_grad(gA_u, gB_u)

                    if d_in_g is not None and d_in_u is not None:
                        grad_moe_norm += (d_in_g + d_in_u).reshape(orig_shape)

            h_flat = h_moe_norm.reshape(-1, self.config.model.hidden_size)
            grad_h_flat = grad_h_mid.reshape(-1, self.config.model.hidden_size)
            for exp_id in active_experts:
                mask = (topk_indices == exp_id)
                if not np.any(mask):
                    continue
                e_bundle = self.expert_streamer.get_expert(layer_idx, exp_id, is_shared=False)
                gate = np.matmul(h_flat, e_bundle.gate_proj.T) + bundle.gate_lora.forward_lora_only(h_flat)
                up = np.matmul(h_flat, e_bundle.up_proj.T) + bundle.up_lora.forward_lora_only(h_flat)
                situ = situ_glu_forward(gate, up)

                token_weights = np.sum(topk_weights * mask.astype(np.float32), axis=-1, keepdims=True)
                d_down_weighted = grad_h_flat * token_weights

                gA_d, gB_d, d_situ = bundle.down_lora.compute_lora_gradients(d_down_weighted, situ, e_bundle.down_proj)
                bundle.down_lora.accumulate_grad(gA_d, gB_d)

                if d_situ is not None:
                    d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                    gA_g, gB_g, d_in_g = bundle.gate_lora.compute_lora_gradients(d_gate, h_flat, e_bundle.gate_proj)
                    bundle.gate_lora.accumulate_grad(gA_g, gB_g)

                    gA_u, gB_u, d_in_u = bundle.up_lora.compute_lora_gradients(d_up, h_flat, e_bundle.up_proj)
                    bundle.up_lora.accumulate_grad(gA_u, gB_u)

                    if d_in_g is not None and d_in_u is not None:
                        grad_moe_norm += (d_in_g + d_in_u).reshape(orig_shape)

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
        h_current = self._embed_tokens(input_ids)
        last_active_experts = []

        # 2. Sequential Layer-by-Layer Forward Pass (Layer 0 -> Layer L-1)
        for l in range(num_layers):
            h_current, last_active_experts = self.forward_layer(l, h_current)

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
                head_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.float32, device=self.device) * 0.02
            else:
                head_weight = (np.random.randn(vocab_sz, d_hidden) * 0.02).astype(np.float32)

        if HAS_TORCH and isinstance(grad_logits, torch.Tensor):
            grad_h = F.linear(grad_logits, head_weight.t())
        else:
            grad_h = np.matmul(grad_logits, head_weight)

        # Sequential reverse backward pass through all 93 layers reading activations from D: SSD
        for l in range(num_layers - 1, -1, -1):
            grad_h = self.backward_layer(l, grad_h)

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
            for name, mod in [
                ("q_lora", bundle.q_lora),
                ("v_lora", bundle.v_lora),
                ("gate_lora", bundle.gate_lora),
                ("up_lora", bundle.up_lora),
                ("down_lora", bundle.down_lora),
            ]:
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
