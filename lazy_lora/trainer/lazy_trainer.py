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

from lazy_lora.core.config import LazyLoraConfig, get_default_config, synthetic_allowed
from lazy_lora.core.lora_layer import LazyLoRALinear
from lazy_lora.core.moe_router import KimiK3MoERouter
from lazy_lora.core.situ_activation import situ_glu_forward, situ_glu_backward
from lazy_lora.core.attention import kda_attention, mla_attention, apply_attn_res
from lazy_lora.streaming.mmap_loader import MmapTensorStreamer
from lazy_lora.streaming.trunk_streamer import LayerTrunkStreamer, RMSNormFunction
from lazy_lora.streaming.expert_streamer import DynamicExpertStreamer
from lazy_lora.streaming.activation_ring_buffer import ActivationRingBuffer
from lazy_lora.trainer.loss import compute_cross_entropy_loss
from lazy_lora.trainer.optimizer import LazyLoRAOptimizer
from lazy_lora.monitor.metrics import MetricsTracker
from lazy_lora.monitor.dashboard import TerminalDashboard
from lazy_lora.native import kernel as native_kernel

# Rows per expert above which the widened-matrix sgemm beats the fused kernel (measured
# on the i7-7700HQ: fused ~27 + 1.7*M ms, C decode + sgemm ~60 + 0.5*M ms per expert).
FUSED_MAX_ROWS = 48


if HAS_TORCH:
    class RoutedExpertsFunction(torch.autograd.Function):
        """
        The routed-expert map of one MoE layer as a single autograd node.

        The rest of a decoder layer (bank mixing, norms, attention, shared expert, latent
        projections) is cheap enough to replay under autograd in the backward pass, so it
        is. The routed experts are not: keeping ~600 streamed expert matrices alive in an
        autograd graph is tens of gigabytes. This node streams them once in forward and
        once more in backward, computing the LoRA gradients and dL/dh_latent by hand,
        while autograd handles everything around it. Forward and backward therefore share
        one definition of the layer, which is what the old hand-written backward lacked.
        """

        @staticmethod
        def forward(ctx, h_latent, topk_weights, topk_indices, trainer, layer_idx, bundle, active_experts):
            out = trainer._routed_experts_forward(layer_idx, bundle, h_latent, topk_indices, topk_weights, active_experts)
            ctx.save_for_backward(h_latent, topk_weights, topk_indices)
            ctx.trainer, ctx.layer_idx, ctx.bundle, ctx.active_experts = trainer, layer_idx, bundle, active_experts
            return out

        @staticmethod
        def backward(ctx, grad_out):
            h_latent, topk_weights, topk_indices = ctx.saved_tensors
            grad_h, grad_w = ctx.trainer._routed_experts_backward(
                ctx.layer_idx, ctx.bundle, h_latent, topk_indices, topk_weights, ctx.active_experts, grad_out
            )
            return grad_h, grad_w, None, None, None, None, None


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
        is_dense: bool = False,
        dense_intermediate_size: int = 33792,
        is_kda: bool = True,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        q_head_total: int = 18432,
        kv_head_total: int = 24576,
    ):
        self.layer_idx = layer_idx
        self.is_dense = is_dense
        self.is_kda = is_kda

        # Attention LoRA adapters sit on the projections that actually produce q and v,
        # which differ between the two attention types Kimi Linear interleaves.
        if is_kda:
            # KDA: q_proj / v_proj map hidden -> num_heads * head_dim
            self.q_lora = LazyLoRALinear(hidden_size, attn_out_size, r, alpha, dropout, device)
            self.v_lora = LazyLoRALinear(hidden_size, attn_out_size, r, alpha, dropout, device)
        else:
            # MLA: the latent up-projections q_b_proj and kv_b_proj
            self.q_lora = LazyLoRALinear(q_lora_rank, q_head_total, r, alpha, dropout, device)
            self.v_lora = LazyLoRALinear(kv_lora_rank, kv_head_total, r, alpha, dropout, device)

        if is_dense:
            # The first `first_k_dense_replace` layers of Kimi K3 carry a plain MLP
            # (mlp.gate_proj / up_proj / down_proj, 7168 -> 33792 -> 7168) and no experts.
            self.dense_gate_lora = LazyLoRALinear(hidden_size, dense_intermediate_size, r, alpha, dropout, device)
            self.dense_up_lora = LazyLoRALinear(hidden_size, dense_intermediate_size, r, alpha, dropout, device)
            self.dense_down_lora = LazyLoRALinear(dense_intermediate_size, hidden_size, r, alpha, dropout, device)
            return

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
        if self.is_dense:
            return [
                ("q_lora", self.q_lora),
                ("v_lora", self.v_lora),
                ("dense_gate_lora", self.dense_gate_lora),
                ("dense_up_lora", self.dense_up_lora),
                ("dense_down_lora", self.dense_down_lora),
            ]
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
        self.compute_dtype = (torch.float32 if (HAS_TORCH and self.mmap_streamer.upcast_float32)
                              else (torch.bfloat16 if HAS_TORCH else None))
        # Optional expert access trace (lazy_lora.monitor.trace.ExpertTraceWriter); set by
        # measurement scripts. Records every routing decision of the forward pass.
        self.trace = None
        # LAZYLORA_GPU=1: routed-expert decode and matmuls on the CUDA device (fp32). The
        # packed 17.5 MB expert crosses PCIe, is decoded with the LUT on the GPU and
        # multiplied there; everything else (attention, shared expert, LoRA) stays on the
        # CPU. On the GTX 1050 this takes an expert from ~55-120 ms to ~25 ms.
        self.gpu = bool(HAS_TORCH and os.environ.get("LAZYLORA_GPU", "") == "1" and torch.cuda.is_available())
        if self.gpu:
            print(f"[gpu] routed experts on {torch.cuda.get_device_name(0)}", flush=True)
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
        # One sub-directory per process: two engines sharing act_layer_NNN.bin (a mock
        # test next to a real run, or two runs) silently read each other's activations in
        # the backward pass. The directory is removed in close().
        self._act_dir = os.path.join(self.config.paths.activation_cache_dir, f"run_{os.getpid()}")
        self._remove_stale_activation_dirs()
        self.act_buffer = ActivationRingBuffer(
            cache_dir=self._act_dir,
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
                is_dense=(l < self.config.model.first_k_dense_replace),
                dense_intermediate_size=self.config.model.intermediate_size,
                is_kda=self.config.model.is_kda_layer(l),
                q_lora_rank=self.config.model.q_lora_rank,
                kv_lora_rank=self.config.model.kv_lora_rank,
                q_head_total=self.config.model.num_attention_heads
                * (self.config.model.qk_nope_head_dim + self.config.model.qk_rope_head_dim),
                kv_head_total=self.config.model.num_attention_heads
                * (self.config.model.qk_nope_head_dim + self.config.model.v_head_dim),
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

    def _remove_stale_activation_dirs(self) -> None:
        """Delete run_<pid> directories whose process is gone. Never touch a live one: a
        manual cleanup once deleted the directory of a running measurement at layer 25."""
        import shutil
        base = self.config.paths.activation_cache_dir
        try:
            names = os.listdir(base)
        except OSError:
            return
        for name in names:
            if not name.startswith("run_"):
                continue
            try:
                pid = int(name[4:])
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
                continue                      # alive: leave it alone
            except ProcessLookupError:
                shutil.rmtree(os.path.join(base, name), ignore_errors=True)
            except PermissionError:
                continue                      # alive, other user

    def close(self) -> None:
        """Drop this run's activation scratch directory and open shard descriptors."""
        import shutil
        try:
            self.act_buffer.clean_cache()
            shutil.rmtree(self._act_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            self.mmap_streamer.close()
        except Exception:
            pass

    def _embed_tokens(self, input_ids: Union["torch.Tensor", np.ndarray]) -> Union["torch.Tensor", np.ndarray]:
        """Embed input tokens by reading only the rows the batch actually touches.

        The full table is 163840 x 7168 (2.35 GB); a 512-token batch needs a few hundred
        rows, so it is gathered straight from mmap instead of being copied into RAM.
        """
        vocab_sz = self.config.model.vocab_size
        d_hidden = self.config.model.hidden_size

        ids_np = input_ids.detach().cpu().numpy() if (HAS_TORCH and isinstance(input_ids, torch.Tensor)) else np.asarray(input_ids)
        unique_ids, inverse = np.unique(ids_np.reshape(-1), return_inverse=True)
        rows = self.mmap_streamer.load_tensor_rows(
            "model.embed_tokens.weight", unique_ids, target_device=self.device
        )
        if rows is not None:
            if HAS_TORCH and isinstance(rows, torch.Tensor):
                gathered = rows[torch.from_numpy(inverse.astype(np.int64))]
                return gathered.view(*ids_np.shape, rows.shape[1])
            return rows[inverse].reshape(*ids_np.shape, rows.shape[1])

        embed_weight = self.mmap_streamer.load_tensor("model.embed_tokens.weight", target_device=self.device)

        if embed_weight is None:
            synthetic_allowed("model.embed_tokens.weight")
            # Synthetic embedding table for testing
            if HAS_TORCH:
                embed_weight = torch.randn(vocab_sz, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
            else:
                embed_weight = (np.random.randn(vocab_sz, d_hidden) * 0.02).astype(np.float32)

        if HAS_TORCH and isinstance(input_ids, torch.Tensor):
            return F.embedding(input_ids, embed_weight)
        else:
            return embed_weight[input_ids]

    LM_HEAD_CHUNK_ROWS = 16384

    def _lm_head_bands(self):
        """Yield (row_start, row_end, weight_band) over the LM head, one band at a time.

        Keeps at most LM_HEAD_CHUNK_ROWS x hidden of the 2.35 GB head resident, instead of
        materialising the whole matrix (which the forward and the backward each did once).
        """
        vocab_sz = self.config.model.vocab_size
        d_hidden = self.config.model.hidden_size
        shape = self.mmap_streamer.tensor_shape("lm_head.weight")

        if shape is None:
            synthetic_allowed("lm_head.weight")
            # Synthetic head for tests / profiling: emit it as a single band.
            if HAS_TORCH:
                w = torch.randn(vocab_sz, d_hidden, dtype=torch.bfloat16, device=self.device) * 0.02
            else:
                w = (np.random.randn(vocab_sz, d_hidden) * 0.02).astype(np.float32)
            yield 0, vocab_sz, w
            return

        for start in range(0, shape[0], self.LM_HEAD_CHUNK_ROWS):
            end = min(start + self.LM_HEAD_CHUNK_ROWS, shape[0])
            band = self.mmap_streamer.load_tensor_row_slice(
                "lm_head.weight", start, end, target_device=self.device
            )
            if band is None:
                continue
            yield start, end, band
            del band

    def _finalize_hidden(self, hidden_state, bank):
        """
        Close out the residual bank and apply the model's final RMSNorm.

        After the last layer the model mixes the bank into the stream one more time with
        its own output gate, then applies model.norm before the LM head. Differentiable:
        the backward replays it to get the gradient into the stream and the bank.
        """
        proj = self.mmap_streamer.load_tensor("model.output_attn_res_proj.weight", target_device=self.device)
        norm = self.mmap_streamer.load_tensor("model.output_attn_res_norm.weight", target_device=self.device)
        if proj is None or norm is None:
            synthetic_allowed("model.output_attn_res_proj / output_attn_res_norm")
        else:
            hidden_state = self._mix_block_residual(hidden_state, bank, proj, norm)

        final_norm = self.mmap_streamer.load_tensor("model.norm.weight", target_device=self.device)
        if final_norm is None:
            synthetic_allowed("model.norm.weight")
        else:
            hidden_state = RMSNormFunction.forward(
                hidden_state, final_norm, eps=self.config.model.rms_norm_eps
            )
        return hidden_state

    def _project_lm_head(self, hidden_state: Union["torch.Tensor", np.ndarray]) -> Union["torch.Tensor", np.ndarray]:
        """Project final hidden state through the LM head, band by band."""
        vocab_sz = self.config.model.vocab_size

        if HAS_TORCH and isinstance(hidden_state, torch.Tensor):
            logits = None
            for start, end, w in self._lm_head_bands():
                if w.dtype != hidden_state.dtype:
                    w = w.to(hidden_state.dtype)
                part = F.linear(hidden_state, w)
                if logits is None:
                    logits = torch.zeros(
                        (*hidden_state.shape[:-1], vocab_sz), dtype=part.dtype, device=part.device
                    )
                logits[..., start:end] = part
                del part
            return logits
        else:
            logits = np.zeros((*hidden_state.shape[:-1], vocab_sz), dtype=np.float32)
            for start, end, w in self._lm_head_bands():
                logits[..., start:end] = np.matmul(hidden_state, np.asarray(w, dtype=np.float32).T)
            return logits

    def _lm_head_backward(
        self,
        grad_logits: Union["torch.Tensor", np.ndarray],
    ) -> Union["torch.Tensor", np.ndarray]:
        """Propagate the loss gradient back through the LM head, band by band."""
        if HAS_TORCH and isinstance(grad_logits, torch.Tensor):
            grad_h = None
            for start, end, w in self._lm_head_bands():
                g_band = grad_logits[..., start:end].to(w.dtype)
                part = F.linear(g_band, w.t())
                grad_h = part if grad_h is None else grad_h + part
                del part, g_band
            return grad_h
        else:
            grad_h = None
            for start, end, w in self._lm_head_bands():
                part = np.matmul(grad_logits[..., start:end], np.asarray(w, dtype=np.float32))
                grad_h = part if grad_h is None else grad_h + part
            return grad_h

    # ------------------------------------------------------------------ block-residual bank
    #
    # Kimi Linear keeps a bank of residual snapshots: at every block boundary layer
    # (layer_idx % attn_res_block_size == 0) the incoming stream h_in is pushed onto the
    # bank and the stream restarts from the attention output. Each layer mixes the bank
    # into the stream twice (before attention, before the MoE) with a learned softmax.
    #
    # The bank entries are exactly the h_in of the boundary layers, which the activation
    # ring buffer already stores, so the backward pass rebuilds the bank a layer sees from
    # those files instead of storing it again. Gradient that flows into a bank entry is
    # parked in self._grad_bank and added to that boundary layer's grad_h_in when the
    # backward sweep reaches it.

    def _reset_block_residual(self) -> None:
        """Start a fresh residual bank for a new forward pass."""
        self._block_residual = None
        self._grad_bank: Dict[int, Any] = {}

    def _is_boundary(self, layer_idx: int) -> bool:
        bs = self.config.model.attn_res_block_size
        return bool(bs) and layer_idx % bs == 0

    def _bank_entries_before(self, layer_idx: int) -> List[int]:
        """Boundary layers whose h_in is in the bank when `layer_idx` begins (in push order)."""
        bs = self.config.model.attn_res_block_size
        if not bs or layer_idx <= 0:
            return []
        return [bs * j for j in range((layer_idx - 1) // bs + 1)]

    def _bank_at_entry(self, layer_idx: int, like):
        """Rebuild the bank tensor [N, num_entries, hidden] a layer saw, from saved activations."""
        entries = self._bank_entries_before(layer_idx)
        if not entries:
            return None
        parts = []
        for l in entries:
            act = self.act_buffer.load_activation(l, target_device=self.device, as_torch=True)
            if act is None:
                raise RuntimeError(f"activation of boundary layer {l} is missing; cannot rebuild the bank for layer {layer_idx}")
            parts.append(act.reshape(-1, act.shape[-1]))
        return torch.stack(parts, dim=1).to(like.dtype)

    def _mix_block_residual(self, prefix_sum, bank, proj_weight, norm_weight):
        """Mix the residual bank into the live stream (identity when the bank is empty)."""
        if bank is None or bank.shape[1] == 0:
            return prefix_sum
        shape = prefix_sum.shape
        flat = prefix_sum.reshape(-1, shape[-1])
        mixed = apply_attn_res(
            flat, bank.to(flat.dtype), proj_weight, norm_weight,
            eps=self.config.model.rms_norm_eps,
        )
        return mixed.view(shape)

    @staticmethod
    def _add_to_prefix(prefix_sum, sublayer_out):
        """prefix_sum + sublayer output, where a restarted stream starts from the output."""
        return sublayer_out if prefix_sum is None else prefix_sum + sublayer_out

    def _route(self, layer_idx: int, h_moe_norm):
        """
        Run this layer's MoE gate.

        Every layer carries its own gate matrix and score-correction bias on disk; without
        them the router falls back to its random init, which selects experts at random.
        """
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.gate."
        gate_w = self.mmap_streamer.load_tensor(f"{prefix}weight", target_device=self.device)
        gate_b = self.mmap_streamer.load_tensor(f"{prefix}e_score_correction_bias", target_device=self.device)
        if gate_w is None or gate_b is None:
            synthetic_allowed(f"layer {layer_idx} router gate weight / e_score_correction_bias")
        topk_indices, topk_weights = self.router.forward(h_moe_norm, weight=gate_w, bias=gate_b)
        del gate_w, gate_b
        return topk_indices, topk_weights

    def _attention_forward(self, layer_idx: int, bundle, h_norm):
        """
        Run the layer's real attention sublayer.

        The attention weights are streamed here and dropped as soon as the sublayer is
        done, so only one layer's worth is resident at a time.
        """
        m = self.config.model
        is_kda = m.is_kda_layer(layer_idx)
        if not HAS_TORCH:
            raise RuntimeError(
                "The Kimi Linear attention sublayers (KDA / MLA) need PyTorch. "
                "Run the engine with the workspace venv interpreter, which has torch installed."
            )
        as_numpy = not isinstance(h_norm, torch.Tensor)
        if as_numpy:
            h_norm = torch.from_numpy(np.asarray(h_norm, dtype=np.float32))
        w = self.trunk_streamer.load_attention_weights(
            layer_idx,
            is_kda=is_kda,
            num_heads=m.num_attention_heads,
            head_dim=m.head_dim,
            conv_kernel=m.short_conv_kernel_size,
            q_lora_rank=m.q_lora_rank,
            kv_lora_rank=m.kv_lora_rank,
            qk_nope_head_dim=m.qk_nope_head_dim,
            qk_rope_head_dim=m.qk_rope_head_dim,
            v_head_dim=m.v_head_dim,
        )

        if is_kda:
            out = kda_attention(
                h_norm, w,
                num_heads=m.num_attention_heads,
                head_dim=m.head_dim,
                gate_lower_bound=m.gate_lower_bound,
                eps=m.rms_norm_eps,
                q_lora=bundle.q_lora,
                v_lora=bundle.v_lora,
            )
        else:
            out = mla_attention(
                h_norm, w,
                num_heads=m.num_attention_heads,
                qk_nope_head_dim=m.qk_nope_head_dim,
                qk_rope_head_dim=m.qk_rope_head_dim,
                v_head_dim=m.v_head_dim,
                kv_lora_rank=m.kv_lora_rank,
                eps=m.rms_norm_eps,
                q_lora=bundle.q_lora,
                v_lora=bundle.v_lora,
            )
        del w
        if as_numpy:
            return out.detach().to(torch.float32).numpy()
        return out

    def _load_dense_weight(self, layer_idx: int, name: str, shape):
        """One tensor of the dense MLP (gate_proj / up_proj / down_proj), with mock fallback."""
        w = self.mmap_streamer.load_tensor(f"model.layers.{layer_idx}.mlp.{name}.weight", target_device=self.device)
        if w is None:
            synthetic_allowed(f"layer {layer_idx} dense mlp {name}")
            w = torch.randn(*shape, dtype=self.compute_dtype, device=self.device) * 0.01
        return w

    def _dense_mlp_forward(self, layer_idx, bundle, h_mid, h_norm):
        """
        Dense MLP sublayer: h -> (gate, up) -> SiTU-GLU -> down -> residual add.

        The three 33792 x 7168 matrices are loaded one at a time and dropped after use:
        together they are 1.45 GB in bf16 and 2.9 GB in fp32, most of this machine's RAM.
        (Under autograd they are retained by the graph anyway.)
        """
        d_in = self.config.model.hidden_size
        d_mid = self.config.model.intermediate_size
        gate_w = self._load_dense_weight(layer_idx, "gate_proj", (d_mid, d_in))
        if h_norm.dtype != gate_w.dtype:
            h_norm = h_norm.to(gate_w.dtype)
        gate = F.linear(h_norm, gate_w) + bundle.dense_gate_lora.forward_lora_only(h_norm)
        del gate_w
        up_w = self._load_dense_weight(layer_idx, "up_proj", (d_mid, d_in))
        up = F.linear(h_norm, up_w) + bundle.dense_up_lora.forward_lora_only(h_norm)
        del up_w
        situ = situ_glu_forward(gate, up)
        del gate, up
        down_w = self._load_dense_weight(layer_idx, "down_proj", (d_in, d_mid))
        mlp_out = F.linear(situ, down_w) + bundle.dense_down_lora.forward_lora_only(situ)
        del down_w, situ
        return mlp_out

    def forward_layer(
        self,
        layer_idx: int,
        h_in: "torch.Tensor",
    ) -> Tuple["torch.Tensor", List[int]]:
        """
        Out-of-core forward for one layer: save h_in to the activation ring buffer, run the
        layer against the current bank, then push h_in onto the bank if this is a block
        boundary. The layer itself is `_run_layer`, shared with the backward pass.
        """
        if not (HAS_TORCH and isinstance(h_in, torch.Tensor)):
            raise RuntimeError("the LazyLoRA engine runs on torch tensors; the NumPy path was removed")
        self.act_buffer.save_activation(layer_idx, h_in)
        t0 = time.time()
        bytes0 = self.mmap_streamer.bytes_read
        h_out, active_experts = self._run_layer(layer_idx, h_in, self._block_residual)
        if self.trace is not None and getattr(self, "_last_routing", None) is not None:
            idx, w = self._last_routing
            self._last_routing = None
            self.trace.record_layer(layer_idx, idx, w, extra={
                "seconds": round(time.time() - t0, 2),
                "bytes_read": int(self.mmap_streamer.bytes_read - bytes0),
                "is_kda": bool(self.config.model.is_kda_layer(layer_idx)),
            })
        if self._is_boundary(layer_idx):
            entry = h_in.reshape(-1, h_in.shape[-1]).unsqueeze(1)
            self._block_residual = entry if self._block_residual is None else torch.cat(
                [self._block_residual.to(entry.dtype), entry], dim=1)
        self.expert_streamer.evict_layer_experts(layer_idx)
        return h_out, active_experts

    def _run_layer(self, layer_idx: int, h_in, bank):
        """
        One decoder layer as a torch graph, following KimiDecoderLayer exactly:

        1. hidden = mix(h_in, bank)                      (pre-attention bank mix)
        2. boundary layer: bank_local = bank + [h_in], prefix restarts (None)
        3. attention on norm(hidden); prefix = prefix + attn_out (or attn_out)
        4. h_mid = mix(prefix, bank_local)               (pre-MoE bank mix)
        5. MoE (or the dense MLP on layer 0) on norm(h_mid); h_out = prefix + out

        Called under no_grad by the forward pass and under enable_grad by the backward
        pass, so the two cannot disagree about what the layer computes.
        """
        trunk = self.trunk_streamer.load_layer_trunk(layer_idx)
        bundle = self.lora_layers[layer_idx]

        hidden = self._mix_block_residual(h_in, bank, trunk.self_attention_res_proj, trunk.self_attention_res_norm)
        if self._is_boundary(layer_idx):
            entry = h_in.reshape(-1, h_in.shape[-1]).unsqueeze(1)
            bank_local = entry if bank is None else torch.cat([bank.to(entry.dtype), entry], dim=1)
            prefix_sum = None
        else:
            bank_local = bank
            prefix_sum = h_in

        h_norm = RMSNormFunction.forward(hidden, trunk.input_layernorm)
        attn_out = self._attention_forward(layer_idx, bundle, h_norm)
        prefix_sum = self._add_to_prefix(prefix_sum, attn_out)
        del attn_out

        h_mid = self._mix_block_residual(prefix_sum, bank_local, trunk.mlp_res_proj, trunk.mlp_res_norm)
        self.trunk_streamer.release_layer_trunk()
        h_moe_norm = RMSNormFunction.forward(h_mid, trunk.post_attention_layernorm)

        if bundle.is_dense:
            mlp_out = self._dense_mlp_forward(layer_idx, bundle, h_mid, h_moe_norm)
            return self._add_to_prefix(prefix_sum, mlp_out), []

        topk_indices, topk_weights = self._route(layer_idx, h_moe_norm)
        if self.trace is not None and not torch.is_grad_enabled():
            self._last_routing = (topk_indices, topk_weights)
        active_experts = self.expert_streamer.sort_by_disk_order(
            layer_idx, self.router.get_active_expert_set(topk_indices))
        moe_out = self._moe_forward(layer_idx, bundle, h_moe_norm, topk_indices, topk_weights, active_experts)
        return self._add_to_prefix(prefix_sum, moe_out.view_as(h_mid)), active_experts

    def _moe_forward(self, layer_idx, bundle, h_moe_norm, topk_indices, topk_weights, active_experts):
        """
        Kimi K3 latent MoE: shared expert at full width, plus the routed experts in the
        3584-wide latent space (down-project, experts, RMSNorm of the aggregate, up-project).
        """
        s_bundle = self.expert_streamer.get_expert(layer_idx, 0, is_shared=True)
        x = h_moe_norm if h_moe_norm.dtype == s_bundle.gate_proj.dtype else h_moe_norm.to(s_bundle.gate_proj.dtype)
        s_gate = F.linear(x, s_bundle.gate_proj) + bundle.shared_gate_lora.forward_lora_only(x)
        s_up = F.linear(x, s_bundle.up_proj) + bundle.shared_up_lora.forward_lora_only(x)
        s_situ = situ_glu_forward(s_gate, s_up)
        shared_out = F.linear(s_situ, s_bundle.down_proj) + bundle.shared_down_lora.forward_lora_only(s_situ)
        del s_bundle, s_gate, s_up, s_situ

        prefix = f"model.layers.{layer_idx}.block_sparse_moe."
        latent_down_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_down_proj.weight", target_device=self.device)
        latent_up_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_up_proj.weight", target_device=self.device)
        latent_norm_w = self.mmap_streamer.load_tensor(f"{prefix}routed_expert_norm.weight", target_device=self.device)
        d_h = self.config.model.hidden_size
        d_l = self.config.model.routed_expert_hidden_size
        if latent_down_w is None or latent_up_w is None or latent_norm_w is None:
            synthetic_allowed(f"layer {layer_idx} latent MoE projections (routed_expert_down/up_proj, routed_expert_norm)")
            if latent_down_w is None:
                latent_down_w = torch.randn(d_l, d_h, dtype=self.compute_dtype, device=self.device) * 0.01
            if latent_up_w is None:
                latent_up_w = torch.randn(d_h, d_l, dtype=self.compute_dtype, device=self.device) * 0.01
            if latent_norm_w is None:
                latent_norm_w = torch.ones(d_l, dtype=self.compute_dtype, device=self.device)
        latent_down_w = latent_down_w.to(self.compute_dtype)
        latent_up_w = latent_up_w.to(self.compute_dtype)

        h_latent = F.linear(x.reshape(-1, d_h).to(self.compute_dtype), latent_down_w)          # [N, 3584]
        del latent_down_w
        routed_latent = RoutedExpertsFunction.apply(
            h_latent, topk_weights, topk_indices, self, layer_idx, bundle, active_experts)
        routed_latent = RMSNormFunction.forward(routed_latent, latent_norm_w.to(routed_latent.dtype))
        routed_out = F.linear(routed_latent.to(latent_up_w.dtype), latent_up_w)                  # [N, 7168]
        del latent_up_w, latent_norm_w
        return shared_out + routed_out.view_as(shared_out)

    # ------------------------------------------------------------------ one routed expert
    #
    # Everything below runs in float32: on this CPU an fp32 GEMM is 3.4x faster than bf16,
    # and the packed weights decode exactly into fp32. Two implementations:
    #   packed  : lazy_lora.native (fused decode+dot for few rows, C decode + sgemm for many)
    #   widened : the bundle already holds bf16/fp32 matrices (mock experts, fallback)

    def _expert_matrices(self, e, rows: int):
        """(W1, W3, W2) as fp32 tensors when the widened path is used, else None."""
        if not e.packed:
            return (e.gate_proj.float(), e.up_proj.float(), e.down_proj.float())
        if self.gpu:
            from lazy_lora.streaming.expert_streamer import _dequantize_mxfp4
            return tuple(_dequantize_mxfp4(p.to("cuda", non_blocking=True), s.to("cuda", non_blocking=True),
                                           out_dtype=torch.float32)
                         for p, s in ((e.gate_packed, e.gate_scale), (e.up_packed, e.up_scale),
                                      (e.down_packed, e.down_scale)))
        if rows > FUSED_MAX_ROWS:
            k = native_kernel()
            return (k.dequant(e.gate_packed, e.gate_scale), k.dequant(e.up_packed, e.up_scale),
                    k.dequant(e.down_packed, e.down_scale))
        return None

    def _expert_forward(self, bundle, e, x, mats):
        """gate, up, situ, down (all fp32) for the rows x [n, 3584] of one expert."""
        xf = x.float()
        if mats is not None and mats[0].is_cuda:
            W1, W3, W2 = mats
            xg = xf.to("cuda", non_blocking=True)
            gate = F.linear(xg, W1) + bundle.gate_lora.forward_lora_only(x).float().to("cuda")
            up = F.linear(xg, W3) + bundle.up_lora.forward_lora_only(x).float().to("cuda")
            situ = situ_glu_forward(gate, up)
            down = F.linear(situ, W2)
            gate, up, situ, down = gate.cpu(), up.cpu(), situ.cpu(), down.cpu()
            down = down + bundle.down_lora.forward_lora_only(situ).float()
            return gate, up, situ, down
        if mats is not None:
            W1, W3, W2 = mats
            gate = F.linear(xf, W1)
            up = F.linear(xf, W3)
        else:
            k = native_kernel()
            gate = k.gemm(xf, e.gate_packed, e.gate_scale)
            up = k.gemm(xf, e.up_packed, e.up_scale)
        gate = gate + bundle.gate_lora.forward_lora_only(x).float()
        up = up + bundle.up_lora.forward_lora_only(x).float()
        situ = situ_glu_forward(gate, up)
        if mats is not None:
            down = F.linear(situ, mats[2])
        else:
            down = native_kernel().gemm(situ, e.down_packed, e.down_scale)
        down = down + bundle.down_lora.forward_lora_only(situ).float()
        return gate, up, situ, down

    def _expert_dx(self, e, mats, d_out, which: str):
        """d_out @ W for one of the expert's matrices ('gate' | 'up' | 'down'), fp32."""
        d = d_out.float()
        if mats is not None:
            W = {"gate": mats[0], "up": mats[1], "down": mats[2]}[which]
            if W.is_cuda:
                return (d.to("cuda", non_blocking=True) @ W).cpu()
            return d @ W
        k = native_kernel()
        p, s = {"gate": (e.gate_packed, e.gate_scale), "up": (e.up_packed, e.up_scale),
                "down": (e.down_packed, e.down_scale)}[which]
        return k.gemm_t(d, p, s)

    def _routed_experts_forward(self, layer_idx, bundle, h_latent, topk_indices, topk_weights, active_experts):
        """
        sum_e w_e(token) * expert_e(h_latent), streaming the experts in disk order.

        Each expert runs only on the rows that selected it (gather / index_add), not on the
        whole batch behind a mask: at N=127 an expert serves ~2-3 tokens on average, so the
        masked form did ~40x the useful arithmetic (report finding K5). The result is the
        same up to float rounding, since masked rows contributed exact zeros.

        The sum over the 16 experts of a token is accumulated in float32 and returned in
        float32 (the latent RMSNorm that follows runs on it before the cast back), as the
        C reference accumulates in double. Summing in bf16 shifts layer outputs by ~1e-3
        and changes routing decisions a few layers later.
        """
        out = torch.zeros(h_latent.shape, dtype=torch.float32, device=h_latent.device)
        for exp_id, e in self.expert_streamer.stream_experts(layer_idx, active_experts):
            mask = (topk_indices == exp_id)                                  # [N, top_k]
            rows = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)      # tokens using e
            if rows.numel() == 0:
                continue
            x = h_latent.index_select(0, rows)
            mats = self._expert_matrices(e, rows.numel())
            _gate, _up, _situ, e_down = self._expert_forward(bundle, e, x, mats)
            tw = (topk_weights.index_select(0, rows) * mask.index_select(0, rows).to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
            out.index_add_(0, rows, e_down * tw.float())
            del e, x, mats, _gate, _up, _situ, e_down
        return out

    def _routed_experts_backward(self, layer_idx, bundle, h_latent, topk_indices, topk_weights, active_experts, grad_out):
        """
        Backward of `_routed_experts_forward`: streams the experts a second time, accumulates
        the LoRA gradients of gate/up/down, and returns (dL/dh_latent, dL/dtopk_weights).
        Like the forward, each expert only touches the rows that selected it.

        dL/dtopk_weights is the per-token dot product of the upstream gradient with the
        expert output, placed in the slot that selected the expert. It lets the gradient
        reach the (frozen) router's input, as it does in the real model.
        """
        grad_h = torch.zeros(h_latent.shape, dtype=torch.float32, device=h_latent.device)
        grad_w = torch.zeros(topk_weights.shape, dtype=torch.float32, device=h_latent.device)
        with torch.no_grad():
            for exp_id, e in self.expert_streamer.stream_experts(layer_idx, active_experts):
                mask = (topk_indices == exp_id)
                rows = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                x = h_latent.index_select(0, rows)
                g_out = grad_out.index_select(0, rows).float()
                m = mask.index_select(0, rows)
                mats = self._expert_matrices(e, rows.numel())
                gate, up, situ, e_down = self._expert_forward(bundle, e, x, mats)

                dot = (g_out * e_down).sum(dim=-1, keepdim=True)                     # [n_e, 1]
                grad_w.index_add_(0, rows, dot * m.to(dot.dtype))

                tw = (topk_weights.index_select(0, rows) * m.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
                d_down = g_out * tw.float()

                gA, gB, d_situ = bundle.down_lora.compute_lora_gradients(
                    d_down, situ, dx_base=self._expert_dx(e, mats, d_down, "down"))
                bundle.down_lora.accumulate_grad(gA, gB)
                d_gate, d_up = situ_glu_backward(d_situ, gate, up)
                gA, gB, d_in_g = bundle.gate_lora.compute_lora_gradients(
                    d_gate, x, dx_base=self._expert_dx(e, mats, d_gate, "gate"))
                bundle.gate_lora.accumulate_grad(gA, gB)
                gA, gB, d_in_u = bundle.up_lora.compute_lora_gradients(
                    d_up, x, dx_base=self._expert_dx(e, mats, d_up, "up"))
                bundle.up_lora.accumulate_grad(gA, gB)
                grad_h.index_add_(0, rows, d_in_g.float() + d_in_u.float())
                del e, x, g_out, mats, gate, up, situ, e_down, d_down, d_situ, d_gate, d_up, d_in_g, d_in_u
        return grad_h.to(h_latent.dtype), grad_w.to(topk_weights.dtype)

    def _lora_params_of(self, bundle) -> List[Tuple[Any, "torch.Tensor"]]:
        """(module, parameter) pairs for every LoRA tensor in a layer bundle."""
        out = []
        for _, mod in bundle.all_modules():
            for p in (mod.lora_A, mod.lora_B):
                if p is not None:
                    out.append((mod, p))
        return out

    @staticmethod
    def _accumulate_param_grad(p, g) -> None:
        if g is None:
            return
        g = g.detach().to(p.dtype)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.add_(g)

    def backward_layer(
        self,
        layer_idx: int,
        grad_h_out: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Out-of-core backward for one layer.

        Loads h_in and the bank the layer saw from the activation ring buffer, replays
        `_run_layer` under autograd with h_in, the bank and this layer's LoRA tensors as
        leaves, and reads the gradients off it. The routed experts are handled inside
        RoutedExpertsFunction (their LoRA gradients are accumulated there). Gradient into
        bank entries is parked in `_grad_bank` and delivered to the boundary layer that
        owns the entry when the sweep reaches it; at a boundary layer, its own parked
        gradient is added to grad_h_in.
        """
        if not (HAS_TORCH and isinstance(grad_h_out, torch.Tensor)):
            raise RuntimeError("the LazyLoRA engine runs on torch tensors; the NumPy path was removed")
        h_in = self.act_buffer.load_activation(layer_idx, target_device=self.device, as_torch=True)
        if h_in is None:
            raise RuntimeError(f"activation of layer {layer_idx} is missing from the ring buffer")
        bank = self._bank_at_entry(layer_idx, h_in)
        bundle = self.lora_layers[layer_idx]
        mods_params = self._lora_params_of(bundle)
        params = [p for _, p in mods_params]

        with torch.enable_grad():
            x = h_in.detach().clone().requires_grad_(True)
            leaves = [x]
            b = None
            if bank is not None:
                b = bank.detach().clone().requires_grad_(True)
                leaves.append(b)
            out, _ = self._run_layer(layer_idx, x, b)
            grads = torch.autograd.grad(
                outputs=out, inputs=leaves + params,
                grad_outputs=grad_h_out.to(out.dtype),
                allow_unused=True, retain_graph=False,
            )
        del out
        grad_x = grads[0]
        grad_b = grads[1] if b is not None else None
        for (mod, p), g in zip(mods_params, grads[len(leaves):]):
            self._accumulate_param_grad(p, g)

        grad_h_in = grad_x.detach() if grad_x is not None else torch.zeros_like(h_in)
        if grad_b is not None:
            for j, l_entry in enumerate(self._bank_entries_before(layer_idx)):
                piece = grad_b[:, j].detach().view_as(grad_h_in)
                prev = self._grad_bank.get(l_entry)
                self._grad_bank[l_entry] = piece.clone() if prev is None else prev + piece
        if self._is_boundary(layer_idx) and layer_idx in self._grad_bank:
            grad_h_in = grad_h_in + self._grad_bank.pop(layer_idx).to(grad_h_in.dtype)

        self.expert_streamer.evict_layer_experts(layer_idx)
        return grad_h_in

    def _finalize_backward(self, h_last, grad_final):
        """Gradient through the output bank mix and the final RMSNorm, into h_last and the bank."""
        num_layers = self.config.model.num_hidden_layers
        bank = self._bank_at_entry(num_layers, h_last)
        with torch.enable_grad():
            x = h_last.detach().clone().requires_grad_(True)
            leaves = [x]
            b = None
            if bank is not None:
                b = bank.detach().clone().requires_grad_(True)
                leaves.append(b)
            out = self._finalize_hidden(x, b)
            grads = torch.autograd.grad(out, leaves, grad_outputs=grad_final.to(out.dtype), allow_unused=True)
        grad_x = grads[0].detach() if grads[0] is not None else torch.zeros_like(h_last)
        if b is not None and grads[1] is not None:
            for j, l_entry in enumerate(self._bank_entries_before(num_layers)):
                piece = grads[1][:, j].detach().view_as(grad_x)
                prev = self._grad_bank.get(l_entry)
                self._grad_bank[l_entry] = piece.clone() if prev is None else prev + piece
        return grad_x

    def _report_forward_loss(self, step: int, loss_val) -> None:
        """Print and persist the forward loss as soon as it is computed."""
        try:
            value = float(loss_val.item() if (HAS_TORCH and isinstance(loss_val, torch.Tensor)) else loss_val)
        except Exception:
            return

        import math as _math
        perplexity = _math.exp(value) if value < 20 else float("inf")
        print(f"\r  📉 [FORWARD LOSS] step {step}: {value:.4f}  (perplexity {perplexity:.1f})",
              flush=True)

        path = os.path.join(self.config.paths.workspace_dir, "forward_loss.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f'{{"step": {step}, "loss": {value:.6f}, "perplexity": {perplexity:.4f}, '
                        f'"time": {time.time():.0f}}}\n')
        except Exception:
            pass

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
        bytes_read_start = self.mmap_streamer.bytes_read

        # 1. Embed tokens
        if HAS_TORCH:
            no_grad_ctx = torch.no_grad()
            no_grad_ctx.__enter__()

        try:
            h_current = self._embed_tokens(input_ids)
            last_active_experts = []
            self._reset_block_residual()

            # 2. Sequential Layer-by-Layer Forward Pass (Layer 0 -> Layer L-1)
            for l in range(num_layers):
                if l % 5 == 0 or l == num_layers - 1:
                    print(f"\r  ⚡ [FORWARD PASS] Layer {l+1:02d}/{num_layers} (MoE Stream)", end="", flush=True)
                h_current, last_active_experts = self.forward_layer(l, h_current)

            print(f"\r  ⚡ [FORWARD COMPLETE] (93 Layers) -> Computing LM Head Cross-Entropy Loss...", end="", flush=True)
            # 3. Close the residual bank, final norm, LM head projection and loss
            h_last = h_current
            h_current = self._finalize_hidden(h_last, self._block_residual)
            logits = self._project_lm_head(h_current)
            loss_val, grad_logits = compute_cross_entropy_loss(
                logits,
                target_ids,
                ignore_index=self.config.model.pad_token_id,
            )

            # Record the forward loss the moment it exists. It is the one number that says
            # whether the pipeline reproduces the model, and the backward pass that follows
            # takes hours - losing it to a crash there would waste the whole run.
            self._report_forward_loss(step, loss_val)

            if not getattr(self, "forward_only", False):
                # 4. Out-of-Core Real Reverse Backward Pass (Layer L-1 -> Layer 0)
                # Backprop through LM head projection (streamed band by band), then through
                # the output bank mix and final norm
                grad_h = self._lm_head_backward(grad_logits)
                grad_h = self._finalize_backward(h_last, grad_h)

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
            disk_bytes_read=self.mmap_streamer.bytes_read - bytes_read_start,
        )

        if step % self.config.training.logging_steps == 0:
            self.dashboard.render(m)

        return loss_scalar

    def save_lora_checkpoint(self, step: int, data_cursor: int = 0) -> str:
        """
        Write a complete, atomic checkpoint.

        Contains the LoRA tensors, the optimizer moments and step counter, the LR schedule,
        the RNG states and the dataset cursor, so a resumed run continues the same
        trajectory. The file is written to a temporary name and renamed into place, so a
        crash mid-write never leaves a half checkpoint under the final name.
        """
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

        if not HAS_TORCH:
            np.savez_compressed(ckpt_path.replace(".pt", ".npz"), **state_dict)
            return ckpt_path.replace(".pt", ".npz")

        from dataclasses import asdict
        payload = {
            "format": 2,
            "step": int(step),
            "data_cursor": int(data_cursor),
            "lora": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
            "config": {"lora": asdict(self.config.lora), "training": asdict(self.config.training)},
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)
        try:
            with open(os.path.join(self.config.paths.checkpoints_dir, "latest.txt"), "w") as f:
                f.write(os.path.basename(ckpt_path) + "\n")
        except OSError:
            pass
        self._prune_checkpoints(keep=self.config.training.keep_checkpoints)
        return ckpt_path

    def _prune_checkpoints(self, keep: int) -> None:
        """Keep only the newest `keep` step checkpoints: each is ~1.8 GB (fp32 LoRA + Adam)
        and the NVMe has ~14 GB left beside the packed trunk."""
        if keep <= 0:
            return
        d = self.config.paths.checkpoints_dir
        try:
            files = sorted(f for f in os.listdir(d) if f.startswith("lazy_lora_step_") and f.endswith(".pt"))
        except OSError:
            return
        for f in files[:-keep]:
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Restore LoRA tensors, optimizer state and RNG from a checkpoint; returns its metadata."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "lora" not in payload:
            # A bare LoRA state dict (format 1): weights only, nothing else to restore.
            payload = {"format": 1, "step": 0, "data_cursor": 0, "lora": payload}
        lora = payload["lora"]
        missing = []
        with torch.no_grad():
            for l, bundle in enumerate(self.lora_layers):
                for name, mod in bundle.all_modules():
                    for ab, p in (("lora_A", mod.lora_A), ("lora_B", mod.lora_B)):
                        key = f"layers.{l}.{name}.{ab}"
                        if key in lora:
                            p.copy_(lora[key].to(p.dtype))
                        else:
                            missing.append(key)
        if missing:
            raise RuntimeError(f"checkpoint {path} lacks {len(missing)} LoRA tensors, e.g. {missing[:3]}")
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        rng = payload.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
        return {"step": int(payload.get("step", 0)), "data_cursor": int(payload.get("data_cursor", 0)),
                "format": payload.get("format", 1)}

    def train(self, num_steps: Optional[int] = None, resume_from: Optional[str] = None) -> List[float]:
        """
        Executes full LazyLoRA Out-of-Core training loop.
        Streams Turkish training dataset, processes micro-batches, updates weights,
        renders real-time dashboard, and periodically saves checkpoints. With
        `resume_from`, restores that checkpoint (weights, optimizer, RNG, data cursor)
        and continues from its step.
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
            tokenizer_dir=self.config.paths.base_model_dir,
        )

        print("=" * 82)
        print("  🚀 [STARTING LAZYLORA OUT-OF-CORE TURKISH TRAINING]  🚀")
        print(f"  Target Steps : {target_steps} | Micro-Batch: {self.config.training.micro_batch_size}")
        print(f"  Target Device: {self.device.upper()} | Model Layers: {self.config.model.num_hidden_layers}")
        print(f"  Checkpoints  : {self.config.paths.checkpoints_dir}")
        print("=" * 82)

        step = 0
        data_cursor = 0
        loss_history = []
        if resume_from:
            meta = self.load_checkpoint(resume_from)
            step, data_cursor = meta["step"], meta["data_cursor"]
            print(f"  ↩ [RESUMED] {resume_from}: step {step}, data cursor {data_cursor}")

        while step < target_steps:
            skip = data_cursor
            data_cursor = 0
            saw_batch = False
            for input_ids, target_ids in iterator.get_batches(
                batch_size=self.config.training.micro_batch_size,
                as_torch=HAS_TORCH,
                device=self.device,
                skip_samples=skip,
            ):
                saw_batch = True
                step += 1
                loss_val = self.train_step(step=step, input_ids=input_ids, target_ids=target_ids)
                loss_history.append(loss_val)
                data_cursor = iterator.samples_consumed

                if step % self.config.training.save_steps == 0:
                    saved_path = self.save_lora_checkpoint(step, data_cursor)
                    print(f"\n💾 [CHECKPOINT SAVED] Step {step:05d} -> {saved_path}")

                if step >= target_steps:
                    break
            if not saw_batch and skip:
                data_cursor = 0          # the cursor pointed past the end: start a new epoch
            elif not saw_batch:
                raise RuntimeError(f"no training samples in {train_file}")

        final_ckpt = self.save_lora_checkpoint(step, data_cursor)
        print("\n" + "=" * 82)
        print(f"🎉 [TRAINING COMPLETE] {step} Steps Executed Successfully!")
        print(f" Final LoRA Adapter Checkpoint: {final_ckpt}")
        print("=" * 82)
        self.close()
        return loss_history


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LazyLoRA Kimi K3 Training Runner")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--device", default=None, help="Device override (cuda:0 / cpu)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (shorter runs cost proportionally less disk time)")
    parser.add_argument("--forward-only", action="store_true",
                        help="Run the forward pass and report the loss, without backward or optimizer")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume from (weights, optimizer, RNG and data cursor)")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.steps:
        cfg.training.max_steps = args.steps
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.device:
        cfg.streaming.device = args.device
    if args.seq_len:
        cfg.training.max_seq_len = args.seq_len

    trainer = LazyLoRATrainer(config=cfg)
    trainer.forward_only = args.forward_only
    trainer.train(num_steps=args.steps, resume_from=args.resume)


if __name__ == "__main__":
    main()

