"""
Dynamic Expert-Wise Streaming Engine for Out-of-Core MoE Training.
Loads ONLY active top-k experts and shared experts into RAM on-demand.
Supports MXFP4 quantized weights (weight_packed + weight_scale) from Kimi K3.
"""

import os
import queue
import threading
from typing import Dict, List, Optional, Tuple, Union, Set
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
from lazy_lora.core.situ_activation import situ_glu_forward
from lazy_lora.core.config import synthetic_allowed


class ExpertWeightBundle:
    """
    Weights of one MoE expert, in one of two forms:

    * dequantised: gate_proj / up_proj / down_proj as [out, in] tensors (mock experts, the
      shared expert, and the fallback path), or
    * packed: the MXFP4 bytes as loaded, (packed, scale) per matrix, consumed directly by
      lazy_lora.native (no 132 MB widening per expert). `packed` is True in that case.
    """
    def __init__(
        self,
        expert_idx: int,
        gate_proj=None, up_proj=None, down_proj=None,
        gate_packed=None, gate_scale=None, up_packed=None, up_scale=None,
        down_packed=None, down_scale=None,
    ):
        self.expert_idx = expert_idx
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.gate_packed, self.gate_scale = gate_packed, gate_scale
        self.up_packed, self.up_scale = up_packed, up_scale
        self.down_packed, self.down_scale = down_packed, down_scale
        self.packed = gate_packed is not None


# FP4 (E2M1) values indexed by the whole nibble: bit 3 is the sign, the low 3 bits pick
# the magnitude. The reference kernel uses exactly this table.
FP4_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
FP4_E2M1_MAGNITUDES = FP4_E2M1[:8]      # kept for callers that want magnitudes only

_PAIR_LUT = None                        # [256, 2] float32: byte -> (even, odd) element

# 2^(253-127) * 6 already overflows float32, so any scale byte at or above this marks a
# group that cannot be represented - in practice, damaged bytes. 255 is E8M0's own NaN.
MAX_SAFE_E8M0 = 253

_damaged_groups = 0
_damaged_total = 0


def _note_damaged_groups(count: int, total: int) -> None:
    """Count unusable scale groups and say so once every 10000, not once per tensor."""
    global _damaged_groups, _damaged_total
    before = _damaged_groups
    _damaged_groups += count
    _damaged_total += total
    if _damaged_groups // 10000 != before // 10000:
        print(f"\n[!] {_damaged_groups} unusable MXFP4 scale groups skipped so far "
              f"({_damaged_groups / max(_damaged_total, 1):.3%} of those read). "
              f"These are damaged bytes in the checkpoint; the groups contribute zero.",
              flush=True)


def damaged_group_stats():
    """(unusable groups, groups inspected) since the process started."""
    return _damaged_groups, _damaged_total


def _pair_lut(device):
    """
    One lookup per packed byte instead of unpacking each nibble arithmetically.

    Dequantisation was costing as much as the disk read it feeds (9.0 s against 9.1 s
    for twelve experts). Masking, shifting, widening to int64 and materialising a
    per-element scale walks ~11M elements a dozen times per tensor; a 256-entry table
    indexed by the byte turns that into one gather, and the group scale multiplies a
    [rows, groups, 1] view instead of an expanded copy.
    """
    global _PAIR_LUT
    if _PAIR_LUT is None or _PAIR_LUT.device != device:
        pairs = [[FP4_E2M1[b & 0x0F], FP4_E2M1[(b >> 4) & 0x0F]] for b in range(256)]
        _PAIR_LUT = torch.tensor(pairs, dtype=torch.float32, device=device)
    return _PAIR_LUT


def _dequantize_mxfp4(weight_packed, weight_scale, group_size: int = 32, out_dtype=None):
    """
    Dequantize Kimi K3's routed experts, which are stored as MXFP4.

    config.json declares `"format": "mxfp4-pack-quantized"` with `num_bits: 4`,
    `group_size: 32`, `type: "float"` and `scale_dtype: torch.uint8`:

    * `weight_packed` holds two FP4 (E2M1) codes per byte, low nibble first. Each code
      is a sign bit plus a 3-bit index into FP4_E2M1_MAGNITUDES.
    * `weight_scale` holds one E8M0 exponent per group of 32 input channels, so the
      group multiplier is 2^(scale - 127), not the raw byte value.
    * A scale byte of 255 is E8M0's NaN encoding and marks a group that contributes
      nothing. The reference kernel skips those groups outright
      (`if (sb == 255) continue;` in k3_matmul_mxfp4).
    * Anything at or above 253 is treated the same way. 2^(253-127) multiplied by the
      largest FP4 magnitude of 6 already exceeds float32, so such a group can only
      yield inf, and inf * 0 in the next matmul yields NaN which then travels through
      every remaining layer. No trained weight is 1e38; a scale that large means the
      bytes are damaged. This checkpoint has such regions - three crashes during the
      download left corrupted expert blocks in a handful of shards - and one of them
      turned an entire 93-layer forward pass into NaN.

    Sanity check on layer 1 expert 0: the result lands at |w| ~ 0.015, matching the
    unquantised shared expert of the same layer (absmean 0.0149).
    """
    if HAS_TORCH and isinstance(weight_packed, torch.Tensor):
        rows, pcols = weight_packed.shape
        n_in = pcols * 2

        # byte -> (even, odd) value, then flatten the pair axis back into the row
        values = _pair_lut(weight_packed.device)[weight_packed.long()].reshape(rows, n_in)

        # Checking the scale bytes (a few hundred KB) rather than the dequantised values
        # (tens of MB) keeps the guard essentially free.
        zero = torch.zeros((), dtype=torch.float32, device=values.device)
        unusable = weight_scale >= MAX_SAFE_E8M0
        multiplier = torch.where(
            unusable, zero, torch.exp2(weight_scale.to(torch.float32) - 127.0)
        )
        if bool(unusable.any()):
            _note_damaged_groups(int(unusable.sum()), int(unusable.numel()))

        if weight_scale.dim() == 2:
            n_groups = weight_scale.shape[1]
            group = n_in // n_groups
            # Broadcast over the group instead of expanding it into a full-size tensor
            values = (values.view(rows, n_groups, group) * multiplier.unsqueeze(-1)).view(rows, n_in)
        else:
            values = values * multiplier.view(-1, 1)

        return values.to(out_dtype or torch.bfloat16)

    if isinstance(weight_packed, np.ndarray):
        codes = np.stack(
            [weight_packed & 0x0F, (weight_packed >> 4) & 0x0F], axis=-1
        ).reshape(weight_packed.shape[0], weight_packed.shape[1] * 2).astype(np.int32)

        lut = np.asarray(FP4_E2M1_MAGNITUDES, dtype=np.float32)
        values = lut[codes & 0x7]
        values = np.where(codes & 0x8 != 0, -values, values)

        if weight_scale.ndim == 2:
            groups = max(values.shape[1] // weight_scale.shape[1], 1)
            raw_scale = np.repeat(weight_scale, groups, axis=1)[:, :values.shape[1]]
        else:
            raw_scale = weight_scale.reshape(-1, 1)

        multiplier = np.where(raw_scale == 255, 0.0,
                              np.exp2(raw_scale.astype(np.float32) - 127.0))
        return values * multiplier

    return weight_packed


class DynamicExpertStreamer:
    """
    Expert-Wise Dynamic Streamer for Kimi K3 MoE layers.
    Zero-cache mode: loads each expert from mmap on-demand and discards immediately.
    """

    def __init__(
        self,
        mmap_streamer: MmapTensorStreamer,
        device: str = "cpu",
        max_resident_experts: int = 18,  # 16 top-k + 2 shared
        async_prefetch: bool = True,
        hidden_size: int = 7168,
        latent_size: int = 3584,
        moe_intermediate_size: int = 3072,
        shared_intermediate_size: Optional[int] = None,
    ):
        self.mmap_streamer = mmap_streamer
        self.device = device
        self.max_resident_experts = max_resident_experts
        self.async_prefetch = async_prefetch
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.moe_intermediate_size = moe_intermediate_size
        # Kimi K3 fuses its 2 shared experts into one module of width 2 * moe_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size or (moe_intermediate_size * 2)
        # With the native MXFP4 kernel the routed experts are handed over as packed bytes
        # and never widened here; the reader thread then only reads.
        from lazy_lora.native import kernel as _native_kernel
        self.keep_packed = _native_kernel() is not None

    @property
    def weight_dtype(self):
        if HAS_TORCH and getattr(self.mmap_streamer, "upcast_float32", False):
            return torch.float32
        return torch.bfloat16 if HAS_TORCH else np.float32

    def _load_single_expert(self, layer_idx: int, expert_idx: int, is_shared: bool = False) -> ExpertWeightBundle:
        """
        Load gate (w1), up (w3), down (w2) projections for an expert from mmap.
        Kimi K3 naming:
          Routed: model.layers.{L}.block_sparse_moe.experts.{E}.w1/w2/w3.weight_packed/weight_scale
          Shared: model.layers.{L}.block_sparse_moe.shared_experts.gate_proj/up_proj/down_proj.weight
        """
        if is_shared:
            # Shared experts are a single module (not indexed), with full-precision weights
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.shared_experts."
            gate = self.mmap_streamer.load_tensor(f"{prefix}gate_proj.weight", target_device=self.device)
            up = self.mmap_streamer.load_tensor(f"{prefix}up_proj.weight", target_device=self.device)
            down = self.mmap_streamer.load_tensor(f"{prefix}down_proj.weight", target_device=self.device)
        else:
            # Routed experts: MXFP4 quantized (weight_packed + weight_scale)
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}."

            # Try the packed format first
            w1_packed = self.mmap_streamer.load_tensor(f"{prefix}w1.weight_packed", target_device=self.device)
            w1_scale = self.mmap_streamer.load_tensor(f"{prefix}w1.weight_scale", target_device=self.device)
            w2_packed = self.mmap_streamer.load_tensor(f"{prefix}w2.weight_packed", target_device=self.device)
            w2_scale = self.mmap_streamer.load_tensor(f"{prefix}w2.weight_scale", target_device=self.device)
            w3_packed = self.mmap_streamer.load_tensor(f"{prefix}w3.weight_packed", target_device=self.device)
            w3_scale = self.mmap_streamer.load_tensor(f"{prefix}w3.weight_scale", target_device=self.device)

            if w1_packed is not None and w1_scale is not None and self.keep_packed:
                return ExpertWeightBundle(
                    expert_idx,
                    gate_packed=w1_packed, gate_scale=w1_scale,
                    up_packed=w3_packed, up_scale=w3_scale,
                    down_packed=w2_packed, down_scale=w2_scale,
                )
            if w1_packed is not None and w1_scale is not None:
                gate = _dequantize_mxfp4(w1_packed, w1_scale, out_dtype=self.weight_dtype)
                down = _dequantize_mxfp4(w2_packed, w2_scale, out_dtype=self.weight_dtype)
                up = _dequantize_mxfp4(w3_packed, w3_scale, out_dtype=self.weight_dtype)
            else:
                # Fallback: try full-precision names
                gate = self.mmap_streamer.load_tensor(f"{prefix}w1.weight", target_device=self.device)
                down = self.mmap_streamer.load_tensor(f"{prefix}w2.weight", target_device=self.device)
                up = self.mmap_streamer.load_tensor(f"{prefix}w3.weight", target_device=self.device)

        # Synthetic fallback with CORRECT dimensions for testing (mock suite only)
        if gate is None or up is None or down is None:
            synthetic_allowed(f"layer {layer_idx} {'shared expert' if is_shared else f'routed expert {expert_idx}'} weights")
            if is_shared:
                # Shared expert: intermediate_size = moe_intermediate_size * num_shared_experts
                d_in = self.hidden_size                 # 7168
                d_mid = self.shared_intermediate_size   # 3072*2 = 6144
            else:
                # Routed expert: operates in latent space
                d_in = self.latent_size            # 3584
                d_mid = self.moe_intermediate_size # 3072

            if HAS_TORCH:
                gate = torch.randn(d_mid, d_in, dtype=torch.bfloat16, device=self.device) * 0.01
                up = torch.randn(d_mid, d_in, dtype=torch.bfloat16, device=self.device) * 0.01
                down = torch.randn(d_in, d_mid, dtype=torch.bfloat16, device=self.device) * 0.01
            else:
                gate = (np.random.randn(d_mid, d_in) * 0.01).astype(np.float32)
                up = (np.random.randn(d_mid, d_in) * 0.01).astype(np.float32)
                down = (np.random.randn(d_in, d_mid) * 0.01).astype(np.float32)

        # Ensure the compute dtype (bfloat16, or float32 under LAZYLORA_COMPUTE_FP32)
        if HAS_TORCH and isinstance(gate, torch.Tensor) and gate.dtype != self.weight_dtype:
            gate = gate.to(self.weight_dtype)
            up = up.to(self.weight_dtype)
            down = down.to(self.weight_dtype)

        return ExpertWeightBundle(expert_idx, gate, up, down)

    def sort_by_disk_order(self, layer_idx: int, expert_ids: List[int]) -> List[int]:
        """
        Order experts by their physical offset in the shard files.

        The active set of a 512-token batch spans hundreds of experts scattered across the
        shards. Reading them in router order makes a mechanical disk seek back and forth;
        reading them in on-disk order turns the same bytes into a near-sequential sweep.
        """
        index = self.mmap_streamer.index

        def key(exp_id: int):
            name = f"model.layers.{layer_idx}.block_sparse_moe.experts.{exp_id}.w1.weight_packed"
            resolved = index._resolve_name(name)
            if resolved is None:
                return (1, "", 0)
            shard_path, start, _end, _shape, _dtype = index.tensor_locations[resolved]
            return (0, shard_path, start)

        return sorted(expert_ids, key=key)

    def request_prefetch_layer(self, layer_idx: int, active_experts: List[int], include_shared: bool = True) -> None:
        """
        Kept for compatibility. Prefetching across layers cannot work: routing is data
        dependent, so the next layer's expert set is unknown until its router has run.
        Use stream_experts, which prefetches WITHIN a layer, where the whole list is
        known up front.
        """
        pass

    def stream_experts(self, layer_idx: int, expert_ids: List[int], depth: int = 2):
        """
        Yield (expert_id, bundle) while a background thread reads ahead.

        Measured on this machine the disk sits idle for most of a layer: each expert is
        six small reads followed by MXFP4 decoding and GEMMs on the CPU, and neither side
        overlaps the other. Reading ahead by `depth` experts keeps the drive moving while
        the current expert is being multiplied. The ids are expected in on-disk order
        (see sort_by_disk_order), so the read-ahead stays a forward sweep.

        Only `depth` bundles are resident beyond the one in use, which bounds the extra
        RAM at roughly depth * 17.5 MB for Kimi K3's experts.
        """
        if not expert_ids:
            return

        if not self.async_prefetch or len(expert_ids) < 2:
            for exp_id in expert_ids:
                yield exp_id, self._load_single_expert(layer_idx, exp_id, False)
            return

        queue_out: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        stop = threading.Event()

        def reader():
            try:
                for exp_id in expert_ids:
                    if stop.is_set():
                        break
                    bundle = self._load_single_expert(layer_idx, exp_id, False)
                    while not stop.is_set():
                        try:
                            queue_out.put((exp_id, bundle), timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:                      # surfaced on the consumer side
                queue_out.put(("__error__", exc))
            finally:
                queue_out.put((None, None))

        worker = threading.Thread(target=reader, daemon=True)
        worker.start()

        try:
            while True:
                exp_id, bundle = queue_out.get()
                if exp_id is None:
                    break
                if exp_id == "__error__":
                    raise bundle
                yield exp_id, bundle
        finally:
            stop.set()
            worker.join(timeout=2.0)

    def get_expert(self, layer_idx: int, expert_idx: int, is_shared: bool = False) -> ExpertWeightBundle:
        """Fetch expert bundle on-demand from zero-copy mmap."""
        return self._load_single_expert(layer_idx, expert_idx, is_shared)

    def evict_layer_experts(self, layer_idx: int) -> None:
        """No-op in zero-cache mode (nothing to evict)."""
        import gc
        gc.collect()

    def close(self) -> None:
        pass
