#!/usr/bin/env python3
"""
Generate a tiny, real, Kimi-K3-shaped checkpoint so the engine can be run without the
1.56 TB one.

    python scripts/make_tiny_model.py <out_dir> [--seed 0]

It writes real safetensors bytes: two shards, a model.safetensors.index.json and a
config.json: 8,324,144 bytes of tensor data plus two small headers, worked out from the
geometry rather than observed (the script prints the exact byte count it wrote). Four layers
that mix both attention types, hidden size 256, eight MXFP4 experts per MoE layer with
top-2 routing. Every tensor the streaming loader asks for is present, which is the whole
point: with this checkpoint on disk the engine runs with LAZYLORA_ALLOW_SYNTHETIC unset -
the safety gate that forbids substituting random weights (core/config.py:166-182) stays
CLOSED, and nothing is faked at run time. The values are deterministic noise; the bytes,
the format, the quantisation and the geometry are real, which is what makes the engine's
behaviour on it meaningful.

---------------------------------------------------------------------------------------
!! WRITTEN WITHOUT BEING EXECUTED !!
This file was written by reading the loader, the trainer and the MXFP4 kernel line by
line while the machine was busy with the 29-day, 100-step training run, so it has never
been run. It must be validated once that run frees the machine: `bash scripts/quickstart.sh`
is the validation. Until then, treat every number in this docstring and every number it
prints as DERIVED FROM THE CODE, not measured: the byte counts and the tensor counts below
were worked out from the geometry by hand, and the runtimes quoted in docs/QUICKSTART.md
were reasoned about, not timed.
---------------------------------------------------------------------------------------

WHY EACH CHOICE IS WHAT IT IS
=============================

File layout
-----------
* `SafetensorsIndex.build_index` (streaming/mmap_loader.py:64-133) globs `*.safetensors`
  in the model directory and parses each file itself: 8-byte little-endian header length,
  then that many bytes of JSON, then the data, with `data_offset_base = 8 + header_len`
  (mmap_loader.py:104). It reads neither `config.json` nor `model.safetensors.index.json`.
  We emit both anyway, because `scripts/check_shards.py` walks the index (check_shards.py:53-90)
  and because a checkpoint directory without them is not a checkpoint directory.
* Two shards named `model-0000N-of-00002.safetensors`: the `model-*` glob is what
  `lazy_lora/tests/test_real_safetensors_headers.py:29` looks for, and two of them
  exercise the multi-shard indexer instead of the one-file case.
* `check_shards.py:83-86` recomputes `8 + header_len + last_data_offset` and compares it
  with the file size, so the header is padded to an 8-byte boundary with spaces *inside*
  the declared header length and the tensor data follows with no gaps.
* dtypes are the two the reader maps: `"BF16"` -> uint16 raw bits and `"U8"` -> uint8
  (mmap_loader.py:25-34). BF16 is reinterpreted with `torch.Tensor.view(torch.bfloat16)`
  (mmap_loader.py:326), so the bytes must be the top 16 bits of the float32, rounded to
  nearest even.
* Tensor names are written in the plain `model.` / `lm_head.` form the engine asks for;
  `_resolve_name` (mmap_loader.py:194-205) would also accept the `language_model.`
  prefixes of the real checkpoint, but there is no reason to make a reader guess.

Geometry
--------
Symbols: L layers, H hidden, P = num_attention_heads * head_dim, Di dense intermediate,
Dm moe_intermediate_size, Dl routed_expert_hidden_size, Ds = Dm * num_shared_experts,
E experts, V vocab.

* `num_hidden_layers=4`, `full_attn_layers=[2]` (1-based, exactly as the real config
  lists it, see core/config.py:41-46): `is_kda_layer` (core/config.py:60-62) then makes
  layer index 1 an MLA layer and layers 0, 2, 3 KDA layers, so both attention paths run.
* `first_k_dense_replace=1`: layer 0 carries a plain MLP (`_load_dense_weight`,
  trainer/lazy_trainer.py:587-593) and layers 1-3 carry experts, as in the real model.
* `attn_res_block_size=2`: `_is_boundary` (lazy_trainer.py:473-475) makes layers 0 and 2
  block boundaries, and `_bank_entries_before` (lazy_trainer.py:477-482) then gives
  layer 1 a one-entry residual bank and layer 3 a two-entry one. The bank, its gradient
  routing and `apply_attn_res` are all exercised; with a larger block size on 4 layers
  they would not be.
* `head_dim >= num_attention_heads`, because KDA reads the per-head decay out of a tensor
  stored with head_dim entries: `w.A_log.float()[:H]` (core/attention.py:189).
* `v_head_dim == head_dim`. MLA reshapes its output to `[B, T, num_heads * v_head_dim]`
  and multiplies it by the output gate `g_proj`, which is `[P, H]`
  (core/attention.py:256-257). The two widths must therefore agree.
* `qk_nope_head_dim + qk_rope_head_dim != head_dim` on purpose: that asymmetry is real
  (128 + 64 against a head_dim of 128) and it is what `q_b_proj`'s row count encodes.
* `Dm % 32 == 0` and `Dl % 32 == 0`, and both <= 4096. The MXFP4 kernel decodes exactly
  `K/32` groups of 32 (native/mxfp4_gemm.c:49-61) into a fixed stack tile sized for
  K <= 4096 (mxfp4_gemm.c:72); a K that is not a multiple of 32 leaves the tail of that
  tile uninitialised and the product silently mixes in stack garbage, and the Python
  guard cannot catch it because `K // 32` floors (native/__init__.py:42).
* `vocab_size > 355`: the dataset iterator's byte fallback emits `byte + 100`
  (dataset/stream_dataset.py:136), so a smaller vocabulary would index off the table.
  `pad_token_id` is kept inside the vocabulary as well, although it only ever appears as
  a cross-entropy `ignore_index` (trainer/loss.py:41-47).
* `num_experts_per_token <= num_experts`, or `torch.topk` in the router
  (core/moe_router.py:92) has nothing to select from.

Tensors, and who reads them
---------------------------
Global (5):
    model.embed_tokens.weight            [V, H]  BF16   lazy_trainer.py:332 (row gather)
    lm_head.weight                       [V, H]  BF16   lazy_trainer.py:366-384 (row bands)
    model.norm.weight                    [H]     BF16   lazy_trainer.py:403
    model.output_attn_res_proj.weight    [1, H]  BF16   lazy_trainer.py:396
    model.output_attn_res_norm.weight    [H]     BF16   lazy_trainer.py:397

Every layer, prefix `model.layers.{n}.` (6), all read by `load_layer_trunk`
(streaming/trunk_streamer.py:99-106):
    input_layernorm.weight               [H]
    post_attention_layernorm.weight      [H]
    self_attention_res_norm.weight       [H]
    self_attention_res_proj.weight       [1, H]     <- squeezed at core/attention.py:68
    mlp_res_norm.weight                  [H]
    mlp_res_proj.weight                  [1, H]

KDA layers, prefix `model.layers.{n}.self_attn.` (14) - the list is KDA_TENSORS
(trunk_streamer.py:37-42) and the shapes are the ones `_fill_synthetic_attention`
declares (trunk_streamer.py:197-213). Note that `dt_bias` and `A_log` carry no
`.weight` suffix:
    q_proj.weight  k_proj.weight  v_proj.weight   [P, H] each
    q_conv1d.weight k_conv1d.weight v_conv1d.weight [P, 1, K]  <- F.conv1d groups=P, attention.py:83-87
    f_a_proj.weight [head_dim, H]      f_b_proj.weight [P, head_dim]
    dt_bias         [P]                <- viewed as [1,1,heads,head_dim], attention.py:188
    A_log           [head_dim]         <- only the first `num_heads` entries are read, attention.py:189
    b_proj.weight   [num_heads, H]     g_proj.weight [P, H]
    o_norm.weight   [head_dim]         o_proj.weight [H, P]

MLA layers, same prefix (8) - MLA_TENSORS (trunk_streamer.py:44-48):
    q_a_proj.weight             [q_lora_rank, H]
    q_a_layernorm.weight        [q_lora_rank]
    q_b_proj.weight             [num_heads * (qk_nope + qk_rope), q_lora_rank]
    kv_a_proj_with_mqa.weight   [kv_lora_rank + qk_rope, H]
    kv_a_layernorm.weight       [kv_lora_rank]
    kv_b_proj.weight            [num_heads * (qk_nope + v_head_dim), kv_lora_rank]
    g_proj.weight               [P, H]
    o_proj.weight               [H, P]

Dense layer (n < first_k_dense_replace), prefix `model.layers.{n}.mlp.` (3),
read by `_load_dense_weight` / `_dense_mlp_forward` (lazy_trainer.py:587-616):
    gate_proj.weight [Di, H]   up_proj.weight [Di, H]   down_proj.weight [H, Di]

MoE layers, prefix `model.layers.{n}.block_sparse_moe.` (8 + 6E):
    gate.weight                        [E, H]   BF16  lazy_trainer.py:522, router at moe_router.py:84
    gate.e_score_correction_bias       [E]      BF16  lazy_trainer.py:523, steers selection only
    routed_expert_down_proj.weight     [Dl, H]  BF16  lazy_trainer.py:711
    routed_expert_up_proj.weight       [H, Dl]  BF16  lazy_trainer.py:712
    routed_expert_norm.weight          [Dl]     BF16  lazy_trainer.py:713
    shared_experts.gate_proj.weight    [Ds, H]  BF16  expert_streamer.py:225
    shared_experts.up_proj.weight      [Ds, H]  BF16  expert_streamer.py:226
    shared_experts.down_proj.weight    [H, Ds]  BF16  expert_streamer.py:227
    experts.{e}.w1.weight_packed [Dm, Dl/2] U8   experts.{e}.w1.weight_scale [Dm, Dl/32] U8
    experts.{e}.w3.weight_packed [Dm, Dl/2] U8   experts.{e}.w3.weight_scale [Dm, Dl/32] U8
    experts.{e}.w2.weight_packed [Dl, Dm/2] U8   experts.{e}.w2.weight_scale [Dl, Dm/32] U8

  w1 is the gate branch, w3 the up branch and w2 the down branch
  (expert_streamer.py:240-250). All six tensors of an expert must exist: the packed
  branch is chosen on `w1.weight_packed` alone, and a half-packed expert would then
  crash. The shapes follow from the kernel's own assertions
  (native/__init__.py:42 for `gemm`, :68 for `gemm_t`): for `y = x @ W^T` with
  `x [M, K]`, `packed` is `[R, K/2]` and `scales` is `[R, K/32]`. The gate and up
  matrices consume the latent width (K = Dl, R = Dm) and the down matrix consumes the
  expert width (K = Dm, R = Dl).
  Ds = moe_intermediate_size * num_shared_experts: Kimi K3 fuses its shared experts into
  one wider module, which is how the streamer sizes it (lazy_trainer.py:213,
  expert_streamer.py:202-203).

MXFP4 encoding
--------------
OCP MX FP4, exactly as the C kernel and the torch dequantiser read it:
* two E2M1 codes per byte, element `2j` in the LOW nibble of byte `j` and `2j+1` in the
  high nibble (mxfp4_gemm.c:56-59, expert_streamer.py:99).
* a code is a sign bit (bit 3) plus a 3-bit index into
  {0, 0.5, 1, 1.5, 2, 3, 4, 6} (expert_streamer.py:55-56, mxfp4_gemm.c:33-34).
* one E8M0 scale byte per group of 32 input channels; the multiplier is `2^(byte - 127)`,
  not the byte (expert_streamer.py:141, mxfp4_gemm.c:41).
* a byte >= 253 marks a group that cannot be represented and contributes zero
  (expert_streamer.py:63, mxfp4_gemm.c:41). This generator never emits one: the scales
  sit in 119..123, i.e. multipliers 2^-8 .. 2^-4. With the code distribution below the
  mean code magnitude is 1.57 and the mean multiplier 0.0242, which puts |w| around 0.04 -
  the same order of magnitude as the real experts (absmean 0.015,
  expert_streamer.py:126-127), not the same number. Both figures are arithmetic on the
  distribution, not a measurement; the `mxfp4 check` line the script prints is the one that
  reports what was actually written. The scales vary per group so that the per-group
  scaling is genuinely exercised rather than being a constant factor.

Determinism
-----------
Every tensor is drawn from SHAKE-256 keyed by (seed, tensor name), so the bytes depend on
nothing but the seed and the geometry - not on the order tensors are generated in, not on
the NumPy version, and not on the platform. The script prints the sha256 of each shard;
two people with the same seed must see the same digests. NumPy is the only import needed
to *generate* the shards; the `--no-verify` path uses nothing else. The default path also
re-reads the result with the engine's own indexer, which pulls torch in (see
`verify_with_engine`).
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np

# --------------------------------------------------------------------------- geometry

# The default tiny geometry. Every key here must be a field of
# lazy_lora.core.config.KimiK3ArchitectureConfig; `_check_against_dataclass` enforces it,
# so this file cannot grow a second architecture that drifts away from the engine's.
DEFAULT_GEOMETRY = {
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "intermediate_size": 512,          # dense MLP of layer 0
    "moe_intermediate_size": 64,       # Dm, multiple of 32
    "routed_expert_hidden_size": 128,  # Dl, multiple of 32
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 2,           # -> shared width Ds = 128
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 96,                    # -> P = 384, deliberately != hidden_size
    "q_lora_rank": 128,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 96,
    "qk_rope_head_dim": 48,            # -> q_head_dim = 144, deliberately != head_dim
    "v_head_dim": 96,                  # must equal head_dim
    "vocab_size": 2048,
    "pad_token_id": 2047,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rms_norm_eps": 1e-5,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 1.0,
    "full_attn_layers": [2],           # 1-based: layer index 1 is MLA
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
    "mla_use_output_gate": True,
    "attn_res_block_size": 2,          # boundaries at layers 0 and 2
    "situ_beta": 4.0,
    "situ_linear_beta": 25.0,
    "activation_func": "situ",
    "dtype": "bfloat16",
}

# Keys that config.json may carry although they are not architecture fields.
EXTRA_CONFIG_KEYS = {
    "model_type", "architectures", "torch_dtype", "quantization_config",
    "tie_word_embeddings", "_lazylora_tiny",
}


def build_geometry(overrides):
    g = dict(DEFAULT_GEOMETRY)
    g.update({k: v for k, v in overrides.items() if v is not None})
    validate_geometry(g)
    return g


def validate_geometry(g):
    """The five invariants the engine relies on and never checks, plus the kernel's."""
    dm, dl = g["moe_intermediate_size"], g["routed_expert_hidden_size"]
    if dm % 32 or dl % 32:
        raise SystemExit(f"moe_intermediate_size ({dm}) and routed_expert_hidden_size ({dl}) "
                         f"must be multiples of 32: the MXFP4 kernel decodes whole groups of 32 "
                         f"and K // 32 floors, so a remainder is multiplied against uninitialised "
                         f"stack memory (native/mxfp4_gemm.c:49-61, native/__init__.py:42).")
    if dm > 4096 or dl > 4096:
        raise SystemExit("moe_intermediate_size and routed_expert_hidden_size must be <= 4096: "
                         "the kernel's decode tile is a fixed-size stack array (mxfp4_gemm.c:72).")
    if g["v_head_dim"] != g["head_dim"]:
        raise SystemExit("v_head_dim must equal head_dim: MLA reshapes to num_heads * v_head_dim "
                         "and gates it with g_proj, which is num_heads * head_dim wide "
                         "(core/attention.py:256-257).")
    if g["head_dim"] < g["num_attention_heads"]:
        raise SystemExit("head_dim must be >= num_attention_heads: the per-head decay is read as "
                         "A_log[:num_heads] out of a head_dim-long tensor (core/attention.py:189).")
    if g["num_experts_per_token"] > g["num_experts"]:
        raise SystemExit("num_experts_per_token must be <= num_experts (core/moe_router.py:92).")
    if g["vocab_size"] <= 355:
        raise SystemExit("vocab_size must exceed 355: the dataset iterator's byte fallback emits "
                         "byte + 100 (dataset/stream_dataset.py:136).")
    if not 0 <= g["pad_token_id"] < g["vocab_size"]:
        raise SystemExit("pad_token_id must lie inside the vocabulary.")
    if max(g["full_attn_layers"] or [0]) > g["num_hidden_layers"]:
        raise SystemExit("full_attn_layers is 1-based and must not name a layer beyond "
                         "num_hidden_layers (core/config.py:41-46).")
    if g["first_k_dense_replace"] >= g["num_hidden_layers"]:
        raise SystemExit("first_k_dense_replace must leave at least one MoE layer.")


def _check_against_dataclass(g):
    """Refuse to emit an architecture key KimiK3ArchitectureConfig does not have."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from dataclasses import fields
        from lazy_lora.core.config import KimiK3ArchitectureConfig
    except Exception as exc:                       # running outside the repo
        print(f"[!] could not import KimiK3ArchitectureConfig ({exc}); "
              f"the config.json key check was skipped", flush=True)
        return
    known = {f.name for f in fields(KimiK3ArchitectureConfig)}
    unknown = sorted(set(g) - known)
    if unknown:
        raise SystemExit(
            f"these keys are not fields of KimiK3ArchitectureConfig and would create a second, "
            f"drifting architecture: {unknown}")
    missing = sorted(known - set(g))
    if missing:
        print(f"[i] architecture fields left at their engine defaults: {', '.join(missing)}", flush=True)


# ------------------------------------------------------------------ deterministic bytes

def _stream(seed, label, nbytes):
    """`nbytes` bytes from SHAKE-256 keyed by (seed, label).

    Keyed by the tensor's own name rather than by a running counter, so the content of a
    tensor does not depend on how many tensors were generated before it, and the same
    bytes come out on every platform and every NumPy version.
    """
    return hashlib.shake_256(f"lazylora-tiny|{seed}|{label}".encode("utf-8")).digest(nbytes)


def _u01(seed, label, count):
    """`count` float64 in [0, 1)."""
    raw = _stream(seed, label, count * 4)
    return np.frombuffer(raw, dtype="<u4").astype(np.float64) * (1.0 / 4294967296.0)


def uniform(seed, label, shape, std):
    """Uniform with the requested standard deviation, as float32."""
    count = int(np.prod(shape))
    half = math.sqrt(3.0) * std
    return ((_u01(seed, label, count) * 2.0 - 1.0) * half).reshape(shape).astype(np.float32)


def near_one(seed, label, shape, jitter=0.02):
    """RMSNorm weights: around 1, as trained norms are."""
    return (1.0 + uniform(seed, label, shape, jitter)).astype(np.float32)


def to_bf16(x):
    """float32 -> the raw uint16 bits of bfloat16, round to nearest even.

    mmap_loader.py:326 reinterprets these bytes with `.view(torch.bfloat16)`, so they
    must be the top half of the float32 word, not a conversion of any other kind.
    """
    # Every operand is an explicit uint32 so that no NumPy version can promote the
    # arithmetic to float64 behind our backs. |x| here is far below 1e38, so the addition
    # cannot overflow, and after the shift the result fits in 16 bits exactly.
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounded = (u + lsb + np.uint32(0x7FFF)) >> np.uint32(16)
    return rounded.astype("<u2")


# ------------------------------------------------------------------------ MXFP4 packing

# Magnitudes indexed by the low three bits of a code; bit 3 is the sign.
FP4_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Weighted towards the small codes, as a trained weight distribution is.
FP4_CDF = np.cumsum(np.array([0.06, 0.20, 0.24, 0.18, 0.14, 0.10, 0.05, 0.03]))
SCALE_LO, SCALE_HI = 119, 123          # E8M0 bytes: multipliers 2^-8 .. 2^-4, never >= 253


def mxfp4_block(seed, label, rows, k):
    """(packed [rows, k/2] uint8, scale [rows, k/32] uint8) for a logical [rows, k] matrix."""
    assert k % 32 == 0, k
    mag = np.searchsorted(FP4_CDF, _u01(seed, label + "|mag", rows * k)).astype(np.uint8)
    np.clip(mag, 0, 7, out=mag)
    sign = (_u01(seed, label + "|sign", rows * k) < 0.5).astype(np.uint8) << 3
    codes = (mag | sign).reshape(rows, k)
    # byte j holds element 2j in the low nibble and 2j+1 in the high nibble
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    n_groups = k // 32
    scale = (SCALE_LO + np.floor(_u01(seed, label + "|scale", rows * n_groups)
                                 * (SCALE_HI - SCALE_LO + 1))).astype(np.uint8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(scale.reshape(rows, n_groups))


def mxfp4_absmean(packed, scale):
    """Mean |w| of a packed block, for the report line (pure NumPy, mirrors the kernel)."""
    lo = FP4_MAGNITUDE[packed & 0x07]
    hi = FP4_MAGNITUDE[(packed >> 4) & 0x07]
    mags = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    mags[:, 0::2] = lo
    mags[:, 1::2] = hi
    mult = np.exp2(scale.astype(np.float32) - 127.0)
    groups = mags.shape[1] // scale.shape[1]
    return float((mags.reshape(mags.shape[0], scale.shape[1], groups)
                  * mult[:, :, None]).mean())


# ---------------------------------------------------------------------------- the plan

class Plan:
    """Ordered (name, array, dtype string) triples, one per tensor."""

    def __init__(self, seed):
        self.seed = seed
        self.items = []

    def bf16(self, name, shape, std=None, kind="weight"):
        if kind == "norm":
            arr = near_one(self.seed, name, shape)
        else:
            arr = uniform(self.seed, name, shape, std)
        self.items.append((name, to_bf16(arr).reshape(shape), "BF16"))

    def u8(self, name, arr):
        self.items.append((name, np.ascontiguousarray(arr, dtype=np.uint8), "U8"))


def build_plan(g, seed):
    """Every tensor of the tiny model, in the order it is written."""
    H = g["hidden_size"]
    heads, hd = g["num_attention_heads"], g["head_dim"]
    P = heads * hd
    Di = g["intermediate_size"]
    Dm = g["moe_intermediate_size"]
    Dl = g["routed_expert_hidden_size"]
    Ds = Dm * g["num_shared_experts"]
    E = g["num_experts"]
    V = g["vocab_size"]
    K = g["short_conv_kernel_size"]
    qr, kr = g["q_lora_rank"], g["kv_lora_rank"]
    nope, rope, vh = g["qk_nope_head_dim"], g["qk_rope_head_dim"], g["v_head_dim"]
    full_attn = set(g["full_attn_layers"])

    def fan(n):
        """1/sqrt(fan_in): keeps the pre-activation of every projection near unit scale."""
        return 1.0 / math.sqrt(n)

    p = Plan(seed)

    # ---- global
    p.bf16("model.embed_tokens.weight", (V, H), std=0.02)
    p.bf16("lm_head.weight", (V, H), std=0.02)
    p.bf16("model.norm.weight", (H,), kind="norm")
    # The residual-bank score is `sum(rmsnorm(v) * norm_weight * proj_weight)` over H
    # channels (core/attention.py:66-70). With proj entries at 1/sqrt(H) the scores land
    # near unit scale and the softmax over bank entries actually mixes; at scale 1 it
    # would saturate onto a single entry and the mechanism would be untested.
    p.bf16("model.output_attn_res_proj.weight", (1, H), std=fan(H))
    p.bf16("model.output_attn_res_norm.weight", (H,), kind="norm")

    layer_items = []
    for n in range(g["num_hidden_layers"]):
        start = len(p.items)
        pre = f"model.layers.{n}."
        is_kda = (n + 1) not in full_attn
        is_dense = n < g["first_k_dense_replace"]

        # ---- per-layer trunk (trunk_streamer.py:99-106)
        p.bf16(pre + "input_layernorm.weight", (H,), kind="norm")
        p.bf16(pre + "post_attention_layernorm.weight", (H,), kind="norm")
        p.bf16(pre + "self_attention_res_norm.weight", (H,), kind="norm")
        p.bf16(pre + "self_attention_res_proj.weight", (1, H), std=fan(H))
        p.bf16(pre + "mlp_res_norm.weight", (H,), kind="norm")
        p.bf16(pre + "mlp_res_proj.weight", (1, H), std=fan(H))

        a = pre + "self_attn."
        if is_kda:
            # ---- KDA (trunk_streamer.py:37-42, shapes at :197-213)
            for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                p.bf16(a + name + ".weight", (P, H), std=fan(H))
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                # depthwise causal conv, F.conv1d(groups=P) with weight [P, 1, K]
                p.bf16(a + name + ".weight", (P, 1, K), std=0.5)
            p.bf16(a + "f_a_proj.weight", (hd, H), std=fan(H))
            p.bf16(a + "f_b_proj.weight", (P, hd), std=fan(hd))
            p.bf16(a + "dt_bias", (P,), std=0.1)          # no .weight suffix on disk
            p.bf16(a + "A_log", (hd,), std=0.5)           # exp(A_log[:heads]) scales the decay
            p.bf16(a + "b_proj.weight", (heads, H), std=fan(H))
            p.bf16(a + "o_norm.weight", (hd,), kind="norm")
            p.bf16(a + "o_proj.weight", (H, P), std=fan(P))
        else:
            # ---- MLA (trunk_streamer.py:44-48)
            p.bf16(a + "q_a_proj.weight", (qr, H), std=fan(H))
            p.bf16(a + "q_a_layernorm.weight", (qr,), kind="norm")
            p.bf16(a + "q_b_proj.weight", (heads * (nope + rope), qr), std=fan(qr))
            p.bf16(a + "kv_a_proj_with_mqa.weight", (kr + rope, H), std=fan(H))
            p.bf16(a + "kv_a_layernorm.weight", (kr,), kind="norm")
            p.bf16(a + "kv_b_proj.weight", (heads * (nope + vh), kr), std=fan(kr))
            p.bf16(a + "g_proj.weight", (P, H), std=fan(H))
            p.bf16(a + "o_proj.weight", (H, P), std=fan(P))

        if is_dense:
            # ---- dense MLP (lazy_trainer.py:587-616)
            m = pre + "mlp."
            p.bf16(m + "gate_proj.weight", (Di, H), std=fan(H))
            p.bf16(m + "up_proj.weight", (Di, H), std=fan(H))
            p.bf16(m + "down_proj.weight", (H, Di), std=fan(Di))
        else:
            # ---- latent MoE (lazy_trainer.py:697-730, expert_streamer.py:215-284)
            b = pre + "block_sparse_moe."
            p.bf16(b + "gate.weight", (E, H), std=fan(H))
            p.bf16(b + "gate.e_score_correction_bias", (E,), std=0.02)
            p.bf16(b + "routed_expert_down_proj.weight", (Dl, H), std=fan(H))
            p.bf16(b + "routed_expert_up_proj.weight", (H, Dl), std=fan(Dl))
            p.bf16(b + "routed_expert_norm.weight", (Dl,), kind="norm")
            p.bf16(b + "shared_experts.gate_proj.weight", (Ds, H), std=fan(H))
            p.bf16(b + "shared_experts.up_proj.weight", (Ds, H), std=fan(H))
            p.bf16(b + "shared_experts.down_proj.weight", (H, Ds), std=fan(Ds))
            for e in range(E):
                x = f"{b}experts.{e}."
                for role, rows, k in (("w1", Dm, Dl), ("w2", Dl, Dm), ("w3", Dm, Dl)):
                    packed, scale = mxfp4_block(seed, x + role, rows, k)
                    p.u8(x + role + ".weight_packed", packed)
                    p.u8(x + role + ".weight_scale", scale)

        layer_items.append((n, start, len(p.items)))

    return p.items, layer_items


# --------------------------------------------------------------------- safetensors file

def write_shard(path, tensors, metadata):
    """One safetensors file: <u64 header length><JSON header><tensor data>."""
    header = {}
    offset = 0
    for name, arr, dtype_str in tensors:
        nbytes = int(arr.nbytes)
        header[name] = {"dtype": dtype_str, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The official format aligns the data to 8 bytes by padding the header with spaces,
    # and the padding counts towards the declared header length. check_shards.py:83-86
    # recomputes 8 + header_len + last_offset and compares it with the file size.
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _name, arr, _dtype in tensors:
            f.write(arr.tobytes())
    return offset, len(blob)


def sha256_of(path, bufsize=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------- config.json

def build_config(g):
    """config.json: the architecture keys verbatim, plus a short fixed set of extras.

    Nothing here is read by the engine (it never opens config.json); it is read by people,
    by `scripts/quickstart.sh` via `load_tiny_config`, and by anything that expects a
    model directory to look like a model directory.
    """
    extras = {
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"],
        "torch_dtype": g["dtype"],
        "tie_word_embeddings": False,
        "quantization_config": {
            "quant_method": "mxfp4-pack-quantized",
            "format": "mxfp4-pack-quantized",
            "num_bits": 4,
            "group_size": 32,
            "type": "float",
            "scale_dtype": "uint8",
        },
        "_lazylora_tiny": (
            "Synthetic Kimi-K3-shaped checkpoint written by scripts/make_tiny_model.py. "
            "Real bytes, real format, random values; it exists so the engine can be run "
            "and checked without the 1.56 TB checkpoint. It is not Kimi K3 and it knows "
            "nothing."
        ),
    }
    unexpected = sorted(set(extras) - EXTRA_CONFIG_KEYS)
    if unexpected:
        raise SystemExit(f"non-architecture config keys must be declared in "
                         f"EXTRA_CONFIG_KEYS first: {unexpected}")
    cfg = dict(g)
    cfg.update(extras)
    return cfg


# --------------------------------------------------------------------- self-verification

def verify_with_engine(out_dir, plan):
    """Re-read the written shards with the engine's own indexer.

    Uses SafetensorsIndex, so what is checked is exactly what the trainer will see:
    the same header parsing, the same name resolution, the same shape and dtype table.
    Uses no torch itself, though importing mmap_loader pulls torch in when it is installed
    (mmap_loader.py:17-18), which is most of what this step costs.
    """
    os.environ.setdefault("LAZYLORA_CACHE_DIR", os.path.join(out_dir, ".index_cache"))
    os.environ["LAZYLORA_TRUNK_DIR"] = ""          # no NVMe trunk overlay for a tiny model
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lazy_lora.streaming.mmap_loader import SafetensorsIndex

    index = SafetensorsIndex(out_dir)
    if index.bad_shards:
        return [f"shards the engine could not index: {index.bad_shards}"]
    problems = []
    for name, arr, dtype_str in plan:
        resolved = index._resolve_name(name)
        if resolved is None:
            problems.append(f"{name}: not found by the engine's index")
            continue
        _path, start, end, shape, dtype = index.tensor_locations[resolved]
        if list(shape) != list(arr.shape):
            problems.append(f"{name}: index says {shape}, wrote {list(arr.shape)}")
        if dtype != dtype_str:
            problems.append(f"{name}: index says {dtype}, wrote {dtype_str}")
        if (end - start) != int(arr.nbytes):
            problems.append(f"{name}: index spans {end - start} bytes, wrote {arr.nbytes}")
    extra = set(index.tensor_locations) - {n for n, _a, _d in plan}
    if extra:
        problems.append(f"{len(extra)} tensors in the directory that this run did not write, "
                        f"e.g. {sorted(extra)[:3]} - a stale shard from another geometry?")
    return problems


# --------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Write a tiny, real, Kimi-K3-shaped safetensors checkpoint.")
    ap.add_argument("out_dir", help="directory to write config.json, the shards and the index into")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=2, help="number of shard files (default 2)")
    ap.add_argument("--layers", type=int, default=None, help="override num_hidden_layers")
    ap.add_argument("--hidden", type=int, default=None, help="override hidden_size")
    ap.add_argument("--experts", type=int, default=None, help="override num_experts")
    ap.add_argument("--top-k", type=int, default=None, help="override num_experts_per_token")
    ap.add_argument("--vocab", type=int, default=None, help="override vocab_size")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip re-reading the result with the engine's own indexer")
    args = ap.parse_args()

    ov = {
        "num_hidden_layers": args.layers,
        "hidden_size": args.hidden,
        "num_experts": args.experts,
        "num_experts_per_token": args.top_k,
        "vocab_size": args.vocab,
    }
    # The pad id has to move with the vocabulary *before* validation, not after it:
    # build_geometry validates, and the default pad_token_id (2047) is outside any
    # vocabulary smaller than 2048, so `--vocab 512` would otherwise exit on
    # "pad_token_id must lie inside the vocabulary" before it could be corrected.
    if args.vocab is not None:
        ov["pad_token_id"] = args.vocab - 1
    g = build_geometry(ov)
    _check_against_dataclass(g)

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # A shard left over from a different geometry would be indexed alongside the new ones
    # (build_index globs *.safetensors), so the directory is cleared of them first.
    stale = sorted(f for f in os.listdir(out_dir) if f.endswith(".safetensors"))
    for f in stale:
        os.remove(os.path.join(out_dir, f))
    if stale:
        print(f"[i] removed {len(stale)} shard(s) from a previous run: {', '.join(stale)}")

    t0 = time.time()
    plan, layer_spans = build_plan(g, args.seed)

    # Split on layer boundaries so a shard never holds half a layer, which is also how the
    # real checkpoint is cut.
    n_shards = max(1, args.shards)
    per = max(1, int(math.ceil(len(plan) / n_shards)))
    cuts = [0]
    for _n, _start, end in layer_spans:
        if end - cuts[-1] >= per and len(cuts) < n_shards:
            cuts.append(end)
    cuts.append(len(plan))
    groups = [plan[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    groups = [gr for gr in groups if gr]
    n_shards = len(groups)

    weight_map = {}
    total_bytes = 0
    files = []
    for i, group in enumerate(groups, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        path = os.path.join(out_dir, fname)
        data_bytes, header_bytes = write_shard(
            path, group,
            {"format": "pt", "produced_by": "LazyLoRA scripts/make_tiny_model.py",
             "seed": str(args.seed)},
        )
        for name, _arr, _dtype in group:
            weight_map[name] = fname
        total_bytes += data_bytes
        files.append((fname, path, len(group), data_bytes, header_bytes))

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                  f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(build_config(g), f, indent=2, sort_keys=True)

    # ---- report
    P = g["num_attention_heads"] * g["head_dim"]
    on_disk = sum(os.path.getsize(p) for _f, p, _c, _b, _h in files)
    print(f"model dir     : {out_dir}")
    print(f"geometry      : {g['num_hidden_layers']} layers "
          f"(MLA at 1-based {sorted(g['full_attn_layers'])}, dense below "
          f"{g['first_k_dense_replace']}), hidden {g['hidden_size']}, "
          f"attn {g['num_attention_heads']}x{g['head_dim']} = {P}, "
          f"experts {g['num_experts']} top-{g['num_experts_per_token']} "
          f"({g['moe_intermediate_size']} wide in a {g['routed_expert_hidden_size']} latent), "
          f"vocab {g['vocab_size']}")
    print(f"tensors       : {len(plan)}")
    for fname, _path, count, data_bytes, header_bytes in files:
        print(f"  {fname}  {count:4d} tensors  {data_bytes / 1e6:6.2f} MB data "
              f"+ {header_bytes / 1e3:.1f} kB header")
    print(f"bytes on disk : {on_disk / 1e6:.2f} MB   (index total_size {total_bytes / 1e6:.2f} MB, "
          f"the difference is the headers)")
    print(f"generated in  : {time.time() - t0:.1f} s")
    for fname, path, _c, _b, _h in files:
        print(f"sha256        : {sha256_of(path)}  {fname}")

    # One expert block, decoded the way the kernel decodes it, as a sanity line. Derived
    # from the distribution, this should print about 0.038 (mean code magnitude 1.57 times
    # mean multiplier 0.0242); the real checkpoint's experts sit at |w| ~ 0.015
    # (expert_streamer.py:126-127), i.e. the same order of magnitude, not the same number.
    first_moe = g["first_k_dense_replace"]
    label = f"model.layers.{first_moe}.block_sparse_moe.experts.0.w1"
    packed, scale = mxfp4_block(args.seed, label, g["moe_intermediate_size"],
                                g["routed_expert_hidden_size"])
    print(f"mxfp4 check   : layer {first_moe} expert 0 w1 absmean |w| = "
          f"{mxfp4_absmean(packed, scale):.4f}, scale bytes "
          f"{int(scale.min())}..{int(scale.max())} (must stay below 253)")

    if not args.no_verify:
        problems = verify_with_engine(out_dir, plan)
        if problems:
            print("\nFAIL: the engine's own indexer disagrees with what was written:")
            for line in problems[:20]:
                print("  -", line)
            return 1
        print(f"verified      : all {len(plan)} tensors resolve through the engine's own "
              f"SafetensorsIndex with the expected shape and dtype")

    print("\nOK. Point the engine at it:")
    print(f"  export LAZYLORA_MODEL_DIR={out_dir}")
    print("  (LAZYLORA_ALLOW_SYNTHETIC stays unset: nothing here needs substituting)")
    return 0


# ------------------------------------------------------------- used by scripts/quickstart.sh

def load_tiny_config(model_dir, config=None):
    """A LazyLoraConfig whose architecture is the one in <model_dir>/config.json.

    The engine hardcodes Kimi K3's 93 layers and 7168 hidden size in
    KimiK3ArchitectureConfig (core/config.py:16-58) and never reads a config.json, so a
    caller that wants to run against the tiny model has to say so. Only keys that are
    fields of the dataclass are applied, so nothing else in config.json can leak in.
    """
    from dataclasses import fields
    from lazy_lora.core.config import get_default_config

    cfg = config or get_default_config()
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    known = {f.name for f in fields(cfg.model)}
    applied = []
    for key in sorted(raw):
        if key in known:
            setattr(cfg.model, key, raw[key])
            applied.append(key)
    cfg.paths.base_model_dir = os.path.abspath(model_dir)
    return cfg, applied


if __name__ == "__main__":
    sys.exit(main())
