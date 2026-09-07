"""
Faithful Kimi Linear attention sublayers for the LazyLoRA out-of-core engine.

Kimi K3's text tower ("kimi_linear") interleaves two different attention types:

* **KDA (Kimi Delta Attention)** on 69 of the 93 layers - a gated delta-rule linear
  attention with depthwise short convolutions on q/k/v, per-channel decay and a
  sigmoid-gated output RMSNorm.
* **MLA (Multi-head Latent Attention)** on the remaining 24 layers - DeepSeek-style
  latent q/kv compression with an output gate. `mla_use_nope` is set, so no rotary
  embedding is applied and the "rope" halves are carried through unrotated.

The reference implementation delegates the KDA recurrence to Triton kernels from
`fla`, which need a compute capability this machine's GTX 980 Ti does not have and
which do not run on CPU at all. The recurrence is therefore written out here in
plain PyTorch, matching the semantics of `fused_recurrent_kda`:
l2-normalised q/k, sigmoid beta, per-channel decay clamped at `gate_lower_bound`.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is required for the real engine
    HAS_TORCH = False
    torch = None


def rms_norm(x, weight, eps: float = 1e-5):
    """RMSNorm over the last dimension, computed in float32 for stability."""
    dtype = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(dtype)


def l2_norm(x, eps: float = 1e-6):
    """L2-normalise the last dimension (applied to q and k inside the KDA kernel)."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps: float = 1e-5):
    """
    Kimi Linear's block-residual mixer.

    Instead of accumulating sublayer outputs into one ever-growing residual stream, the
    model keeps a bank of residual snapshots (one every `attn_res_block_size` layers) and
    mixes them with the live stream through a softmax over learned per-vector scores. The
    result is a convex combination, which is what keeps activations bounded across 93
    layers - without it the stream grows without limit.

    prefix_sum:     [N, hidden]
    block_residual: [N, num_blocks, hidden]  (may have num_blocks == 0)
    proj_weight:    [1, hidden]   (self_attention_res_proj / mlp_res_proj / output_attn_res_proj)
    norm_weight:    [hidden]      (the matching *_res_norm)
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)

    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)                 # [N, num_blocks + 1]
    probs = scores.softmax(-1).unsqueeze(1)             # [N, 1, num_blocks + 1]

    mixed = torch.matmul(probs, v_float).squeeze(1)     # [N, hidden]
    return mixed.to(v.dtype)


def short_convolution(x, weight):
    """
    Causal depthwise convolution followed by SiLU, as in `fla.modules.ShortConvolution`.

    x:      [B, T, C]
    weight: [C, 1, K]
    """
    kernel = weight.shape[-1]
    xt = x.transpose(1, 2)                       # [B, C, T]
    xt = F.pad(xt, (kernel - 1, 0))              # causal left padding
    y = F.conv1d(xt, weight.to(xt.dtype), groups=x.shape[-1])
    return F.silu(y.transpose(1, 2))


def _with_lora(base, x, lora):
    """base + LoRA delta, when an adapter is attached to this projection."""
    if lora is None:
        return base
    return base + lora.forward_lora_only(x)


KDA_CHUNK = 16


def _kda_chunk(state, q, k, v, decay, beta):
    """
    The gated delta rule over one chunk of `n` tokens, from state S [B, H, D, D]:
        S_t = S_{t-1} diag(a_t)
        S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T
        o_t = S_t^T q_t
    Returns (new state, outputs [B, n, H, D]).
    """
    # Same elementwise formulation as the validated single-loop version (matmul changes
    # the fp32 summation order, which moved the 13-layer C comparison by ~4e-4).
    B, H = k.shape[0], k.shape[2]
    outs = []
    for t in range(q.shape[1]):
        state = state * decay[:, t].unsqueeze(-1)
        k_t = k[:, t]                                                # [B, H, D]
        read = (state * k_t.unsqueeze(-1)).sum(dim=2)                # S^T k  [B, H, D]
        err = v[:, t] - read
        state = state + beta[:, t].view(B, H, 1, 1) * k_t.unsqueeze(-1) * err.unsqueeze(2)
        outs.append((state * q[:, t].unsqueeze(-1)).sum(dim=2))
    return state, torch.stack(outs, dim=1)


def _kda_recurrence(q, k, v, decay, beta):
    """
    Run the recurrence over T tokens in chunks of KDA_CHUNK.

    Under autograd each chunk is a checkpoint: only the chunk-boundary state is kept and
    the chunk is recomputed during the backward. Without it the graph of a 256-token
    layer held every per-step [B, H, D, D] intermediate (several GB), pushed the process
    into swap, and a single layer's backward took a quarter of an hour on one core.
    """
    B, T, H, D = q.shape
    state = torch.zeros(B, H, D, D, dtype=torch.float32, device=q.device)
    outs = []
    use_ckpt = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad
                                             or decay.requires_grad or beta.requires_grad)
    for s in range(0, T, KDA_CHUNK):
        e = min(T, s + KDA_CHUNK)
        args = (state, q[:, s:e], k[:, s:e], v[:, s:e], decay[:, s:e], beta[:, s:e])
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
            state, o = checkpoint(_kda_chunk, *args, use_reentrant=False)
        else:
            state, o = _kda_chunk(*args)
        outs.append(o)
    return torch.cat(outs, dim=1)


def kda_attention(
    h,
    w,
    num_heads: int = 96,
    head_dim: int = 128,
    gate_lower_bound: Optional[float] = -5.0,
    eps: float = 1e-5,
    q_lora=None,
    v_lora=None,
):
    """
    Kimi Delta Attention.

    `w` is an object exposing the layer's tensors: q_proj, k_proj, v_proj,
    q_conv1d, k_conv1d, v_conv1d, f_a_proj, f_b_proj, dt_bias, A_log, b_proj,
    g_proj, o_norm, o_proj.

    Returns the attention output [B, T, hidden].
    """
    B, T, _ = h.shape
    H, D = num_heads, head_dim

    q = short_convolution(_with_lora(F.linear(h, w.q_proj), h, q_lora), w.q_conv1d)
    k = short_convolution(F.linear(h, w.k_proj), w.k_conv1d)
    v = short_convolution(_with_lora(F.linear(h, w.v_proj), h, v_lora), w.v_conv1d)

    q = l2_norm(q.view(B, T, H, D).float())
    k = l2_norm(k.view(B, T, H, D).float())
    v = v.view(B, T, H, D).float()

    # Data-dependent decay. A_log is stored with head_dim entries but is indexed PER
    # HEAD - only the first num_heads of them are nonzero, so indexing it per channel
    # silently gives most heads a decay of exp(0) = 1 and no forgetting at all.
    #
    #   u     = exp(A_log[h]) * (z + dt_bias)
    #   g     = gate_lower_bound * sigmoid(u)      -> lands in (lower_bound, 0]
    #   decay = exp(g)
    #
    # The bound is built into the sigmoid rather than applied as a clamp afterwards.
    z = F.linear(F.linear(h, w.f_a_proj), w.f_b_proj).view(B, T, H, D).float()
    z = z + w.dt_bias.view(1, 1, H, D).float()
    a = torch.exp(w.A_log.float()[:H]).view(1, 1, H, 1)
    lb = gate_lower_bound if gate_lower_bound is not None else -5.0
    g = lb * torch.sigmoid(a * z)
    decay = torch.exp(g)                                     # [B, T, H, D]

    beta = torch.sigmoid(F.linear(h, w.b_proj).float())      # [B, T, H]

    # Gated delta rule, one token at a time, with q pre-scaled by d_k^-0.5:
    #   S <- S * diag(decay_t)
    #   S <- S + beta_t * k_t (v_t - S^T k_t)^T
    #   o_t = S^T q_t
    q = q * (D ** -0.5)
    outputs = _kda_recurrence(q, k, v, decay, beta)

    o = outputs.to(h.dtype)

    # Sigmoid-gated output RMSNorm over the head dimension, then output projection
    gate = F.linear(h, w.g_proj).view(B, T, H, D)
    o = rms_norm(o, w.o_norm, eps=eps) * torch.sigmoid(gate.float()).to(o.dtype)
    o = o.reshape(B, T, H * D)
    return F.linear(o, w.o_proj)


def mla_attention(
    h,
    w,
    num_heads: int = 96,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    kv_lora_rank: int = 512,
    eps: float = 1e-5,
    q_lora=None,
    v_lora=None,
):
    """
    Multi-head Latent Attention (DeepSeek-style), with `mla_use_nope` semantics:
    the rope halves are concatenated unrotated, and an output gate is applied.
    """
    B, T, _ = h.shape
    H = num_heads
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scaling = q_head_dim ** -0.5

    q_latent = rms_norm(F.linear(h, w.q_a_proj), w.q_a_layernorm, eps=eps)
    q = _with_lora(F.linear(q_latent, w.q_b_proj), q_latent, q_lora)
    q = q.view(B, T, H, q_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    compressed = F.linear(h, w.kv_a_proj_with_mqa)
    k_latent, k_rot = torch.split(compressed, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_latent = rms_norm(k_latent, w.kv_a_layernorm, eps=eps)
    kv = _with_lora(F.linear(kv_latent, w.kv_b_proj), kv_latent, v_lora)
    kv = kv.view(B, T, H, qk_nope_head_dim + v_head_dim).transpose(1, 2)
    k_pass, value = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    k_rot = k_rot.view(B, 1, T, qk_rope_head_dim).expand(B, H, T, qk_rope_head_dim)

    query = torch.cat((q_pass, q_rot), dim=-1)
    key = torch.cat((k_pass, k_rot), dim=-1)

    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(value.dtype)

    o = torch.matmul(probs, value)                            # [B, H, T, v_head_dim]
    o = o.transpose(1, 2).reshape(B, T, H * v_head_dim)
    o = o * torch.sigmoid(F.linear(h, w.g_proj).float()).to(o.dtype)
    return F.linear(o, w.o_proj)
