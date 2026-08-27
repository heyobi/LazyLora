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

    # Data-dependent per-channel decay: g = -exp(A_log) * softplus(f(h) + dt_bias)
    g = F.linear(F.linear(h, w.f_a_proj), w.f_b_proj).view(B, T, H, D).float()
    g = g + w.dt_bias.view(1, 1, H, D).float()
    g = -torch.exp(w.A_log.float()).view(1, 1, 1, D) * F.softplus(g)
    if gate_lower_bound is not None:
        g = g.clamp(min=gate_lower_bound)
    decay = torch.exp(g)                                     # [B, T, H, D]

    beta = torch.sigmoid(F.linear(h, w.b_proj).float())      # [B, T, H]

    # Gated delta rule, one token at a time:
    #   S <- S * diag(decay_t)
    #   S <- S + beta_t * k_t (v_t - S^T k_t)^T
    #   o_t = S^T q_t
    state = torch.zeros(B, H, D, D, dtype=torch.float32, device=h.device)
    outputs = torch.empty(B, T, H, D, dtype=torch.float32, device=h.device)
    for t in range(T):
        state = state * decay[:, t].unsqueeze(-1)
        k_t = k[:, t]                                        # [B, H, D]
        v_t = v[:, t]
        read = (state * k_t.unsqueeze(-1)).sum(dim=2)        # [B, H, D]
        err = v_t - read
        state = state + beta[:, t].view(B, H, 1, 1) * k_t.unsqueeze(-1) * err.unsqueeze(2)
        outputs[:, t] = (state * q[:, t].unsqueeze(-1)).sum(dim=2)

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
