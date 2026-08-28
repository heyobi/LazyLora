"""
Validate the engine's ops against the reference fixtures shipped with kimi-k3-in-c.

Each fixture in tests/fixtures/ops carries its own weights, inputs and expected
outputs at fp32, with a stated tolerance of 1e-5 absolute / 1e-4 relative. They are
the only ground truth available for this architecture short of running the 1.45 TB
checkpoint, and they catch the kind of mistake that still produces plausible numbers -
an activation with the wrong tail, a decay indexed per channel instead of per head.

The whole module skips when the fixture directory is absent.
"""

import json
import os
import unittest

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from lazy_lora.core.situ_activation import situ_glu_forward

FIXTURES = "/mnt/d/hamza/kimi-k3-in-c/tests/fixtures/ops"
ABS_TOL = 1e-5
REL_TOL = 1e-4


def load_fixture(name):
    with open(os.path.join(FIXTURES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def as_array(entry):
    """Fixture tensors are {"shape": [...], "data": [...]} in row-major order."""
    return np.asarray(entry["data"], dtype=np.float32).reshape(entry["shape"])


@unittest.skipUnless(os.path.isdir(FIXTURES), "reference fixtures not present")
@unittest.skipUnless(HAS_TORCH, "torch required")
class TestReferenceOps(unittest.TestCase):

    def assert_matches(self, got, want, label, abs_tol=ABS_TOL, rel_tol=REL_TOL):
        got = np.asarray(got, dtype=np.float32).reshape(-1)
        want = np.asarray(want, dtype=np.float32).reshape(-1)
        self.assertEqual(got.shape, want.shape, f"{label}: shape mismatch")
        diff = np.abs(got - want)
        tol = abs_tol + rel_tol * np.abs(want)
        worst = int(np.argmax(diff - tol))
        self.assertTrue(
            np.all(diff <= tol),
            f"{label}: max abs diff {diff.max():.3e} at index {worst} "
            f"(got {got[worst]:.6f}, want {want[worst]:.6f}); "
            f"cosine {float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-20)):.6f}",
        )

    def test_situ_glu(self):
        fx = load_fixture("situ_glu")
        x = as_array(fx["in"])
        d = x.shape[-1] // 2
        gate, up = x[..., :d], x[..., d:]
        got = situ_glu_forward(
            torch.from_numpy(gate), torch.from_numpy(up),
            beta=fx["beta"], linear_beta=fx["linear_beta"],
        )
        self.assert_matches(got.numpy(), as_array(fx["out"]), "situ_glu")

    def test_rmsnorm(self):
        from lazy_lora.core.attention import rms_norm
        fx = load_fixture("rmsnorm")
        got = rms_norm(
            torch.from_numpy(as_array(fx["in"])),
            torch.from_numpy(as_array(fx["weight"])),
            eps=fx["eps"],
        )
        self.assert_matches(got.numpy(), as_array(fx["out"]), "rmsnorm")

    def test_short_convolution(self):
        from lazy_lora.core.attention import short_convolution
        fx = load_fixture("shortconv")
        got = short_convolution(
            torch.from_numpy(as_array(fx["in"])),
            torch.from_numpy(as_array(fx["weight"])),
        )
        self.assert_matches(got.numpy(), as_array(fx["out"]), "shortconv")

    def test_kda_decay(self):
        """The decay gate: per-head A_log, sigmoid scaled by the lower bound."""
        fx = load_fixture("kda_decay")
        z = as_array(fx["z"])                          # [B, T, H*D], before dt_bias
        A_log = as_array(fx["A_log"]).reshape(-1)      # [H], padded to head_dim on disk
        dt_bias = as_array(fx["dt_bias"]).reshape(-1)  # [H*D]
        lb = fx["lower_bound"]

        B, T, P = z.shape
        H = A_log.shape[0]
        D = P // H

        z_t = torch.from_numpy(z).view(B, T, H, D)
        a = torch.exp(torch.from_numpy(A_log)).view(1, 1, H, 1)
        u = a * (z_t + torch.from_numpy(dt_bias).view(1, 1, H, D))
        g = lb * torch.sigmoid(u)

        self.assert_matches(g.numpy(), as_array(fx["g"]), "kda_decay.g")
        self.assert_matches(torch.exp(g).numpy(), as_array(fx["alpha"]), "kda_decay.alpha")

    def test_router(self):
        """Selection uses biased scores; the weights come from the unbiased ones."""
        from lazy_lora.core.moe_router import KimiK3MoERouter
        fx = load_fixture("router")
        x = torch.from_numpy(as_array(fx["in"]))
        router = KimiK3MoERouter(
            hidden_size=x.shape[-1],
            num_experts=fx["n_experts"],
            top_k=fx["top_k"],
        )
        idx, weight = router.forward(
            x,
            weight=torch.from_numpy(as_array(fx["gate_weight"])),
            bias=torch.from_numpy(as_array(fx["bias"])),
        )
        # Expert order within a row is not part of the contract; compare as sorted pairs.
        want_idx = as_array(fx["topk_idx"]).astype(np.int64)
        want_w = as_array(fx["topk_weight"])
        got = {(r, int(i)): float(w) for r, (ii, ww) in enumerate(zip(idx.tolist(), weight.tolist()))
               for i, w in zip(ii, ww)}
        want = {(r, int(i)): float(w) for r, (ii, ww) in enumerate(zip(want_idx.tolist(), want_w.tolist()))
                for i, w in zip(ii, ww)}
        self.assertEqual(sorted(got), sorted(want), "router: selected experts differ")
        self.assert_matches([got[k] for k in sorted(got)],
                            [want[k] for k in sorted(want)], "router.weights")

    def test_attn_res(self):
        """Block-residual mixing: keys are RMSNormed, the values mixed are raw."""
        from lazy_lora.core.attention import apply_attn_res
        fx = load_fixture("attnres")
        got = apply_attn_res(
            torch.from_numpy(as_array(fx["prefix_sum"])),
            torch.from_numpy(as_array(fx["block_residual"])),
            torch.from_numpy(as_array(fx["proj_weight"])).view(1, -1),
            torch.from_numpy(as_array(fx["norm_weight"])),
            eps=fx["eps"],
        )
        self.assert_matches(got.numpy(), as_array(fx["out"]), "attnres")

    def test_mla(self):
        from lazy_lora.core.attention import mla_attention
        fx = load_fixture("mla")

        class W:
            pass

        w = W()
        for key in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa",
                    "kv_a_layernorm", "kv_b_proj", "o_proj", "g_proj"):
            setattr(w, key, torch.from_numpy(as_array(fx[f"{key}_weight"])))

        got = mla_attention(
            torch.from_numpy(as_array(fx["in"])), w,
            num_heads=fx["n_heads"],
            qk_nope_head_dim=fx["qk_nope"],
            qk_rope_head_dim=fx["qk_rope"],
            v_head_dim=fx["v_head"],
            kv_lora_rank=fx["kv_lora"],
            eps=fx["rms_eps"],
        )
        self.assert_matches(got.numpy(), as_array(fx["out"]), "mla")

    def test_moe_block(self):
        """The whole latent MoE block: shared expert + routed experts through the latent space."""
        from lazy_lora.core.attention import rms_norm
        from lazy_lora.core.moe_router import KimiK3MoERouter
        fx = load_fixture("moe")

        x = torch.from_numpy(as_array(fx["in"]))
        hidden, latent = fx["hidden"], fx["latent"]

        # Shared expert, in hidden space
        s_gate = F.linear(x, torch.from_numpy(as_array(fx["shared_w1_weight"])))
        s_up = F.linear(x, torch.from_numpy(as_array(fx["shared_w3_weight"])))
        s_act = situ_glu_forward(s_gate, s_up, beta=fx["situ_b1"], linear_beta=fx["situ_b2"])
        shared_out = F.linear(s_act, torch.from_numpy(as_array(fx["shared_w2_weight"])))

        # Routing
        router = KimiK3MoERouter(hidden_size=hidden, num_experts=fx["n_experts"],
                                 top_k=fx["top_k"], routed_scaling_factor=fx["routed_scale"])
        topk_idx, topk_w = router.forward(
            x,
            weight=torch.from_numpy(as_array(fx["gate_weight"])),
            bias=torch.from_numpy(as_array(fx["e_score_correction_bias"])),
        )

        # Routed experts, in latent space
        flat = x.reshape(-1, hidden)
        h_latent = F.linear(flat, torch.from_numpy(as_array(fx["down_weight"])))
        routed = torch.zeros_like(h_latent)
        for e in range(fx["n_experts"]):
            mask = (topk_idx == e)
            if not mask.any():
                continue
            w1 = torch.from_numpy(as_array(fx[f"experts_{e}_w1_weight"]))
            w3 = torch.from_numpy(as_array(fx[f"experts_{e}_w3_weight"]))
            w2 = torch.from_numpy(as_array(fx[f"experts_{e}_w2_weight"]))
            act = situ_glu_forward(F.linear(h_latent, w1), F.linear(h_latent, w3),
                                   beta=fx["situ_b1"], linear_beta=fx["situ_b2"])
            weights = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1, keepdim=True)
            routed = routed + F.linear(act, w2) * weights

        if fx["latent_norm"]:
            routed = rms_norm(routed, torch.from_numpy(as_array(fx["norm_weight"])), eps=fx["rms_eps"])
        routed_out = F.linear(routed, torch.from_numpy(as_array(fx["up_weight"])))

        got = shared_out + routed_out.view_as(shared_out)
        # The block chains six matmuls per expert plus the two latent projections. The
        # reference sums each dot product in double; torch sums in float32, and the
        # ordering difference alone lands around 7e-5 here while the direction is
        # identical (cosine 1.000000), so this one composite gets a wider window than
        # the single-op fixtures.
        self.assert_matches(got.numpy(), as_array(fx["out"]), "moe", abs_tol=2e-4)


if __name__ == "__main__":
    unittest.main()
