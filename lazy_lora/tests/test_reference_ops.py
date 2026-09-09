"""
Validate the engine's ops against the reference fixtures shipped with kimi-k3-in-c.

Each fixture in tests/fixtures/ops carries its own weights, inputs and expected outputs at
fp32. Seven of the eight match at 1e-5 absolute / 1e-4 relative; the composite MoE block
fixture matches at 2e-4 absolute (cosine 1.000000), because that one chains six matmuls per
expert plus two latent projections and the reference accumulates each dot product in double
while torch sums in float32 - see test_moe_block below and Bulgular.md section 15.6. They
are the only ground truth available for this architecture short of running the 1.45 TB
checkpoint, and they catch the kind of mistake that still produces plausible numbers - an
activation with the wrong tail, a decay indexed per channel instead of per head.

Fixtures absent, fixtures wrong: the difference matters, so the two cases behave differently.

  * LAZYLORA_REF_FIXTURES unset and no sibling kimi-k3-in-c checkout -> the module SKIPS,
    with a message naming the directory it probed and the clone command that fills it.
  * LAZYLORA_REF_FIXTURES set, or a directory present at either location, but the eight
    fixture files are not all there -> the module RAISES at import. A pointer that points
    at nothing is a mistake in the invocation, not a missing optional dependency, and it
    used to hide behind "OK (skipped=8)", which reads exactly like a pass.

So: `OK` means the ops were checked, `OK (skipped=8)` means no fixtures were found and
nothing was checked, and an error at import means the path you gave is wrong.
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
from lazy_lora.core.config import DEFAULT_REFERENCE_REPO, default_reference_fixtures_dir

_EXPLICIT = os.environ.get("LAZYLORA_REF_FIXTURES", "").strip()

FIXTURES = default_reference_fixtures_dir()
if os.environ.get("LAZYLORA_REF_FIXTURES") is not None and not _EXPLICIT:
    # Set but empty: config._env_path resolves "" to the current working directory, which
    # would produce a bewildering "your fixtures directory is the repository root" error.
    # An empty variable means "I did not point anywhere", so fall back to the default path.
    FIXTURES = os.path.abspath(os.path.join(os.path.normpath(DEFAULT_REFERENCE_REPO),
                                            "tests", "fixtures", "ops"))

ABS_TOL = 1e-5
REL_TOL = 1e-4

# The eight fixtures this module checks; one JSON file each, in FIXTURES.
REQUIRED_FIXTURES = ("situ_glu", "rmsnorm", "shortconv", "kda_decay",
                     "router", "attnres", "mla", "moe")

HOW_TO_GET = (
    "These fixtures live in the reference C implementation, not in this repository:\n"
    "    git clone https://github.com/FareedKhan-dev/kimi-k3-in-c ../kimi-k3-in-c\n"
    "Leave the clone beside this repository, or point at it explicitly:\n"
    "    export LAZYLORA_REF_FIXTURES=/path/to/kimi-k3-in-c/tests/fixtures/ops"
)

_TORCH_HINT = ("torch is not installed. "
               "pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.3'")


def _missing_fixture_files(directory):
    return [n for n in REQUIRED_FIXTURES
            if not os.path.isfile(os.path.join(directory, f"{n}.json"))]


def _resolve_fixtures():
    """
    Return a skip reason, or None when the fixtures are all present.

    Raises RuntimeError when the fixtures were asked for and are not usable: a skip in that
    case would report success for a run that checked nothing.
    """
    if not os.path.isdir(FIXTURES):
        if _EXPLICIT:
            raise RuntimeError(
                f"LAZYLORA_REF_FIXTURES={_EXPLICIT!r} is not a directory "
                f"(resolved to {FIXTURES}). Refusing to skip: you asked for the reference "
                f"fixtures, so not finding them is a failure and not a missing extra.\n"
                + HOW_TO_GET
            )
        return (f"reference op fixtures not found at {FIXTURES}; nothing was checked "
                f"against the C implementation.\n" + HOW_TO_GET)

    missing = _missing_fixture_files(FIXTURES)
    if missing:
        raise RuntimeError(
            f"{FIXTURES} exists but {len(missing)} of the {len(REQUIRED_FIXTURES)} "
            f"fixture files are missing: {', '.join(n + '.json' for n in missing)}. "
            f"That is a wrong or partial checkout, not an absent one, so this module fails "
            f"instead of skipping.\n" + HOW_TO_GET
        )
    return None


SKIP_REASON = _resolve_fixtures()


def load_fixture(name):
    path = os.path.join(FIXTURES, f"{name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:                     # disappeared after the import check
        raise AssertionError(f"fixture {path} is missing.\n{HOW_TO_GET}") from exc
    except json.JSONDecodeError as exc:
        raise AssertionError(f"fixture {path} is not valid JSON: {exc}") from exc


def as_array(entry):
    """Fixture tensors are {"shape": [...], "data": [...]} in row-major order."""
    return np.asarray(entry["data"], dtype=np.float32).reshape(entry["shape"])


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
@unittest.skipUnless(HAS_TORCH, _TORCH_HINT)
class TestReferenceOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Printed so the reader can see which directory the numbers below came from.
        print(f"\nreference fixtures: {FIXTURES} "
              f"({len(REQUIRED_FIXTURES)} fixtures, from kimi-k3-in-c)")

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
