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

    def assert_matches(self, got, want, label):
        got = np.asarray(got, dtype=np.float32).reshape(-1)
        want = np.asarray(want, dtype=np.float32).reshape(-1)
        self.assertEqual(got.shape, want.shape, f"{label}: shape mismatch")
        diff = np.abs(got - want)
        tol = ABS_TOL + REL_TOL * np.abs(want)
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


if __name__ == "__main__":
    unittest.main()
