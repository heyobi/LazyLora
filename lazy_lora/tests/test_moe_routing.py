"""
Unit test for Kimi K3 MoE Router top-16 selection and gating math.
"""

import unittest
import numpy as np
from lazy_lora.core.moe_router import KimiK3MoERouter


class TestKimiK3MoERouter(unittest.TestCase):

    def test_routing_dimensions_and_sum(self):
        hidden_size = 7168
        num_experts = 896
        top_k = 16
        router = KimiK3MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=True,
        )

        batch_size = 4
        seq_len = 8
        x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

        indices, weights = router.forward(x)

        # Check shape
        self.assertEqual(indices.shape, (batch_size * seq_len, top_k))
        self.assertEqual(weights.shape, (batch_size * seq_len, top_k))

        # Check expert index range [0, 895]
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < num_experts))

        # Check weights sum to 1.0 per token
        sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-4, atol=1e-4)

    def test_active_expert_set(self):
        indices = np.array([
            [10, 20, 30, 40],
            [20, 30, 50, 60],
            [10, 70, 80, 90],
        ])
        active = KimiK3MoERouter.get_active_expert_set(indices)
        self.assertEqual(active, [10, 20, 30, 40, 50, 60, 70, 80, 90])


if __name__ == "__main__":
    unittest.main()
