"""
Unit test for LoRA Linear Layer and Analytical Gradients.
"""

import unittest
import numpy as np
from lazy_lora.core.lora_layer import LazyLoRALinear


class TestLazyLoRALinear(unittest.TestCase):

    def test_lora_forward_dimensions(self):
        in_dim = 7168
        out_dim = 3584
        r = 16
        alpha = 32

        lora_mod = LazyLoRALinear(in_dim, out_dim, r=r, lora_alpha=alpha)
        
        x = np.random.randn(2, 4, in_dim).astype(np.float32)
        base_w = (np.random.randn(out_dim, in_dim) * 0.02).astype(np.float32)

        # Forward
        out = lora_mod.forward_with_base(x, base_w)
        self.assertEqual(out.shape, (2, 4, out_dim))

    def test_lora_analytical_gradient(self):
        in_dim = 128
        out_dim = 64
        r = 8
        alpha = 16

        lora_mod = LazyLoRALinear(in_dim, out_dim, r=r, lora_alpha=alpha)
        
        x = np.random.randn(4, in_dim).astype(np.float32)
        dy = np.random.randn(4, out_dim).astype(np.float32)
        base_w = (np.random.randn(out_dim, in_dim) * 0.02).astype(np.float32)

        grad_A, grad_B, grad_x = lora_mod.compute_lora_gradients(dy, x, base_w)

        self.assertEqual(grad_A.shape, (r, in_dim))
        self.assertEqual(grad_B.shape, (out_dim, r))
        self.assertEqual(grad_x.shape, (4, in_dim))

        # Numerical gradient verification for grad_B
        eps = 1e-4
        scaling = alpha / r
        
        # Perturb B[0, 0]
        if hasattr(lora_mod.lora_B, "data"):
            orig_val = float(lora_mod.lora_B.data[0, 0].item() if hasattr(lora_mod.lora_B.data[0, 0], "item") else lora_mod.lora_B.data[0, 0])
            lora_mod.lora_B.data[0, 0] = orig_val + eps
            out_plus = lora_mod.forward_with_base(x, base_w)
            loss_plus = float(np.sum(out_plus * dy))

            lora_mod.lora_B.data[0, 0] = orig_val - eps
            out_minus = lora_mod.forward_with_base(x, base_w)
            loss_minus = float(np.sum(out_minus * dy))

            lora_mod.lora_B.data[0, 0] = orig_val  # restore
        else:
            orig_val = lora_mod.lora_B[0, 0]
            lora_mod.lora_B[0, 0] = orig_val + eps
            out_plus = lora_mod.forward_with_base(x, base_w)
            loss_plus = float(np.sum(out_plus * dy))

            lora_mod.lora_B[0, 0] = orig_val - eps
            out_minus = lora_mod.forward_with_base(x, base_w)
            loss_minus = float(np.sum(out_minus * dy))

            lora_mod.lora_B[0, 0] = orig_val  # restore
        num_grad_B00 = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad_B[0, 0], num_grad_B00, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
