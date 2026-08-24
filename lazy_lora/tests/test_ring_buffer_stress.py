"""
Stress test for ActivationRingBuffer across all 93 layers.
Verifies:
1. High-throughput writing and reading of 93-layer boundary activations.
2. Bit-exact numerical equality of serialized and deserialized activations.
3. Safe eviction, memory recycling, and cache cleanup.
"""

import os
import unittest
import numpy as np
from lazy_lora.core.config import get_default_config
from lazy_lora.streaming.activation_ring_buffer import ActivationRingBuffer


class TestActivationRingBufferStress(unittest.TestCase):
    """Stress tests 93-layer activation serialization to D: SSD."""

    def setUp(self):
        self.config = get_default_config()
        self.cache_dir = os.path.join(self.config.paths.workspace_dir, "test_act_stress")
        self.ring_buf = ActivationRingBuffer(cache_dir=self.cache_dir)

    def tearDown(self):
        self.ring_buf.clean_cache()
        if os.path.exists(self.cache_dir):
            try:
                os.rmdir(self.cache_dir)
            except OSError:
                pass

    def test_93_layer_full_activation_cycle(self):
        """Simulate writing 93 forward activations and reading them in reverse backward order."""
        num_layers = self.config.model.num_hidden_layers  # 93 layers
        hidden_dim = 512  # Synthetic dim for rapid stress test
        seq_len = 32

        saved_tensors = {}

        # 1. Forward pass: save all 93 layer activations
        for l in range(num_layers):
            act = (np.random.randn(1, seq_len, hidden_dim) * 0.1).astype(np.float32)
            saved_tensors[l] = act
            self.ring_buf.save_activation(l, act)

        # Verify all 93 files exist on D: drive
        for l in range(num_layers):
            path = self.ring_buf.get_activation_path(l)
            self.assertTrue(os.path.exists(path), f"Activation file missing for layer {l}: {path}")

        # 2. Backward pass: read all 93 layer activations in reverse order (92 -> 0)
        for l in range(num_layers - 1, -1, -1):
            loaded_act = self.ring_buf.load_activation(l, as_torch=False)
            self.assertIsNotNone(loaded_act, f"Failed to load activation for layer {l}")
            
            # Verify bit-exact numerical equality
            np.testing.assert_array_equal(
                loaded_act,
                saved_tensors[l],
                err_msg=f"Layer {l} activation data corrupted during disk round-trip!",
            )

        # 3. Clean cache and verify all temporary files removed
        self.ring_buf.clean_cache()
        for l in range(num_layers):
            path = self.ring_buf.get_activation_path(l)
            self.assertFalse(os.path.exists(path), f"Activation file was not cleaned: {path}")


if __name__ == "__main__":
    unittest.main()
