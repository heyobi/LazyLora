"""
Unit test for Activation Ring Buffer and Streaming Loader.
"""

import os
import shutil
import unittest
import numpy as np
from lazy_lora.streaming.activation_ring_buffer import ActivationRingBuffer
from lazy_lora.core.config import get_default_config


class TestStreamingAndRingBuffer(unittest.TestCase):

    def setUp(self):
        self.test_cache_dir = os.path.join(get_default_config().paths.workspace_dir, "test_activations")
        os.makedirs(self.test_cache_dir, exist_ok=True)
        self.buffer = ActivationRingBuffer(cache_dir=self.test_cache_dir, num_layers=4)

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)

    def test_activation_save_and_load(self):
        layer_idx = 2
        act = np.random.randn(2, 16, 7168).astype(np.float32)

        # Save to disk
        self.buffer.save_activation(layer_idx, act)

        # File should exist on D: drive test directory
        filepath = os.path.join(self.test_cache_dir, f"act_layer_{layer_idx:03d}.bin")
        self.assertTrue(os.path.exists(filepath))

        # Load back
        loaded = self.buffer.load_activation(layer_idx, as_torch=False)
        self.assertIsNotNone(loaded)
        np.testing.assert_allclose(loaded, act, rtol=1e-5, atol=1e-5)

    def test_clean_cache(self):
        for l in range(4):
            act = np.random.randn(1, 4, 128).astype(np.float32)
            self.buffer.save_activation(l, act)

        self.buffer.clean_cache()
        for l in range(4):
            loaded = self.buffer.load_activation(l, as_torch=False)
            self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
