"""
Tests real downloaded Kimi K3 safetensors shards on D: drive.
Verifies:
1. Safetensors 8-byte header size unpack.
2. JSON metadata and tensor offset table extraction.
3. Memory-mapped zero-copy slice reading of real tensor buffers.
"""

import os
import unittest
from pathlib import Path
from lazy_lora.core.config import get_default_config
from lazy_lora.streaming.mmap_loader import SafetensorsIndex, MmapTensorStreamer


class TestRealKimiK3Shards(unittest.TestCase):
    """Inspects and validates real downloaded Kimi K3 weights on D: drive."""

    def setUp(self):
        self.config = get_default_config()
        self.model_dir = self.config.paths.base_model_dir

    def test_real_shard_headers_and_tensors(self):
        """Validates that real downloaded shards can be indexed and sliced via mmap."""
        if not os.path.exists(self.model_dir):
            self.skipTest(f"Model directory {self.model_dir} not found")

        shard_files = sorted(list(Path(self.model_dir).glob("model-*.safetensors")))
        if not shard_files:
            self.skipTest(f"No safetensors shards found in {self.model_dir}")

        print(f"\n[INFO] Found {len(shard_files)} downloaded Kimi K3 shards in {self.model_dir}")

        # Test SafetensorsIndex on downloaded shards
        index = SafetensorsIndex(self.model_dir)
        total_indexed_tensors = len(index.tensor_locations)
        print(f"[INFO] Successfully indexed {total_indexed_tensors} real tensors across {len(shard_files)} shards.")

        self.assertGreater(total_indexed_tensors, 0, "Should index at least one tensor from downloaded shards")

        # Inspect first 5 tensor metadata
        sample_tensor_names = list(index.tensor_locations.keys())[:5]
        for name in sample_tensor_names:
            shard_path, start, end, shape, dtype_str = index.tensor_locations[name]
            size_bytes = end - start
            print(f"  - Real Tensor: {name} | Shape: {shape} | Dtype: {dtype_str} | Bytes: {size_bytes:,}")
            self.assertIn(dtype_str, ["F16", "BF16", "F32", "U8", "I8", "I32"], f"Unexpected dtype {dtype_str}")

        # Test zero-copy mmap tensor extraction on the first tensor
        streamer = MmapTensorStreamer(self.model_dir)
        first_tensor_name = sample_tensor_names[0]
        tensor = streamer.load_tensor(first_tensor_name, as_torch=False)
        self.assertIsNotNone(tensor, f"Failed to load real tensor {first_tensor_name}")
        expected_shape = index.tensor_locations[first_tensor_name][3]
        self.assertEqual(list(tensor.shape), expected_shape)
        print(f"[INFO] Successfully loaded real tensor '{first_tensor_name}' with shape {tensor.shape} via mmap (Zero-RAM copy).")

        streamer.close()


if __name__ == "__main__":
    unittest.main()
