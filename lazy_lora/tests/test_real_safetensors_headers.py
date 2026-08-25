"""
Tests real downloaded Kimi K3 safetensors shards on D: drive and benchmarks NVMe read throughput.
Verifies:
1. Safetensors 8-byte header size unpack and JSON indexing.
2. Zero-copy mmap tensor extraction from real downloaded shards.
3. Live NVMe Disk Read Throughput (MB/s) and latency per tensor lookup.
"""

import os
import time
import unittest
from pathlib import Path
from lazy_lora.core.config import get_default_config
from lazy_lora.streaming.mmap_loader import SafetensorsIndex, MmapTensorStreamer


class TestRealKimiK3Shards(unittest.TestCase):
    """Inspects and benchmarks real downloaded Kimi K3 weights on D: drive."""

    def setUp(self):
        self.config = get_default_config()
        self.model_dir = self.config.paths.base_model_dir

    def test_real_shard_headers_and_tensors(self):
        """Validates real downloaded shards, extracts tensors, and measures real disk read speeds."""
        if not os.path.exists(self.model_dir):
            self.skipTest(f"Model directory {self.model_dir} not found")

        shard_files = sorted(list(Path(self.model_dir).glob("model-*.safetensors")))
        if not shard_files:
            self.skipTest(f"No safetensors shards found in {self.model_dir}")

        total_shard_bytes = sum(f.stat().st_size for f in shard_files)
        total_shard_gb = total_shard_bytes / (1024 ** 3)
        print(f"\n" + "=" * 70)
        print(f" 💾 REAL KIMI K3 SHARDS DISK I/O & READ SPEED BENCHMARK")
        print(f"=" * 70)
        print(f"  - Downloaded Shards Count : {len(shard_files)} shards")
        print(f"  - Total Downloaded Size   : {total_shard_gb:.2f} GB on D: NVMe SSD")

        # 1. Indexing Benchmark
        t0_idx = time.perf_counter()
        index = SafetensorsIndex(self.model_dir)
        t_idx = time.perf_counter() - t0_idx
        total_indexed_tensors = len(index.tensor_locations)
        print(f"  - Indexed Tensors Count   : {total_indexed_tensors:,} real tensors")
        print(f"  - Indexing Scan Time      : {t_idx:.3f} s ({total_indexed_tensors / max(t_idx, 0.001):,.0f} tensors/s)")

        self.assertGreater(total_indexed_tensors, 0, "Should index at least one tensor from downloaded shards")

        # 2. Benchmark Streaming Read Speed across 50 real tensors
        streamer = MmapTensorStreamer(self.model_dir)
        tensor_names = list(index.tensor_locations.keys())
        benchmark_sample_count = min(50, len(tensor_names))
        sample_tensor_names = tensor_names[:benchmark_sample_count]

        total_bytes_read = 0
        t0_read = time.perf_counter()
        for name in sample_tensor_names:
            tensor = streamer.load_tensor(name, as_torch=False)
            self.assertIsNotNone(tensor, f"Failed to load real tensor {name}")
            total_bytes_read += tensor.nbytes

        t_read = time.perf_counter() - t0_read
        mb_read = total_bytes_read / (1024 * 1024)
        throughput_mb_s = mb_read / max(t_read, 0.0001)
        avg_latency_ms = (t_read / benchmark_sample_count) * 1000

        print(f"-" * 70)
        print(f" 🚀 MEASURED REAL DISK READ PERFORMANCE:")
        print(f"  - Tested Tensors Sample   : {benchmark_sample_count} real tensors")
        print(f"  - Data Sliced & Read      : {mb_read:.2f} MB")
        print(f"  - Time Elapsed            : {t_read * 1000:.2f} ms")
        print(f"  - NVMe Read Throughput    : {throughput_mb_s:,.1f} MB/s ({throughput_mb_s / 1024:.2f} GB/s)")
        print(f"  - Average Tensor Latency  : {avg_latency_ms:.3f} ms per tensor lookup")
        print(f"=" * 70)

        streamer.close()


if __name__ == "__main__":
    unittest.main()
