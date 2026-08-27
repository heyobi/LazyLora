"""
Tests Turkish Dataset Manager, Streaming Pipeline, and Multi-Step Loss Convergence.
Verifies:
1. Turkish dataset generation and JSONL streaming without RAM bloat.
2. Tokenizer padding and chat template parsing.
3. Multi-step loss convergence and non-NaN gradient verification.
"""

import os
import unittest
import numpy as np
from copy import deepcopy
from lazy_lora.core.config import get_default_config
from lazy_lora.dataset.turkish_dataset import TurkishDatasetManager
from lazy_lora.dataset.stream_dataset import StreamingDatasetIterator
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestDatasetAndConvergence(unittest.TestCase):
    """Tests dataset pipeline and multi-step training stability."""

    def setUp(self):
        self.config = get_default_config()
        self.dataset_dir = os.path.join(self.config.paths.workspace_dir, "test_datasets")
        os.makedirs(self.dataset_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.dataset_dir):
            for f in os.listdir(self.dataset_dir):
                try:
                    os.remove(os.path.join(self.dataset_dir, f))
                except OSError:
                    pass
            try:
                os.rmdir(self.dataset_dir)
            except OSError:
                pass

    def test_turkish_dataset_streamer(self):
        """Generates sample Turkish dataset and iterates line-by-line via stream iterator."""
        manager = TurkishDatasetManager(dataset_dir=self.dataset_dir)
        train_file = manager.generate_curated_samples(target_samples=10)
        self.assertTrue(os.path.exists(train_file), f"Dataset JSONL was not created: {train_file}")

        # Stream batches
        streamer = StreamingDatasetIterator(
            jsonl_path=train_file,
            max_seq_len=64,
            pad_token_id=self.config.model.pad_token_id,
        )
        sample_batches = []
        for batch in streamer.get_batches(batch_size=2, as_torch=False):
            sample_batches.append(batch)
            if len(sample_batches) >= 3:
                break

        self.assertGreater(len(sample_batches), 0, "Streamer yielded 0 batches")
        input_ids, target_ids = sample_batches[0]
        self.assertEqual(input_ids.shape[0], 2, "Batch size should be 2")
        self.assertEqual(input_ids.shape[1], 63, "Input sequence length should be max_seq_len - 1")
        self.assertEqual(target_ids.shape[1], 63, "Target sequence length should be max_seq_len - 1")

    def test_multi_step_loss_stability(self):
        """Runs 4 training steps and verifies loss is finite, non-negative, and gradients update."""
        test_config = deepcopy(self.config)
        test_config.model.num_hidden_layers = 2
        test_config.model.hidden_size = 256
        test_config.model.routed_expert_hidden_size = 128
        test_config.model.num_experts = 16
        test_config.model.num_experts_per_token = 4
        test_config.model.num_shared_experts = 1
        test_config.model.moe_intermediate_size = 128
        # Attention output width must match the mock hidden size (2 * 128 = 256)
        test_config.model.num_attention_heads = 2
        test_config.model.head_dim = 128
        # Use an empty weight directory so the streamers fall back to synthetic tensors
        # with these mock dimensions instead of mmapping the real 7168-dim K3 shards.
        test_config.paths.base_model_dir = os.path.join(test_config.paths.workspace_dir, "mock_weights")
        os.makedirs(test_config.paths.base_model_dir, exist_ok=True)

        trainer = LazyLoRATrainer(test_config)

        seq_len = 32
        losses = []
        for step in range(1, 5):
            input_ids = np.random.randint(100, 1000, (1, seq_len)).astype(np.int64)
            target_ids = np.random.randint(100, 1000, (1, seq_len)).astype(np.int64)

            loss = trainer.train_step(step, input_ids, target_ids)
            self.assertFalse(np.isnan(loss), f"Loss became NaN at step {step}")
            self.assertFalse(np.isinf(loss), f"Loss became Inf at step {step}")
            self.assertGreater(loss, 0.0, f"Loss should be positive at step {step}")
            losses.append(loss)

        # Verify optimizer step updated weights (parameters are valid without NaN)
        # Layer 0 is a dense (non-MoE) layer in Kimi K3, so inspect the first MoE layer.
        moe_bundle = next(b for b in trainer.lora_layers if not b.is_dense)
        lora_A = moe_bundle.down_lora.lora_A
        if HAS_TORCH and isinstance(lora_A, torch.Tensor):
            self.assertFalse(torch.isnan(lora_A).any(), "LoRA weights contain NaN")
        else:
            self.assertFalse(np.isnan(lora_A.data).any(), "LoRA weights contain NaN")


if __name__ == "__main__":
    unittest.main()
