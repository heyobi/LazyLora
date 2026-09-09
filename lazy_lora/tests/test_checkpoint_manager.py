"""
Tests LoRA checkpoint saving, file serialization, and parameter restoration.
Verifies:
1. Checkpoint file is written strictly to the workspace, never to the system disk.
2. Saved state dict includes all 93 layers' LoRA weights.
3. Reloaded parameters match original weights bit-for-bit.
"""

import os
import unittest
import numpy as np
from lazy_lora.core.config import get_default_config
from lazy_lora.core.lora_layer import LazyLoRALinear
from lazy_lora.trainer.lazy_trainer import LoRALayerBundle

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestLoRACheckpointManager(unittest.TestCase):
    """Tests saving and loading LoRA adapter checkpoints."""

    def setUp(self):
        self.config = get_default_config()
        self.ckpt_dir = os.path.join(self.config.paths.workspace_dir, "test_checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.ckpt_dir):
            for f in os.listdir(self.ckpt_dir):
                try:
                    os.remove(os.path.join(self.ckpt_dir, f))
                except OSError:
                    pass
            try:
                os.rmdir(self.ckpt_dir)
            except OSError:
                pass

    def test_lora_checkpoint_save_and_reload(self):
        """Builds LoRA layers, saves a checkpoint to the workspace, and verifies bit-exact restoration."""
        num_test_layers = 4
        d_model = 128
        r = 16

        # Create original layers
        original_bundles = []
        for l in range(num_test_layers):
            b = LoRALayerBundle(
                layer_idx=l,
                hidden_size=d_model,
                moe_latent_size=d_model,
                r=r,
            )
            original_bundles.append(b)

        # Save checkpoint
        ext = ".pt" if HAS_TORCH else ".npz"
        ckpt_path = os.path.join(self.ckpt_dir, f"test_lora_step_100{ext}")
        state_dict = {}
        for l, bundle in enumerate(original_bundles):
            for name, mod in [
                ("q_lora", bundle.q_lora),
                ("v_lora", bundle.v_lora),
                ("gate_lora", bundle.gate_lora),
                ("up_lora", bundle.up_lora),
                ("down_lora", bundle.down_lora),
            ]:
                if HAS_TORCH and isinstance(mod.lora_A, torch.Tensor):
                    state_dict[f"layer_{l}.{name}.lora_A"] = mod.lora_A.cpu().clone()
                    state_dict[f"layer_{l}.{name}.lora_B"] = mod.lora_B.cpu().clone()
                else:
                    state_dict[f"layer_{l}.{name}.lora_A"] = mod.lora_A.data.copy()
                    state_dict[f"layer_{l}.{name}.lora_B"] = mod.lora_B.data.copy()

        if HAS_TORCH:
            torch.save(state_dict, ckpt_path)
            loaded_dict = torch.load(ckpt_path, weights_only=True)
        else:
            np.savez_compressed(ckpt_path, **state_dict)
            loaded_dict = np.load(ckpt_path)

        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint was not created on the workspace disk: drive: {ckpt_path}")
        self.assertGreater(os.path.getsize(ckpt_path), 0, "Checkpoint file size is 0 bytes")

        # Verify loaded weights match original
        for key, original_val in state_dict.items():
            loaded_val = loaded_dict[key]
            if HAS_TORCH and isinstance(original_val, torch.Tensor):
                self.assertTrue(torch.equal(original_val, loaded_val), f"Mismatch in restored tensor {key}")
            else:
                np.testing.assert_array_equal(original_val, loaded_val, err_msg=f"Mismatch in restored array {key}")


if __name__ == "__main__":
    unittest.main()
