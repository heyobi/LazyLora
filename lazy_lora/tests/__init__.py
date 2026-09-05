"""
LazyLoRA Pre-Training Test Suite & Verification Harness.

The mock tests run the engine without the 1.5 TB checkpoint, so they are the one place
where synthetic stand-in tensors are legitimate. Everywhere else a missing tensor is an
error (see lazy_lora.core.config.synthetic_allowed).
"""
import os

os.environ.setdefault("LAZYLORA_ALLOW_SYNTHETIC", "1")
