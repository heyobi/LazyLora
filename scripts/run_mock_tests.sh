#!/usr/bin/env bash
# run_mock_tests.sh - Run comprehensive pre-training test suite for LazyLoRA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${LAZYLORA_PYTHON:-/home/ibox/venvs/lazylora/bin/python}"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
# The mock suite is the only place synthetic stand-in tensors are legitimate.
export LAZYLORA_ALLOW_SYNTHETIC=1

echo "============================================================"
echo "          RUNNING LAZYLORA PRE-TRAINING TEST SUITE         "
echo "============================================================"

echo "[1/9] Testing Hardware Profiler & Storage Isolation..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_hardware_profiler

echo "[2/9] Testing Kimi K3 MoE Router & Top-16 Gating Math..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_moe_routing

echo "[3/9] Testing LoRA Linear Layer & Analytical Gradients..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_lora_gradient

echo "[4/9] Testing Activation Ring Buffer & Disk Streaming..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_streaming_loader

echo "[5/9] Testing Real Downloaded Kimi K3 Safetensors Shards..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_real_safetensors_headers

echo "[6/9] Stress Testing 93-Layer Activation Serialization..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_ring_buffer_stress

echo "[7/9] Testing LoRA Checkpoint Serialization & Reloading..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_checkpoint_manager

echo "[8/9] Testing Turkish Dataset Streaming & Loss Stability..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_dataset_and_loss_convergence

echo "[9/9] Testing End-to-End Synthetic Mock Training & Live Dashboard..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_synthetic_lazy_train

echo "============================================================"
echo "    ALL 9 PRE-TRAINING VALIDATION TEST SUITES PASSED! ✅    "
echo "============================================================"
