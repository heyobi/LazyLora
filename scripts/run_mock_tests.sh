#!/usr/bin/env bash
# run_mock_tests.sh - Run comprehensive pre-training test suite for LazyLoRA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="/mnt/d/hamza/LazyLora_Workspace/venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo "          RUNNING LAZYLORA PRE-TRAINING TEST SUITE         "
echo "============================================================"

echo "[1/5] Testing Hardware Profiler & Storage Isolation..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_hardware_profiler

echo "[2/5] Testing Kimi K3 MoE Router & Top-16 Gating Math..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_moe_routing

echo "[3/5] Testing LoRA Linear Layer & Analytical Gradients..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_lora_gradient

echo "[4/5] Testing Activation Ring Buffer & Disk Streaming..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_streaming_loader

echo "[5/5] Testing End-to-End Synthetic Mock Training..."
"$PYTHON_EXEC" -m unittest lazy_lora.tests.test_synthetic_lazy_train

echo "============================================================"
echo "       ALL PRE-TRAINING VALIDATION TESTS PASSED! ✅        "
echo "============================================================"
