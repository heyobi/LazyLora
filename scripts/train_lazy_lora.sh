#!/usr/bin/env bash
# train_lazy_lora.sh - Production LazyLoRA Trainer for Kimi K3
# IMPORTANT: Execute only when model checkpoint download is 100% complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="/mnt/d/hamza/LazyLora_Workspace/venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HOME="/mnt/d/hamza/LazyLora_Workspace/cache/huggingface"
export TORCH_HOME="/mnt/d/hamza/LazyLora_Workspace/cache/torch"
export TMPDIR="/mnt/d/hamza/LazyLora_Workspace/cache"
export PYTHONUNBUFFERED=1

echo "Checking model weight integrity before starting training..."
MODEL_DIR="/mnt/d/hamza/kimi_k3_model_weights"

SHARD_COUNT=$(find "$MODEL_DIR" -name "model-*.safetensors" 2>/dev/null | wc -l)
echo "Found $SHARD_COUNT / 96 shards in $MODEL_DIR"

if [ "$SHARD_COUNT" -lt 96 ]; then
    echo "=========================================================================="
    echo " [NOTICE] Model is currently downloading ($SHARD_COUNT/96 shards present)."
    echo " Full training cannot start until all 96 shards are downloaded."
    echo " You can run the pre-training verification tests anytime:"
    echo "   bash scripts/run_mock_tests.sh"
    echo "=========================================================================="
    exit 0
fi

echo "Starting LazyLoRA Out-of-Core MoE Training..."
"$PYTHON_EXEC" -m lazy_lora.trainer.lazy_trainer "$@"
