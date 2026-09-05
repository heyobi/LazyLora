#!/usr/bin/env bash
# train_lazy_lora.sh - Production LazyLoRA Trainer for Kimi K3
# IMPORTANT: Execute only when model checkpoint download is 100% complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${LAZYLORA_PYTHON:-/home/ibox/venvs/lazylora/bin/python}"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
CACHE_DIR="$("$PYTHON_EXEC" -c "from lazy_lora.core.config import default_cache_dir; print(default_cache_dir())")"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export TMPDIR="${CACHE_DIR}"
export PYTHONUNBUFFERED=1

echo "Checking model weight integrity before starting training..."
# Every shard must exist, be complete and carry the tensors the index promises. A
# missing or empty shard used to be counted as present and silently replaced by random
# weights at run time.
if ! "$PYTHON_EXEC" scripts/check_shards.py; then
    echo "=========================================================================="
    echo " [ABORT] The checkpoint is incomplete or damaged; see the report above."
    echo "         Training on it would compute nonsense. Fix the shards first."
    echo "=========================================================================="
    exit 1
fi

echo "Starting LazyLoRA Out-of-Core MoE Training..."
"$PYTHON_EXEC" -m lazy_lora.trainer.lazy_trainer "$@"
