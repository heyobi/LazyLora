#!/usr/bin/env bash
# run_profile.sh - Run LazyLoRA Hardware & Storage Audit in WSL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${LAZYLORA_PYTHON:-/home/ibox/venvs/lazylora/bin/python}"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
"$PYTHON_EXEC" -m lazy_lora.profiler.hardware
