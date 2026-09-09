#!/usr/bin/env bash
# run_profile.sh - print the LazyLoRA hardware and storage audit for this machine.
#
#   LAZYLORA_REPO    this checkout           (default: the parent of this script)
#   LAZYLORA_PYTHON  the interpreter to use  (default: the first one found, see below)
#
# The volumes it audits are whatever LAZYLORA_MODEL_DIR / LAZYLORA_WORKSPACE_DIR /
# LAZYLORA_FAST_SCRATCH_DIR point at; see lazy_lora/core/config.py. The audit itself
# needs no torch and no checkpoint - set LAZYLORA_PROFILE_TORCH=1 only if you also want
# the GPU's CUDA core count, which costs a torch import and a CUDA context.
set -euo pipefail

REPO="${LAZYLORA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PY="${LAZYLORA_PYTHON:-}"
# An activated virtualenv is what a stranger most likely has, so it comes first; then this
# checkout's own .venv, then the author's, then whatever python3 is on PATH.
if [ -z "$PY" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
fi
if [ -z "$PY" ]; then
    for cand in "$REPO/.venv/bin/python" "$HOME/venvs/lazylora/bin/python"; do
        [ -x "$cand" ] && { PY="$cand"; break; }
    done
fi
[ -n "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "run_profile.sh: no python found. Set LAZYLORA_PYTHON." >&2; exit 1; }

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m lazy_lora.profiler.hardware
