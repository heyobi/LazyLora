#!/usr/bin/env bash
# One-screen status of the run. Everything it needs is discovered or overridable:
#
#   LAZYLORA_REPO    this checkout            (default: the parent of this script)
#   LAZYLORA_PYTHON  the interpreter to use   (default: the first one found, see below)
#
# The volumes it reports come from lazy_lora/core/config.py, i.e. from
# LAZYLORA_MODEL_DIR / LAZYLORA_WORKSPACE_DIR / LAZYLORA_FAST_SCRATCH_DIR.
set -u

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
[ -n "$PY" ] || { echo "status.sh: no python found. Set LAZYLORA_PYTHON." >&2; exit 1; }

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# [trunk] lines are the index overlay announcing itself on import; they are not status.
exec "$PY" "$REPO/scripts/status.py" 2>&1 | grep -v "^\[trunk\]"
