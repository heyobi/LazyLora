#!/usr/bin/env bash
export PYTHONPATH=/home/ibox/calisma/LazyLora
exec /home/ibox/venvs/lazylora/bin/python /home/ibox/calisma/LazyLora/scripts/status.py 2>&1 | grep -v "^\[trunk\]"
