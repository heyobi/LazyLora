#!/usr/bin/env bash
# run_mock_tests.sh - the pre-training mock suite: nine unittest modules that exercise the
# engine without the 1.45 TB checkpoint. Run it standalone (`bash scripts/run_mock_tests.sh`)
# or let scripts/quickstart.sh run it as one of its steps.
#
# What each step actually needs:
#
#   steps 2, 3                pure computation - router maths and LoRA gradients. Nothing
#                             on disk, nothing in the environment.
#   steps 1, 4, 6, 7, 8, 9    a writable sandbox for the workspace and scratch paths. This
#                             script makes one and deletes it on exit; about 350 MB peak,
#                             nearly all of it step 6's 93 x 3.5 MB activation cycle.
#   step 5                    safetensors shards in LAZYLORA_MODEL_DIR - the real 1.45 TB
#                             checkpoint, or the tiny one quickstart.sh generates. It skips
#                             by itself when there are none, which is the expected result
#                             standalone. A skip there is not a failure.
#
# No step here uses the kimi-k3-in-c op fixtures. Those are a separate module:
#   python3 -m unittest lazy_lora.tests.test_reference_ops
# which skips when the fixtures are absent and fails when LAZYLORA_REF_FIXTURES points at
# something that is not them.
#
# The suite is deliberately NOT `set -e`: a failing step must not hide the eight that come
# after it. Every step runs, each is reported pass / skip / fail, and the script exits
# non-zero only when a step genuinely failed. Skips never make it exit non-zero.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${LAZYLORA_PYTHON:-python3}"

if ! command -v "$PYTHON_EXEC" >/dev/null 2>&1 && [ ! -x "$PYTHON_EXEC" ]; then
    echo "no python interpreter at '$PYTHON_EXEC'. Install Python 3.10+ or set LAZYLORA_PYTHON." >&2
    exit 2
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
# The mock suite is the only place synthetic stand-in tensors are legitimate.
export LAZYLORA_ALLOW_SYNTHETIC=1

# --------------------------------------------------------------------- sandbox paths
# Unset paths default to /mnt/disk2tb and /mnt/nvme, which exist on the author's machine
# and nowhere else; several steps then die on the first mkdir. Give them a sandbox instead,
# and keep whatever the caller (quickstart.sh, or a reader with a big scratch disk) set.
SANDBOX=""
if [ -z "${LAZYLORA_WORKSPACE_DIR:-}" ] || [ -z "${LAZYLORA_FAST_SCRATCH_DIR:-}" ] \
   || [ -z "${LAZYLORA_MODEL_DIR:-}" ]; then
    SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/lazylora-mocktests-XXXXXX")" || {
        echo "could not create a temporary directory under ${TMPDIR:-/tmp}" >&2; exit 2; }
fi
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lazylora-mocklogs-XXXXXX")" || exit 2
cleanup() {
    rm -rf "$LOG_DIR"
    [ -n "$SANDBOX" ] && rm -rf "$SANDBOX"
    return 0
}
trap cleanup EXIT

: "${LAZYLORA_WORKSPACE_DIR:=$SANDBOX/workspace}"
: "${LAZYLORA_FAST_SCRATCH_DIR:=$SANDBOX/scratch}"
: "${LAZYLORA_MODEL_DIR:=$SANDBOX/model}"
export LAZYLORA_WORKSPACE_DIR LAZYLORA_FAST_SCRATCH_DIR LAZYLORA_MODEL_DIR
mkdir -p "$LAZYLORA_WORKSPACE_DIR" "$LAZYLORA_FAST_SCRATCH_DIR" "$LAZYLORA_MODEL_DIR" || exit 2

echo "============================================================"
echo "          RUNNING LAZYLORA PRE-TRAINING TEST SUITE          "
echo "============================================================"
echo "  python     : $PYTHON_EXEC"
echo "  repository : $REPO_DIR"
echo "  workspace  : $LAZYLORA_WORKSPACE_DIR"
echo "  scratch    : $LAZYLORA_FAST_SCRATCH_DIR"
echo "  model dir  : $LAZYLORA_MODEL_DIR"
if [ -n "$SANDBOX" ]; then
    echo "  (sandbox created for this run and deleted on exit; needs ~350 MB free."
    echo "   Set LAZYLORA_WORKSPACE_DIR / LAZYLORA_FAST_SCRATCH_DIR to place it yourself.)"
fi

# ------------------------------------------------------------------------ step runner
PASS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
SUMMARY=()

run_step() {
    local num="$1" module="$2" desc="$3" note="$4"
    local log="$LOG_DIR/$num-$module.log" rc ran skipped
    echo
    echo "[$num/9] $desc"
    [ -n "$note" ] && echo "      ($note)"
    "$PYTHON_EXEC" -m unittest "lazy_lora.tests.$module" 2>&1 | tee "$log"
    rc="${PIPESTATUS[0]}"

    ran="$(grep -Eo '^Ran [0-9]+ test' "$log" | grep -Eo '[0-9]+' | tail -1)"
    skipped="$(grep -Eo 'skipped=[0-9]+' "$log" | grep -Eo '[0-9]+' | tail -1)"
    [ -z "$ran" ] && ran=0
    [ -z "$skipped" ] && skipped=0

    if [ "$rc" -ne 0 ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        SUMMARY+=("  [$num/9] FAIL  $module")
    elif [ "$ran" -eq 0 ] || { [ "$skipped" -gt 0 ] && [ "$skipped" -ge "$ran" ]; }; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        SUMMARY+=("  [$num/9] SKIP  $module ($skipped of $ran tests skipped)")
    elif [ "$skipped" -gt 0 ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        SUMMARY+=("  [$num/9] PASS  $module ($((ran - skipped)) of $ran ran, $skipped skipped)")
    else
        PASS_COUNT=$((PASS_COUNT + 1))
        SUMMARY+=("  [$num/9] PASS  $module ($ran tests)")
    fi
}

run_step 1 test_hardware_profiler \
    "Hardware profiler & storage isolation" \
    "needs only the sandbox paths above; asserts the system volume is never marked safe for model storage"
run_step 2 test_moe_routing \
    "Kimi K3 MoE router & top-16 gating maths" \
    "pure computation, no files"
run_step 3 test_lora_gradient \
    "LoRA linear layer & analytical gradients" \
    "pure computation, no files"
run_step 4 test_streaming_loader \
    "Activation ring buffer & disk streaming" \
    "writes into the sandbox"
run_step 5 test_real_safetensors_headers \
    "Real downloaded Kimi K3 safetensors shards" \
    "needs safetensors shards in LAZYLORA_MODEL_DIR (the real checkpoint, or quickstart.sh's tiny one); skips when there are none, which is normal and not a failure"
run_step 6 test_ring_buffer_stress \
    "93-layer activation serialisation stress" \
    "writes and re-reads about 325 MB in the sandbox"
run_step 7 test_checkpoint_manager \
    "LoRA checkpoint serialisation & reloading" \
    "writes into the sandbox"
run_step 8 test_dataset_and_loss_convergence \
    "Turkish dataset streaming & loss stability" \
    "writes into the sandbox; synthetic weights, so the loss curve proves the loop and nothing about Kimi K3"
run_step 9 test_synthetic_lazy_train \
    "End-to-end synthetic mock training & live dashboard" \
    "writes into the sandbox; LAZYLORA_ALLOW_SYNTHETIC=1 is set for this suite only"

echo
echo "============================================================"
printf '%s\n' "${SUMMARY[@]}"
echo "------------------------------------------------------------"
echo "  $PASS_COUNT passed, $SKIP_COUNT skipped, $FAIL_COUNT failed, of 9 steps"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "  RESULT: FAILED - see the step output above."
    echo "============================================================"
    exit 1
fi
if [ "$SKIP_COUNT" -gt 0 ]; then
    echo "  RESULT: passed, with $SKIP_COUNT step(s) skipped for want of data they need."
else
    echo "  RESULT: passed."
fi
echo "  These steps exercise the engine on synthetic tensors. They say nothing about"
echo "  Kimi K3 itself; scripts/quickstart.sh and test_reference_ops do that."
echo "============================================================"
