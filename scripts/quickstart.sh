#!/usr/bin/env bash
# =====================================================================================
# quickstart.sh - run the whole LazyLoRA engine, end to end, without the 1.56 TB
# checkpoint. About two minutes with --fast, three or four with the full test suite -
# a figure DERIVED FROM THE CODE, not timed (see the banner below).
#
# It generates a tiny (~8.3 MB) but genuinely Kimi-K3-shaped checkpoint and then runs the
# real engine against it: the real streaming loader, the real KDA and MLA attention, the
# real router, the real MXFP4 experts, the real autograd replay, the real AdamW and the
# real checkpoint format. Only the weights are small. Nothing is faked at run time -
# LAZYLORA_ALLOW_SYNTHETIC stays unset for every step that touches the tiny model, so the
# engine's "refuse to substitute a random tensor" gate (lazy_lora/core/config.py:166-182)
# is closed throughout.
#
# -------------------------------------------------------------------------------------
# !! WRITTEN WITHOUT BEING EXECUTED, THEN EXECUTED IN PUBLIC !!
# This script was written by reading the engine while the author's machine was busy with the
# 100-step training run, so it had never been run when it was committed. The first machine to
# run it was a GitHub Actions runner, on 10 September 2026: five of its seven steps passed on
# that first attempt, one skipped for fixtures that are now committed, and step 7 failed - on
# a gradient that turned out to be correct, with a finite-difference step too small for fp32
# to resolve against a tensor of norm 60.8. That episode is written up in Bulgular.md section
# 20, the step rule is fixed, and .github/workflows/quickstart.yml re-runs the whole thing on
# every push. The runtimes and memory figures quoted here and in docs/QUICKSTART.md are still
# worked out from the code paths rather than measured on the author's hardware.
# -------------------------------------------------------------------------------------
#
# Needs: python3 (3.10+) with numpy and torch (>= 2.3, CPU is fine). torch 2.3 is a hard
# floor: the loader reinterprets numpy uint16 as bfloat16 with
# torch.from_numpy(...).view(torch.bfloat16) (streaming/mmap_loader.py:326), and uint16
# from_numpy support arrived in torch 2.3.
#
# Budget, derived from the code and not measured: about 1.5 GB of free RAM and 1 GB of
# temporary disk for a full run (about 500 MB and almost no disk with --fast). Everything
# it writes goes into one mktemp -d directory under $TMPDIR that is deleted on exit
# (--keep to keep it, --dir to place it). It never writes anywhere else - in particular it
# does not write into the clone. If /tmp is a tmpfs on your system, the ~330 MB step 5
# writes is RAM rather than disk; set TMPDIR to a real filesystem or pass --dir.
#
#   bash scripts/quickstart.sh                # everything
#   bash scripts/quickstart.sh --fast         # skip the synthetic-weight suite (the slowest step)
#   bash scripts/quickstart.sh --keep --dir /tmp/lazylora-demo
#
# Tunables (environment): QS_SEQ_LEN (64), QS_STEPS (10), QS_FD_LAYERS (3),
# QS_FD_PROBES (3), QS_SEED (0), LAZYLORA_PYTHON (python3).
# =====================================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LAZYLORA_PYTHON:-python3}"
# Exported, not just assigned: anything this script shells out to must resolve the same
# interpreter rather than fall back to its own default. scripts/run_mock_tests.sh:6, for
# one, otherwise falls back to a hardcoded venv path on the author's machine, so the
# suites could silently run under a different Python than the other steps.
export LAZYLORA_PYTHON="$PY"
KEEP=0
RUN_MOCK=1
DEMO=""

while [ $# -gt 0 ]; do
    case "$1" in
        --keep) KEEP=1 ;;
        --fast|--no-mock) RUN_MOCK=0 ;;
        --dir) shift; DEMO="${1:-}"; KEEP=1 ;;
        -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

die() { printf '\n\033[31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }
hr() { printf '%s\n' "-------------------------------------------------------------------------------"; }

RESULTS=""
FAILED=0
note() {                                   # note <step> <PASS|FAIL|SKIP> <meaning>
    RESULTS="${RESULTS}$(printf '  %-34s %-4s  %s' "$1" "$2" "$3")
"
    [ "$2" = "FAIL" ] && FAILED=1
    return 0
}

# ------------------------------------------------------------------ 0. dependencies
hr
echo "LazyLoRA quickstart - the whole engine, no checkpoint required"
hr

command -v "$PY" >/dev/null 2>&1 || die "no python interpreter at '$PY'. Install Python 3.10+ or set LAZYLORA_PYTHON."

DEPS="$("$PY" - <<'PY' 2>&1
import sys
missing = []
try:
    import numpy
except Exception as exc:
    missing.append(f"numpy ({exc})")
try:
    import torch
    # mmap_loader.py:326 needs torch.from_numpy on numpy uint16, which is torch >= 2.3.
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 3):
        missing.append(f"torch {torch.__version__} is older than the required 2.3")
except Exception as exc:
    missing.append(f"torch ({exc})")
if missing:
    print("MISSING " + "; ".join(missing))
else:
    print(f"OK python {sys.version.split()[0]}  numpy {numpy.__version__}  torch {torch.__version__}")
PY
)" || die "the interpreter '$PY' could not be run"

case "$DEPS" in
    OK*) echo "  $DEPS" ;;
    *) die "$DEPS
LazyLoRA needs exactly two packages at run time. Install them with:
    $PY -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.3'
    $PY -m pip install 'numpy>=1.24'
or, from a clone, simply:
    $PY -m pip install -e .
Do NOT use requirements.txt for this: it is a full environment freeze." ;;
esac

[ -f "$REPO/lazy_lora/trainer/lazy_trainer.py" ] || die "this does not look like a LazyLoRA checkout: $REPO"

# ------------------------------------------------------------------ 1. the sandbox
if [ -z "$DEMO" ]; then
    DEMO="$(mktemp -d "${TMPDIR:-/tmp}/lazylora-quickstart-XXXXXX")" || die "could not create a temporary directory"
fi
mkdir -p "$DEMO/model" "$DEMO/workspace" "$DEMO/scratch" || die "cannot write into $DEMO"

cleanup() {
    if [ "$KEEP" -eq 1 ]; then
        printf '\nkept: %s\n' "$DEMO"
    else
        rm -rf "$DEMO"
    fi
}
trap cleanup EXIT INT TERM

export LAZYLORA_REPO="$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LAZYLORA_MODEL_DIR="$DEMO/model"
export LAZYLORA_WORKSPACE_DIR="$DEMO/workspace"
export LAZYLORA_FAST_SCRATCH_DIR="$DEMO/scratch"
export LAZYLORA_TRUNK_DIR=""              # no NVMe trunk overlay (mmap_loader.py:147-149)
# The synthetic-tensor gate must be CLOSED for everything that touches the tiny model:
# with a complete checkpoint on disk the engine never needs a substitute, and proving
# that is half the point of this script. Step 5 opens it per process, for itself only.
unset LAZYLORA_ALLOW_SYNTHETIC
unset LAZYLORA_COMPUTE_FP32
unset LAZYLORA_GPU
unset LAZYLORA_PROFILE
unset LAZYLORA_NO_NATIVE

QS_SEED="${QS_SEED:-0}"
QS_SEQ_LEN="${QS_SEQ_LEN:-64}"
QS_STEPS="${QS_STEPS:-10}"
QS_FD_LAYERS="${QS_FD_LAYERS:-3}"
QS_FD_PROBES="${QS_FD_PROBES:-3}"
export QS_SEED QS_SEQ_LEN QS_STEPS QS_FD_LAYERS QS_FD_PROBES

echo "  sandbox        : $DEMO   (deleted on exit; --keep to keep it)"
echo "  model dir      : $LAZYLORA_MODEL_DIR"
echo "  workspace      : $LAZYLORA_WORKSPACE_DIR"
echo "  fast scratch   : $LAZYLORA_FAST_SCRATCH_DIR"
# lazy_lora/tests/__init__.py:10 does os.environ.setdefault("LAZYLORA_ALLOW_SYNTHETIC", "1")
# at import time, so any `-m unittest lazy_lora.tests.*` process runs with the gate open.
# That is harmless where it happens (step 1 loads no weights; step 5 is the suite the flag
# exists for) but the banner must not claim more than is true.
echo "  synthetic gate : CLOSED for every step that touches the tiny model"

# ------------------------------------------------- 2. op fixtures vs the C reference
hr
echo "[1/7] Op-level fixtures against the independent C implementation"
if [ -z "${LAZYLORA_REF_FIXTURES:-}" ]; then
    for cand in "$REPO/tests/fixtures/ops" "$REPO/lazy_lora/tests/fixtures/ops" \
                "$REPO/../kimi-k3-in-c/tests/fixtures/ops"; do
        if [ -d "$cand" ]; then export LAZYLORA_REF_FIXTURES="$cand"; break; fi
    done
fi
if [ -n "${LAZYLORA_REF_FIXTURES:-}" ]; then
    echo "      fixtures: $LAZYLORA_REF_FIXTURES"
    if "$PY" -m unittest lazy_lora.tests.test_reference_ops; then
        note "1 reference ops" PASS "SiTU-GLU, RMSNorm, short conv, KDA decay, router, bank mix, MLA and the whole latent MoE block match kimi-k3-in-c: seven of the eight fixtures at 1e-5 absolute / 1e-4 relative, the MoE block at 2e-4 absolute with cosine 1.000000 (test_reference_ops.py hardcodes abs_tol=2e-4 for it - that is the MXFP4 decode path's own rounding)."
    else
        note "1 reference ops" FAIL "at least one op does NOT match the C reference - the engine is computing something else."
    fi
else
    echo "      not found. These fixtures live in the reference repository:"
    echo "        git clone https://github.com/FareedKhan-dev/kimi-k3-in-c ../kimi-k3-in-c"
    echo "      or set LAZYLORA_REF_FIXTURES to its tests/fixtures/ops directory."
    note "1 reference ops" SKIP "fixtures absent; the only external ground truth in the repo was not checked."
fi

# ------------------------------------------------------------ 3. the tiny checkpoint
hr
echo "[2/7] Generating a tiny Kimi-K3-shaped checkpoint"
if "$PY" "$REPO/scripts/make_tiny_model.py" "$LAZYLORA_MODEL_DIR" --seed "$QS_SEED"; then
    note "2 tiny checkpoint" PASS "real safetensors bytes, real MXFP4 expert blocks; every tensor the loader asks for is on disk."
else
    note "2 tiny checkpoint" FAIL "generation failed."
    printf '%b' "$RESULTS"
    die "nothing downstream can run without the tiny checkpoint."
fi

# --------------------------------------------------------------- 4. shard integrity
hr
echo "[3/7] Checkpoint integrity, with the same script that guards the real run"
if "$PY" "$REPO/scripts/check_shards.py" --model-dir "$LAZYLORA_MODEL_DIR"; then
    note "3 shard integrity" PASS "every shard is present, its size equals header+data, and it holds the tensors the index promises."
else
    note "3 shard integrity" FAIL "check_shards.py rejected the generated checkpoint."
fi

# ------------------------------------------------------------------ 5. forward pass
hr
echo "[4/7] A real forward pass through the real engine"
if "$PY" - <<'PY'
import math, os, sys, time
sys.path.insert(0, os.environ["LAZYLORA_REPO"])
sys.path.insert(0, os.path.join(os.environ["LAZYLORA_REPO"], "scripts"))
assert os.environ.get("LAZYLORA_ALLOW_SYNTHETIC") != "1", "the synthetic gate must be closed here"

import torch
from make_tiny_model import load_tiny_config
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer
from lazy_lora.trainer.loss import compute_cross_entropy_loss

N = int(os.environ["QS_SEQ_LEN"])
cfg, applied = load_tiny_config(os.environ["LAZYLORA_MODEL_DIR"])
cfg.training.max_seq_len = N
print(f"      architecture from config.json: {len(applied)} fields, "
      f"{cfg.model.num_hidden_layers} layers, hidden {cfg.model.hidden_size}, "
      f"{cfg.model.num_experts} experts top-{cfg.model.num_experts_per_token}")

V = cfg.model.vocab_size
seq = (torch.arange(N + 1, dtype=torch.long) * 37 + 11) % (V - 8) + 3
inp, tgt = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)

trainer = LazyLoRATrainer(cfg)
t0 = time.time()
with torch.no_grad():
    h = trainer._embed_tokens(inp)
    trainer._reset_block_residual()
    per_layer = []
    for l in range(cfg.model.num_hidden_layers):
        h, experts = trainer.forward_layer(l, h)
        per_layer.append(len(experts))
    h = trainer._finalize_hidden(h, trainer._block_residual)
    logits = trainer._project_lm_head(h)
    loss, _grad = compute_cross_entropy_loss(logits, tgt, ignore_index=cfg.model.pad_token_id)
dt = time.time() - t0
loss = float(loss)
uniform = math.log(V)
read_mb = trainer.mmap_streamer.bytes_read / 1e6

print(f"      {N} tokens through {cfg.model.num_hidden_layers} layers in {dt:.1f} s, "
      f"{read_mb:.2f} MB streamed off disk")
print(f"      experts touched per layer: {per_layer}  (dense layers show 0)")
print(f"      loss {loss:.4f}   uniform prior ln({V}) = {uniform:.4f}   "
      f"perplexity {math.exp(min(loss, 20)):.1f}")

problems = []
if not math.isfinite(loss):
    problems.append(f"loss is {loss}")
if abs(loss - uniform) > 1.5:
    problems.append(f"loss {loss:.3f} is far from ln(V) = {uniform:.3f}; an untrained model "
                    f"must sit near the uniform prior")
if tuple(logits.shape) != (1, N, V):
    problems.append(f"logits are {tuple(logits.shape)}, expected {(1, N, V)}")
if not torch.isfinite(h).all():
    problems.append("the final hidden state contains NaN or inf")
if read_mb <= 0:
    problems.append("nothing was read from disk, so the weights did not come from the shards")
moe = [c for l, c in enumerate(per_layer) if l >= cfg.model.first_k_dense_replace]
if not all(1 <= c <= cfg.model.num_experts for c in moe):
    problems.append(f"expert counts out of range: {moe}")
trainer.close()
for p in problems:
    print("      PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
then
    note "4 forward pass" PASS "streamed weights, KDA and MLA attention, the residual bank, the router and the MXFP4 experts all ran; the loss sits at the uniform prior, which is where an untrained model belongs."
else
    note "4 forward pass" FAIL "the forward pass did not produce a sane loss."
fi

# ------------------------------------------------- 6. the synthetic-weight mock suite
hr
if [ "$RUN_MOCK" -eq 1 ]; then
    echo "[5/7] The synthetic-weight test suite (ring buffer, router, LoRA gradients, checkpoints)"
    echo "      this one legitimately sets LAZYLORA_ALLOW_SYNTHETIC=1 and writes ~330 MB"

    # test_hardware_profiler describes the author's laptop, not the engine: it asserts that
    # a second non-system volume exists, and on WSL that a /mnt/d path is recommended, which
    # HardwareProfiler only produces when a D: drive is actually mounted
    # (tests/test_hardware_profiler.py:30-31 against profiler/hardware.py). It is reported
    # here but it is deliberately NOT allowed to fail the run - the script's own advice is
    # to ignore it, and the verdict must not contradict the advice.
    echo "      [5a] hardware profile (informational: describes the machine, not the engine)"
    if LAZYLORA_ALLOW_SYNTHETIC=1 "$PY" -m unittest lazy_lora.tests.test_hardware_profiler; then
        note "5a hardware profile" PASS "the storage profiler found a non-system volume and recommended a workspace on it."
    else
        note "5a hardware profile" SKIP "informational only: this test asserts a second non-system volume, and on WSL a mounted D: drive. It describes the author's laptop. Nothing else depends on it, so it does not fail the run."
    fi

    echo "      [5b] the eight suites that are about the engine"
    if LAZYLORA_ALLOW_SYNTHETIC=1 "$PY" -m unittest \
            lazy_lora.tests.test_moe_routing \
            lazy_lora.tests.test_lora_gradient \
            lazy_lora.tests.test_streaming_loader \
            lazy_lora.tests.test_real_safetensors_headers \
            lazy_lora.tests.test_ring_buffer_stress \
            lazy_lora.tests.test_checkpoint_manager \
            lazy_lora.tests.test_dataset_and_loss_convergence \
            lazy_lora.tests.test_synthetic_lazy_train; then
        note "5 mock suite" PASS "the 93-layer activation ring buffer, top-k routing, the analytic LoRA gradient and the checkpoint round-trip all hold."
    else
        note "5 mock suite" FAIL "see the output above. This suite is the older harness that runs on synthetic weights."
    fi
else
    echo "[5/7] synthetic-weight test suite - skipped (--fast)"
    note "5 mock suite" SKIP "not run (--fast)."
fi

# ------------------------------------------- 7. a real training step, saved and reloaded
hr
echo "[6/7] Real training steps: forward, backward, AdamW, checkpoint, reload"
if "$PY" - <<'PY'
import math, os, sys, time
sys.path.insert(0, os.environ["LAZYLORA_REPO"])
sys.path.insert(0, os.path.join(os.environ["LAZYLORA_REPO"], "scripts"))
assert os.environ.get("LAZYLORA_ALLOW_SYNTHETIC") != "1", "the synthetic gate must be closed here"

import torch
from make_tiny_model import load_tiny_config
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer

N = int(os.environ["QS_SEQ_LEN"])
STEPS = int(os.environ["QS_STEPS"])
torch.manual_seed(int(os.environ["QS_SEED"]))

cfg, _ = load_tiny_config(os.environ["LAZYLORA_MODEL_DIR"])
cfg.training.max_seq_len = N
cfg.training.max_steps = STEPS
cfg.training.warmup_steps = 1            # 50 would leave the lr at 2 % for the whole demo
cfg.training.learning_rate = 5e-3
cfg.training.min_learning_rate = 5e-4
cfg.training.logging_steps = 10 ** 6     # the live dashboard belongs to the real run
cfg.lora.r = 8
cfg.lora.lora_alpha = 16

V = cfg.model.vocab_size
seq = (torch.arange(N + 1, dtype=torch.long) * 37 + 11) % (V - 8) + 3
inp, tgt = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)

trainer = LazyLoRATrainer(cfg)
n_params = sum(p.numel() for p in trainer.all_lora_params)
print(f"      {len(trainer.all_lora_params)} LoRA tensors, {n_params:,} trainable parameters "
      f"(rank {cfg.lora.r}); every base weight stays frozen on disk")

t0 = time.time()
losses = []
for step in range(1, STEPS + 1):
    losses.append(trainer.train_step(step=step, input_ids=inp, target_ids=tgt))
dt = time.time() - t0
print(f"      {STEPS} steps on one fixed {N}-token sequence in {dt:.1f} s "
      f"({dt / STEPS:.1f} s/step): {losses[0]:.4f} -> {losses[-1]:.4f}")

moved = max(float(mod.lora_B.detach().abs().max())
            for bundle in trainer.lora_layers for _name, mod in bundle.all_modules())
print(f"      largest |lora_B| after training: {moved:.3e} (it was exactly 0 at init)")

ckpt = trainer.save_lora_checkpoint(step=STEPS, data_cursor=17)
size_mb = os.path.getsize(ckpt) / 1e6
print(f"      checkpoint: {os.path.basename(ckpt)}  {size_mb:.2f} MB")

other = LazyLoRATrainer(cfg)
meta = other.load_checkpoint(ckpt)
same = all(
    torch.equal(m1.lora_A.detach(), m2.lora_A.detach()) and
    torch.equal(m1.lora_B.detach(), m2.lora_B.detach())
    for b1, b2 in zip(trainer.lora_layers, other.lora_layers)
    for (_n1, m1), (_n2, m2) in zip(b1.all_modules(), b2.all_modules())
)
moments = (set(other.optimizer.m_states) == set(trainer.optimizer.m_states) and
           all(torch.equal(trainer.optimizer.m_states[k], other.optimizer.m_states[k]) and
               torch.equal(trainer.optimizer.v_states[k], other.optimizer.v_states[k])
               for k in trainer.optimizer.m_states))

problems = []
if not all(math.isfinite(x) for x in losses):
    problems.append(f"a loss was not finite: {losses}")
if moved == 0.0:
    problems.append("lora_B is still exactly zero: the optimizer did not update the adapter")
if not same:
    problems.append("the reloaded adapter differs from the saved one")
if not moments:
    problems.append("the optimizer moments did not survive the round-trip")
if meta["step"] != STEPS or meta["data_cursor"] != 17:
    problems.append(f"checkpoint metadata came back as {meta}")
if os.path.exists(ckpt + ".tmp"):
    problems.append("a partial checkpoint file was left behind; the write was not atomic")
if losses[-1] >= losses[0]:
    print(f"      NOTE: the loss did not fall ({losses[0]:.4f} -> {losses[-1]:.4f}). On random "
          f"weights this is possible but unexpected; try QS_STEPS=20.")

other.close()
trainer.close()
for p in problems:
    print("      PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
then
    note "6 training step" PASS "the full out-of-core loop ran: activations to disk, autograd replay of every layer, expert streaming twice, AdamW on the adapter, and an atomic checkpoint that reloads bit for bit with its optimizer moments."
else
    note "6 training step" FAIL "see the output above."
fi

# --------------------------------------------------- 8. finite-difference gradient check
hr
echo "[7/7] Finite differences: is the backward pass the gradient of the forward pass?"
if LAZYLORA_COMPUTE_FP32=1 "$PY" - <<'PY'
import os, sys, time
sys.path.insert(0, os.environ["LAZYLORA_REPO"])
sys.path.insert(0, os.path.join(os.environ["LAZYLORA_REPO"], "scripts"))
assert os.environ.get("LAZYLORA_ALLOW_SYNTHETIC") != "1", "the synthetic gate must be closed here"

import torch
from make_tiny_model import load_tiny_config
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer

# The same method as scripts/verify_backward.py, which is what runs against the real
# 1.56 TB checkpoint: perturb one direction, compare <grad, d> with the central
# difference of the loss. fp32 engine, float64 reduction, eps chosen per direction, and
# a retry when a step crosses a top-k routing flip (which makes the difference quotient
# scale as 1/eps instead of converging).
N = int(os.environ["QS_SEQ_LEN"])
LAYERS = [int(x) for x in os.environ["QS_FD_LAYERS"].split(",") if x.strip()]
PROBES = int(os.environ["QS_FD_PROBES"])
SEED = int(os.environ["QS_SEED"])
TOL = 2e-2
REL_STEP = 2e-3          # the finite-difference step, as a fraction of the perturbed tensor's norm

cfg, _ = load_tiny_config(os.environ["LAZYLORA_MODEL_DIR"])
cfg.training.max_seq_len = N
cfg.lora.r = 8
cfg.lora.lora_alpha = 16
trainer = LazyLoRATrainer(cfg)
assert trainer.compute_dtype == torch.float32, "LAZYLORA_COMPUTE_FP32=1 did not take effect"

V = cfg.model.vocab_size
ids = ((torch.arange(min(N, 8), dtype=torch.long) * 37 + 11) % (V - 8) + 3).unsqueeze(0)
worst_overall = 0.0
failures = []

for L in LAYERS:
    torch.manual_seed(SEED)
    with torch.no_grad():
        h = trainer._embed_tokens(ids)
        trainer._reset_block_residual()
        for l in range(L):
            h, _ = trainer.forward_layer(l, h)
    h_in = h.detach().clone()
    bank = None if trainer._block_residual is None else trainer._block_residual.detach().clone()

    bundle = trainer.lora_layers[L]
    with torch.no_grad():
        for _name, mod in bundle.all_modules():
            mod.lora_B.normal_(std=0.02)      # with B = 0 the gradient of A is identically 0
    mods_params = trainer._lora_params_of(bundle)
    params = [p for _m, p in mods_params]
    names = [f"{n}.{ab}" for n, m in bundle.all_modules() for ab in ("A", "B") if m.lora_A is not None]

    def run(h_in_, bank_):
        with torch.no_grad():
            out, _ = trainer._run_layer(L, h_in_, bank_)
            trainer.expert_streamer.evict_layer_experts(L)
        return out

    out0 = run(h_in, bank)
    R = torch.randn_like(out0)
    R64 = R.double()

    def loss_at(h_in_, bank_):
        return float((run(h_in_, bank_).double() * R64).sum())

    for p in params:
        p.grad = None
    with torch.no_grad():
        trainer.act_buffer.save_activation(L, h_in)    # forward_layer(L) would have done this
    t0 = time.time()
    grad_h_in = trainer.backward_layer(L, R)
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    entries = trainer._bank_entries_before(L)
    grad_bank = (torch.stack([trainer._grad_bank[l].reshape(-1, h_in.shape[-1]) for l in entries], dim=1)
                 if entries else None)
    for p in params:
        p.grad = None
    print(f"      layer {L} ({'KDA' if cfg.model.is_kda_layer(L) else 'MLA'}, "
          f"{len(entries)} bank entries): backward in {time.time() - t0:.1f} s")
    print(f"      {'direction':26s} {'analytic':>13s} {'central diff':>13s} {'eps':>9s} {'rel err':>9s}")

    def check(label, analytic, fn, scale=1.0):
        """
        Central difference with the step size measured against the tensor being perturbed,
        and one Richardson extrapolation.

        The step has to be relative, not absolute. The engine computes in fp32, so a loss
        of magnitude F is only known to about 1e-7 * F; a perturbation that moves the loss
        by less than that measures nothing but round-off. The residual bank at layer 3 of
        the tiny model has norm 60.8, and an absolute step of 1.8e-3 along a unit-norm
        direction perturbs it by a relative 3e-5 -- below what fp32 can resolve, which is
        why an earlier version of this check reported a 3.1e-2 "mismatch" on a gradient
        that is in fact correct to four digits (docs/QUICKSTART.md, "the bank direction").
        With eps = 2e-3 * ||tensor|| the same direction agrees to 1.6e-4.

        Richardson combines the steps eps and eps/2 into an estimate whose leading
        truncation term cancels, so the check is accurate at the large step that fp32
        needs.
        """
        global worst_overall
        eps = float(min(0.05 * scale, max(1e-4, REL_STEP * scale)))
        for _ in range(4):
            fd1 = (fn(eps) - fn(-eps)) / (2 * eps)
            fd2 = (fn(eps / 2) - fn(-eps / 2)) / eps
            ratio = fd2 / fd1 if fd1 != 0 else float("inf")
            if 1.7 < ratio < 2.3 and abs(fd2) > 20 * max(abs(analytic), 1e-9):
                eps /= 8                      # the step crossed a routing flip; shorten it
                continue
            break
        fd2 = (4 * fd2 - fd1) / 3             # Richardson: cancels the eps^2 term
        rel = abs(analytic - fd2) / max(abs(analytic), abs(fd2), 1e-12)
        worst_overall = max(worst_overall, rel)
        flag = "   <-- MISMATCH" if rel > TOL else ""
        if flag:
            failures.append(f"layer {L} {label}: analytic {analytic:.6e} vs finite difference {fd2:.6e}")
        print(f"      {label:26s} {analytic:13.6e} {fd2:13.6e} {eps:9.1e} {rel:9.2e}{flag}")

    gen = torch.Generator().manual_seed(SEED + 1)
    order = torch.randperm(len(params), generator=gen)[:PROBES].tolist()
    for i in order:
        d = torch.randn(params[i].shape, generator=gen)
        d = d / d.norm()
        analytic = float((grads[i] * d).sum())

        def fn(eps, p=params[i], d=d):
            with torch.no_grad():
                p.add_(d, alpha=eps)
                try:
                    return loss_at(h_in, bank)
                finally:
                    p.add_(d, alpha=-eps)
        check(f"param {names[i]}", analytic, fn, scale=float(params[i].norm()))

    d = torch.randn_like(h_in)
    d = d / d.norm()
    check("h_in", float((grad_h_in.to(d.dtype) * d).sum()),
          lambda eps, d=d: loss_at(h_in + eps * d, bank), scale=float(h_in.norm()))

    if bank is not None:
        d = torch.randn_like(bank)
        d = d / d.norm()
        check("residual bank", float((grad_bank.to(d.dtype) * d).sum()),
              lambda eps, d=d: loss_at(h_in, bank + eps * d), scale=float(bank.norm()))

trainer.close()
print(f"      worst relative error {worst_overall:.2e} (tolerance {TOL:.0e})")
for f in failures:
    print("      PROBLEM:", f)
sys.exit(1 if failures else 0)
PY
then
    note "7 finite differences" PASS "the analytic gradients of the LoRA tensors, of the layer input and of the residual bank agree with central differences of the forward pass. This is the same check that runs on the real checkpoint."
else
    note "7 finite differences" FAIL "a gradient does not match its finite difference."
fi

# -------------------------------------------------------------- 9. native kernel A/B
hr
echo "[extra] The hand-written MXFP4 kernel against the reference decoder"
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["LAZYLORA_REPO"])
sys.path.insert(0, os.path.join(os.environ["LAZYLORA_REPO"], "scripts"))
import torch
from make_tiny_model import load_tiny_config
from lazy_lora.native import kernel
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer
from lazy_lora.trainer.loss import compute_cross_entropy_loss

N = int(os.environ["QS_SEQ_LEN"])
cfg, _ = load_tiny_config(os.environ["LAZYLORA_MODEL_DIR"])
cfg.training.max_seq_len = N
V = cfg.model.vocab_size
seq = (torch.arange(N + 1, dtype=torch.long) * 37 + 11) % (V - 8) + 3
inp, tgt = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)

def forward_loss():
    trainer = LazyLoRATrainer(cfg)
    packed = trainer.expert_streamer.keep_packed
    with torch.no_grad():
        h = trainer._embed_tokens(inp)
        trainer._reset_block_residual()
        for l in range(cfg.model.num_hidden_layers):
            h, _ = trainer.forward_layer(l, h)
        h = trainer._finalize_hidden(h, trainer._block_residual)
        loss, _ = compute_cross_entropy_loss(trainer._project_lm_head(h), tgt,
                                             ignore_index=cfg.model.pad_token_id)
    trainer.close()
    return float(loss), packed

if kernel() is None:
    print("      the native MXFP4 kernel is not available here (no gcc with OpenMP, or a "
          "non-x86 host).")
    print("      The engine falls back to the torch dequantiser, which computes the same "
          "thing more slowly, so this A/B has nothing to compare.")
    sys.exit(2)

a, packed_a = forward_loss()
os.environ["LAZYLORA_NO_NATIVE"] = "1"
b, packed_b = forward_loss()
del os.environ["LAZYLORA_NO_NATIVE"]
print(f"      native kernel (experts consumed packed={packed_a}) : loss {a:.6f}")
print(f"      torch decoder (experts consumed packed={packed_b}) : loss {b:.6f}")
print(f"      difference {abs(a - b):.2e}")
sys.exit(0 if abs(a - b) < 1e-3 else 1)
PY
case "$?" in
    0) note "extra native A/B" PASS "the hand-written AVX2/OpenMP MXFP4 kernel and the reference decoder agree on the loss, so the fast path is not a different model." ;;
    2) note "extra native A/B" SKIP "the native kernel could not be built here; the engine used the slower torch path throughout." ;;
    *) note "extra native A/B" FAIL "the two MXFP4 paths disagree." ;;
esac

# -------------------------------------------------------------------------- summary
hr
echo "SUMMARY"
hr
printf '%b' "$RESULTS"
hr
if [ "$FAILED" -eq 0 ]; then
    cat <<'TXT'
Everything that ran, passed.

What this shows: the engine streams a Kimi-K3-shaped checkpoint off disk one layer at a
time, computes both attention types, routes tokens through MXFP4 experts, spills its
activations, replays every layer under autograd, and updates a LoRA adapter with AdamW -
and its gradients are the true gradients of its forward pass.

What this does NOT show: anything about Kimi K3 itself, or about any model getting better
at anything. The weights here are 8.3 MB of deterministic noise. The claim this repository
makes is about the mechanism and its verification, and the pre-registered evaluation of
the real run has not been carried out yet.

Next: docs/QUICKSTART.md, part 2, for the commands that need the real 1.56 TB checkpoint.
TXT
    exit 0
else
    echo "At least one step failed. docs/QUICKSTART.md has a section on what each failure means."
    exit 1
fi
