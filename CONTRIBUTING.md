# Contributing

This is a one-author research repository, and the thing it needs most is not code. It is a
second machine, a second operator and a second set of numbers. Everything below is in
service of that.

## What is most wanted, in order

1. **Independent re-verification.** Run `scripts/quickstart.sh`, the op fixtures or
   `scripts/verify_backward.py` on hardware the author does not have, and report what you
   got — including, especially, a number that disagrees with a published one. Use the
   *Verification report* issue template.
2. **Cache-policy experiments on the routing traces.** Which experts a token picks, layer by
   layer, is the whole economics of this design: an expert cache that guesses well turns a
   1.42 TB sweep into far fewer reads. The five traces are in the repository, under
   `evidence/traces/`: 92 MoE layers per text, `trace.bin` + `trace.json` + `analysis.json`
   + `analysis.md`. A cache or prefetch policy can be evaluated against them today with
   NumPy alone — no checkpoint, no GPU. Record your own with `scripts/measure_routing.py`
   only if you want other texts.
3. **Ports to another MoE checkpoint.** The out-of-core scheme assumes a model whose routed
   experts dominate the parameter count and are read a few per token. Making the loader,
   the router and the expert streamer work against a different checkpoint of that shape is
   the most useful structural contribution.

Bug reports, documentation fixes and CI improvements are welcome too, and small ones do not
need to be discussed first.

## Setup

Linux or macOS. The hot read path uses `os.pread`, so Windows is not supported. Python 3.10+.

```sh
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3"
python3 -m pip install -e .          # or, without installing: export PYTHONPATH="$PWD"
```

**torch 2.3 is a hard floor**, not a preference: the loader reinterprets numpy `uint16`
arrays as bfloat16 with `torch.from_numpy(...).view(torch.bfloat16)`
(`lazy_lora/streaming/mmap_loader.py:326`), and `from_numpy` gained `uint16` support in 2.3.
The CPU index URL is there because the engine is a CPU engine; a CUDA build works and is
simply larger. `numpy` and `torch` are the only runtime dependencies.

## The quickstart

```sh
bash scripts/quickstart.sh            # everything
bash scripts/quickstart.sh --fast     # skips the synthetic-weight suite, the slowest step
```

It generates a small but genuinely Kimi-K3-shaped checkpoint with
`scripts/make_tiny_model.py` and runs the real engine against it: the real streaming loader,
attention, router, MXFP4 experts, autograd replay, AdamW and checkpoint format. Only the
weights are small. Everything it writes goes into one `mktemp -d` directory under `$TMPDIR`
that is deleted on exit (`--keep`, `--dir` to place it); it does not write into the clone.

**Its runtimes and memory figures are derived from the code, not measured.** The quickstart
was written by reading the engine while the machine was occupied by the 29-day training run,
and has never been executed here. If it takes ten minutes on your machine instead of the
two or three the header claims, that is a data point and an issue worth opening.

Two more commands a reader can run:

```sh
bash scripts/run_mock_tests.sh        # nine engine test modules on synthetic tensors
python3 -m unittest lazy_lora.tests.test_reference_ops
```

The second one needs the op fixtures from the reference C implementation
([kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)): clone it beside this
repository, or set `LAZYLORA_REF_FIXTURES` to its `tests/fixtures/ops`. Without them the
module skips and says so; with `LAZYLORA_REF_FIXTURES` pointing somewhere wrong it fails
rather than skipping, because a skipped check that reads like a pass is worse than no check.

## Two rules the code does not bend

**A tensor that is not on disk raises.** It is never replaced by a random, zero or all-ones
stand-in. The costliest mistakes in this project came from loaders that quietly invented
weights: the code kept running and printed plausible numbers that meant nothing.
`lazy_lora.core.config.synthetic_allowed()` raises `MissingTensorError` unless
`LAZYLORA_ALLOW_SYNTHETIC=1`, which only `scripts/run_mock_tests.sh` sets. A patch that adds
a synthetic fallback on a path the real run can reach will not be merged; make it raise, and
name the tensor and the shard in the message.

**The forward pass is not trusted until it matches the C reference, and the backward is not
trusted until finite differences agree.** Concretely, and these are the numbers to beat or
contradict:

- Op fixtures: seven of the eight match `kimi-k3-in-c` at 1e-5 absolute / 1e-4 relative. The
  composite MoE block fixture matches at **2e-4 absolute** with cosine 1.000000 — it chains
  six matmuls per expert plus two latent projections, and the reference sums each dot product
  in double while torch sums in float32. Do not claim the MoE block matches at 1e-5.
- Finite differences (`scripts/verify_backward.py`, DEVAM.md §11 and 268-276): four layers
  checked — 1 (KDA + MoE, one bank entry), 3 (MLA), 12 (block boundary), 13 (two bank
  entries) — on 4 tokens, fp32 engine, float64 loss reduction, adaptive epsilon, routing-flip
  detection. Layers 1 and 3 were swept over all 16 LoRA tensors plus the input and
  residual-bank directions; 12 and 13 covered the input and bank directions and the layer's
  LoRA tensors. Worst relative error **9.1e-3**, at layer 1, in two directions whose analytic
  derivative is about 3e-4 — at that magnitude the finite difference is the noisier estimate;
  3.6e-3 on the MLA layer; the rest at or under 2.0e-3. Quote it with that qualification or
  not at all.

A change to an op, a kernel or the gradient path is expected to come with the fixture or the
finite-difference run that shows it still agrees.

## When your numbers disagree with the published ones

Open a **Verification report** issue. It asks for your machine, your torch version, the exact
command, the number you got and the number you expected, because without those a
disagreement cannot be chased. This is the most valuable issue you can file here, and it is
not a complaint: several published figures come from one laptop and one operator. The two
that used to be assertions — the 93-layer cosine comparison and the routing traces — are now
in `evidence/` and can be read and recomputed rather than believed; what still cannot be
re-derived here is anything needing the 1.56 TB checkpoint.

## Housekeeping

- Do not commit `__pycache__`, sandboxes, or measurement scratch files.
- Describe a change by what it does and how you measured it, on which machine.
- Keep the wording of results exact. The 2.78 T parameters of Kimi K3 are **frozen**; what is
  trained is a LoRA adapter. The proof run is deliberate memorisation of five examples and
  demonstrates the mechanism only — not that "the model learned" anything. The evaluation has
  not run, so nothing in a patch or an issue should say that it succeeded.
