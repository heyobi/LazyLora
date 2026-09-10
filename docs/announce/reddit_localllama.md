# r/LocalLLaMA

**Written** 9 September 2026, revised 10 September 2026.
**Status:** not posted. When it is, this line gets the date and the link to the thread.

Where this draft and [`../numbers.md`](../numbers.md) disagree, that table names the source
and the source settles it.

---

## Title

```
Training a LoRA adapter on Kimi K3 (2.78T params, 1.56TB of weights) on a 2017 laptop with 7.6GB of RAM — 7.4 hours per step, and here's the verification
```

Alternates:

```
Out-of-core LoRA on a 2.78T MoE from a USB hard disk: 7.6GB of RAM, 7.4h per 1024-token step
```

```
I checked my out-of-core trainer for Kimi K3 against an independent C implementation layer by layer. All 93 rows are in the repo. Now it's training a LoRA adapter on a 7.6GB laptop.
```

---

## Body

The machine:

| | |
|---|---|
| CPU | i7-7700HQ, 4 cores / 8 threads, AVX2, 2017 |
| RAM | 7.6 GB |
| GPU | GTX 1050, 2 GB (mostly idle — it runs the routed-expert matmuls, nothing else) |
| NVMe | 117 GB, about 11 GB free |
| The model | Kimi K3, 2.78 T params, 1.56 TB of weights on a 2 TB USB hard disk |

Nothing about that fits. One MoE layer of K3 is 896 routed experts at 17.5 MB of MXFP4
each — 15.7 GB for one layer, twice the machine's RAM. So the unit of streaming is the
individual expert, not the decoder layer. Non-expert weights are packed into a 108.8 GB
contiguous file on NVMe and served through an index overlay; everything else comes off the
USB disk. One layer is resident at a time, layer-boundary activations go to a ring buffer
on NVMe, and the backward pass recomputes each layer under autograd and streams the routed
experts a second time. Permanently in RAM: the 590 MB LoRA adapter (rank 16, alpha 32, on
q_proj/v_proj and the expert gate/up/down projections) and its two Adam moments, about
1.8 GB in total — which is also the size of a checkpoint. **The 2.78 T base parameters are
frozen and never change.**

One design decision worth stating up front, because it is the first thing I would ask
about someone else's MoE LoRA: on the routed path this is **one rank-16 adapter per layer,
shared by all 896 experts of that layer**, living in K3's MoE latent space
(3584 → 3072 → 3584) — see `lazy_lora/trainer/lazy_trainer.py:130`. It is not one adapter
per expert; that would be about 26 billion trainable parameters (about 320 k per expert
× 896 × 92 ≈ 2.6 × 10¹⁰, roughly 420 GB of fp32 weights, gradients and Adam moments), which
this machine cannot hold and 400 examples could not train. The cost of the choice is that
the adapter learns a correction applied to whichever expert fires rather than per-expert
specialisation, and
given how domain-specific the routing looks in the traces below, that may matter. A
per-expert or per-group ablation is the obvious follow-up and I cannot run it here.

### The proof run

Five Turkish instruction examples packed into two sequences — 1082 tokens in total, 528 of
them trained answer tokens, packing limit 1024. The two sequences alternate. Loss on
answer tokens only, prompt masked, lr 1e-3.

| step | sequence | loss | ppl | forward finished |
|---:|---|---:|---:|---|
| 1 | A | 0.909 | 2.48 | 8 Sep 12:26 |
| 2 | B | 0.521 | 1.68 | 8 Sep 18:13 |
| 3 | A | 0.500 | 1.65 | 8 Sep 23:42 |
| 4 | B | 0.193 | 1.21 | 9 Sep 05:28 |
| 5 | A | 0.157 | 1.17 | 9 Sep 11:03 |

A: 0.909 → 0.500 → 0.157. B: 0.521 → 0.193. B starts lower because it is measured after
the first update on A. The loss is logged when the forward pass ends and the backward runs
about three hours longer, so consecutive rows are 5.5-5.8 h apart.

Those five rows are the last five lines of `evidence/forward_loss_proof.jsonl`, with Unix
timestamps, so the 5.5-5.8 h is a subtraction you can do yourself rather than a number you
have to accept: consecutive rows are whole steps here, and they come out at 20813, 19756,
20756 and 20093 seconds. The main run is not like that — see the note under "What you can
reproduce" below, because the same file has a trap in it and I would rather hand it to you
than have you find it.

**This is memorisation of five examples and I am saying so in the post rather than in a
reply.** It proves the forward → backward → AdamW → checkpoint loop is correct end to end
and that the adapter of a 2.78 T model can be moved by gradient descent on this hardware.
It proves nothing about whether the model is better at anything.

### Verification, which is the part I actually care about

- **Forward.** Checked against [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c),
  an independent C99 implementation of the same model by FareedKhan-dev, using its
  per-layer dump hook. All 93 layers agree at cosine 0.9857 or better — the minimum row is
  0.985744 at layer 71 — and 0.999840 at the output. Scope: 34 tokens of one English
  paragraph, with LoRA B zero-initialised so the adapter contributes exactly nothing and
  the two engines have to agree exactly. Not yet checked with a trained adapter or at 1024
  tokens. **The whole log is in the repo**: `evidence/cmp93_en34_2026-09-06.log`, 98 lines,
  every layer's cosine, max absolute difference, both engines' standard deviations and
  expert count, plus the 2869 s / 426.59 GB the run cost. Read the rows I did not choose to
  quote — layers 68-72 are the worst stretch (0.9857-0.9897) and they are all in there.
- **Ops.** Eight op-level comparisons against the reference implementation: seven match at
  1e-5 absolute / 1e-4 relative, the latent MoE block at 2e-4 absolute (cosine 1.000000) —
  that one is the MXFP4 decode path's own rounding. The fifteen fixture files those come
  from are published by `kimi-k3-in-c` under Apache-2.0, and they are vendored here at
  `tests/fixtures/ops/` with the attribution and the upstream commit recorded, so the only
  external check this project has runs on every push, on a machine neither implementation's
  author controls — rather than only for a reader who thought to clone a second repository.
- **Backward.** Central finite differences on real weights: layers 1 (KDA + MoE), 3 (MLA),
  12 (block boundary) and 13 (two bank entries) — layers 1 and 3 swept over all 16 LoRA
  tensors plus the input and residual-bank directions, layers 12 and 13 over the input and
  bank directions and that layer's LoRA tensors — 4 tokens, fp32 engine with the loss
  reduced in float64. Worst relative error **9.1e-3**, at layer 1, in two directions whose
  analytic derivative is about 3e-4 — at that magnitude the finite difference is the
  noisier of the two estimates. 3.6e-3 on the MLA layer, ≤ 2.0e-3 everywhere else. Four
  layers of 93, so this is not a whole-model gradient check. It needed adaptive epsilon
  and routing-flip detection to mean anything: nudge a weight too far in a top-16 router
  and a different expert wins, the loss surface steps, and you measure the step. This is
  the one verification with no artefact behind it — the harness prints to the terminal
  rather than to a file, so those four numbers are transcribed from my notes and want a
  re-run on the real checkpoint. `scripts/verify_backward.py` is in the repo; the log is
  not, because there never was one.
- A tensor missing from disk raises `MissingTensorError` instead of being quietly replaced
  by random weights. Synthetic tensors need `LAZYLORA_ALLOW_SYNTHETIC=1`. This sounds
  obvious and it is the single change that caught the most real bugs.

### Speed and memory

| | |
|---|---|
| 1024-token step | **7.44 h** — the mean of the two intervals between the main run's first three steps (7.26 h and 7.62 h). Step 1 measured on its own was 6 h 59 m: forward 3 h 11 m (123 s/layer), backward 3 h 48 m (147 s/layer) |
| Proof run step | 5.5-5.8 h, on the shorter packed sequences above |
| Resident set | 4.5-4.7 GB held for 27 hours in the proof run; 4.0-4.7 GB in the main run, plus about 2.4 GB of swap. Not a peak: the highest figure recorded anywhere in the project is 6.24 GB, on an earlier 256-token step |
| Disk, aggregate | **measured 110 MB/s** across the USB disk (routed experts) and the NVMe trunk (non-expert weights): 3,219,659,335,955 bytes through `read()` in the first 8 h 07 m of the main run |
| Disk, inside one sweep | 61 MB/s effective (14.5 GB per MoE layer in 238 s). The reader thread idles during compute — a real prefetch pipeline is the obvious next win. The enclosure's ~115 MB/s is its own sequential benchmark, not a rate this engine achieves |
| Batching | 8× the tokens costs 2× the layer time (128 → 1024 tokens, 122 → 238 s per layer): cost is per sweep, not per token |
| USB bridge | resets about 45 times an hour under load. Reads retry; no read has failed permanently in 27 hours of training |

### What is not new here

Almost all of it. Layer-at-a-time streaming from disk: AirLLM, and `kimi-k3-in-c` on this
very model. Recomputing a layer under autograd from a stored boundary activation: that is
gradient checkpointing, standard since 2016, and ZeRO-Infinity and LoHan already spill
activations to NVMe. LoRA-only trainables with a small optimizer state: LoRA and QLoRA.
Fused low-bit decode-and-dot kernels: ggml has shipped those for years. Expert caches and
hot sets: the entire MoE serving literature.

[AirLLM](https://github.com/lyogavin/airllm) shipped layer-streamed LoRA *training* in
September — 125B in 6 GB of VRAM — which is the same core idea, arrived at independently.
The differences here are the scale (2.78 T), the streaming unit having to be the 17.5 MB
expert because a layer's experts are 15.7 GB, the budget being 7.6 GB of *system* RAM
rather than a CUDA card with host RAM free, a USB hard disk instead of local NVMe, and the
layer-by-layer verification.

The closest thing that also trains a trillion-parameter MoE locally is [KTransformers](https://github.com/kvcache-ai/ktransformers) +
LLaMA-Factory on Kimi K2.5 (1 T): 2-4 × RTX 4090, an AMX Xeon, ~2 TB of system RAM,
~45 tok/s. If you have that machine, use it. This is a 2.8× larger model on roughly 260×
less RAM, and it pays about seven and a half hours per step for the privilege. It is not a speed result.

As far as I can find, nobody has published a backward pass through a model this size
inside a single consumer machine. If that is wrong, link it and I will add it to the
related-work table in the README.

### Routing measurements that fell out of it — and these you can recompute

Traces of five texts (Turkish 264 tokens, English 159, Chinese 111, a Turkish news-style
paragraph 261, Python 167) across all 92 MoE layers, logging (layer, token, expert,
weight). **All five are in the repository**, in `evidence/traces/`, about 6 MB. Everything
in this section comes out of:

```
python scripts/analyze_trace.py evidence/traces/tr_paragraph_L93_2026-09-06
python scripts/analyze_trace.py evidence/traces/tr_paragraph_L93_2026-09-06 evidence/traces/en_paragraph_L93_2026-09-06
```

NumPy, no checkpoint, no GPU, seconds. The format is three int32s and two flat arrays per
record and it is documented in `lazy_lora/monitor/trace.py`, so you can also just read it
yourself in ten lines and not trust my analysis script either. Because routing is causal,
one trace gives the unique-expert curve at every prefix length, which is where the
concentration-vs-batch-size numbers come from.

- **Concentration.** A batch touches 43-56 % of the experts a uniform router would, and it
  is not monotone with depth: ~430 unique experts at layers 1-36, a trough of 243 at layers
  49-60, then a mild widening back to ~295 at layers 73-92.
- **Domain over language.** Expert overlap between Turkish, English and Chinese versions
  of the same paragraph is 0.35-0.39 — the same as two unrelated passages in one language.
  Prose vs Python is 0.20. A language signature exists only in layers 1-8 of 92 and never
  comes back, which disagrees with what has been reported for smaller multilingual MoEs.
  Caveat: five texts, and the EN/ZH versions are translations of a Turkish original written
  for this study.
- **Locality is per token, not per batch.** Consecutive tokens share 26 % of their experts
  (1 % at random), and a per-layer LRU over 64-128 experts (1.1-2.2 GB) hits 62-72 % when
  decoding one token at a time — within a few points of what [a pre-registered study](https://arxiv.org/abs/2608.18261)
  measured on a model ninety times smaller. But a *training batch* reads the union: 42 %
  of all experts at 128 tokens, 53 % at 256, ~85 % at 1024 (that last point measured on
  layers 0-12, the least concentrated layers, so treat it as an upper bound). So caching
  helps decoding and does not survive a training batch — the right primitive for a
  training step is a bandwidth-optimal sequential sweep amortised over the batch. A 200 GB
  static hot set saves 22 % of expert reads in simulation, and this machine has 11 GB of
  NVMe free, so that lever does not exist here.
- **The Turkish tax is in the tokenizer, not the router.** The same content costs 1.7× the
  tokens and 1.6× the bits per byte of English; at equal length Turkish routes slightly
  *more* concentrated.
- **Massive activations.** In the last two MLA layers, one token per text hits a residual
  norm of 1-2 × 10⁴ against a median of 78. The C reference reproduces it exactly, which
  is a nice independent confirmation that it is the model and not my arithmetic.

The five texts were written for this study, including the Turkish news-style one, which is
about hazelnut production statistics and is not lifted from any publication. Nothing in the
bundle was regenerated for release; the only edit is that this machine's filesystem paths
were replaced with placeholders.

### What you can reproduce

Honest tiers, because the checkpoint is 1.56 TB and you do not have it:

| You have | You can check |
|---|---|
| Only the repo | Every routing table above, from `evidence/traces/` via `scripts/analyze_trace.py` — NumPy, seconds. The eight op-level comparisons against the C implementation, from the fifteen fixtures vendored at `tests/fixtures/ops/` — these used to require cloning a second repository and they do not any more. The proof run's step time and the main run's step-1 forward, by subtracting timestamps in `evidence/forward_loss_*.jsonl` and `evidence/run_manifest.json` — read the three warnings under this table first. All 93 cosine rows, by reading `evidence/cmp93_en34_2026-09-06.log`. `bash scripts/quickstart.sh`, which builds a tiny K3-shaped checkpoint and runs the whole loop on it, finite differences included. And `bash scripts/run_mock_tests.sh` — the engine end to end on synthetic weights, which proves the plumbing runs and nothing about numerics |
| + the 1.56 TB checkpoint and a C dump | *regenerating* the 93-layer comparison, rather than reading the log of it |
| + the checkpoint | the finite-difference check on the real model, which has no artefact — that harness prints to the terminal |
| + the published adapter | the evaluation, once it exists |

`sha256sum -c SHA256SUMS` inside `evidence/` covers the whole bundle.

Three warnings about the two loss files, because they are the messiest thing in the bundle
and you would find all three anyway:

1. **They are one file.** `forward_loss_proof.jsonl` is 57 lines,
   `forward_loss_main.jsonl` is 58, and the first 57 are byte-identical. The trainer appends
   every completed forward pass to a single `forward_loss.jsonl` and does not record which
   run a line belongs to, so the "proof" file is that file when the proof run ended and the
   "main" file is the same file one row later. The missing `run` field is a real defect and
   it is on the fix list; publishing the log verbatim rather than tidying it is the reason
   you can see the defect at all.
2. **Two rows say `"step": 1` at loss 0.909084**, 27 h 38 m apart. Not a copy-paste. The
   proof set is the first five records of the 400-example training file — `build_train_set.py`
   writes it as `picked[:5]` of the same selection, and I checked that by parsing both files
   and comparing records, not by trusting the code — and both runs start from a
   zero-initialised adapter, so the first packed sequence and the first forward pass are the
   same computation. Same text, same frozen weights, same zero adapter, same number to six
   decimals, 27 hours apart: a determinism check I am glad to have. It does mean the five
   proof examples sit inside the main run's training set; they are outside every evaluation
   slice.
3. **It is the production log verbatim**, so it also carries mock-suite lines at loss
   ≈ 12.0066 and ≈ 6.9078 — ln(163840) and ln(1000), not measurements of Kimi K3 — and its
   very first line is `"loss": nan, "perplexity": inf`, which strict JSON parsers refuse.
   The real rows are the ones at loss < 1.

And be precise about which timings the bundle actually gives you. The proof run's step time
is a subtraction of consecutive rows, because there the loss lines are whole steps apart.
The main run's is not, in the bundle as published: the 9 September snapshot carries one row
for it. That row's forward is derivable — `run_manifest.json`'s `started=1788947613.66`
against `time=1788959106` is 11492 s, 3 h 11 m 32 s, two seconds under the 3 h 11 m 34 s the
trainer printed, the gap being process start versus log write. The backward, 3 h 48 m 07 s,
and the 6 h 59 m 41 s total for step 1 are the trainer's own printed timings; nothing in the
bundle confirms them. Nor does it confirm the 7.44 h I quote above as the step time: that is
the mean of the two intervals between the run's first three logged forward passes, 7.26 h
and 7.62 h, read from the live trainer log rather than from the September snapshot in the
repository. I would rather say all of that than let "do the subtraction yourself" cover more
than it does.

Quickstart and the full command list with prerequisites are in the README:
https://github.com/heyobi/LazyLora

About the quickstart: it builds a tiny random model in a temporary sandbox so you can
exercise the engine without the 1.56 TB checkpoint. It was written by reading the engine
while this laptop was busy with the training run, so for a while it had never been executed
anywhere — it runs on GitHub Actions now, on every push. Its first run found a real defect,
and the defect was in the check rather than in the engine: the finite-difference test failed
on layer 3's residual-bank direction at 3.1e-2. The analytic gradient was right; the step
was too small for fp32 to resolve against a tensor of norm 60.85. The step is now taken
relative to the perturbed tensor's norm with one Richardson extrapolation, and that
direction agrees to 2.09e-05. A second CI run had passed that same check, because a
different random direction was drawn — so the first run caught both the defect and the fact
that it was intermittent. The whole episode is written up in Bulgular §20. Dependencies are
two packages, `numpy>=1.24` and `torch>=2.3`.

### The caveat, and what happens next

Currently running: 400 Turkish instruction examples from `atasoglu/databricks-dolly-15k-tr`
packed into 154 sequences of at most 1024 tokens, 100 optimizer steps at batch size 1 —
0.65 of an epoch, so about 260 of the 400 examples are seen exactly once. Started 9
September; at the measured 7.44 h step that is about 31 days, so it lands around
9-11 October.

Then the evaluation, which was fixed **before** training: bits per byte on a 2048-token
slice of Turkish news published after the model's release. Baseline 0.455, success is
≤ 0.441, and English Wikipedia must not degrade past 0.198 from 0.194. That threshold was
committed to git on 8 September at 07:54, 29 hours before the run started, and both the
commit and the run manifest are in the repository. Weakness I will state before anyone
else does: git dates come from this laptop's clock and the repo was private until launch,
so even with an annotated tag on that commit, what you have is two timestamps from one
machine and one operator — a pre-commitment, not an independent witness.

My own expectation: it may well not move. 100 batch-1 updates over roughly 51,000 trained
tokens is not much signal, and I am measuring an instruction-tuned adapter with a
language-modelling metric on news. A negative result gets posted as a negative result,
with the diagnostics.

So: no benchmark claims, no "it got smarter", no evaluation number yet. What I have is a
loop that is verified at every level I could think of, running on hardware that should not
be able to do it — and, unusually for a one-laptop project, the raw measurements are in
the repository rather than in a folder on my desk. Please poke holes, especially in the
verification. The comparison log is `evidence/cmp93_en34_2026-09-06.log`, the traces are
`evidence/traces/`, the finite-difference harness is `scripts/verify_backward.py`; and if
you find a way the cross-implementation check could pass while both engines are wrong
together, that is the comment I most want to read.
