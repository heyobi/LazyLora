# Questions a sceptic asks first

Every answer here was written before anyone asked the question. This project makes a claim that sounds wrong — a LoRA adapter trained on a 2.78-trillion-parameter model on a 2017 laptop with 7.6 GB of RAM — and the useful response to a claim like that is not to believe it but to try to break it. Each answer below ends with the file you would open to check it, and then with where the checking stops: what is in this repository, and what would need the 1.56 TB checkpoint that is not. Where the honest answer is "you cannot check that from here", it says so, and it says so before the part that sounds impressive. If your question is not here, `.github/ISSUE_TEMPLATE/verification_report.md` exists for exactly that, and a question that makes this file longer is the kind of issue this project most wants to receive.

Where this document and [docs/numbers.md](docs/numbers.md) disagree, that table names the source and the source settles it.

---

### 1. "How do I know you did not make this up?"

Two things you can check from a fresh clone with nothing else on the machine. The first is
[`evidence/cmp93_en34_2026-09-06.log`](evidence/cmp93_en34_2026-09-06.log): 98 lines, one
row for every one of the 93 layers, each carrying the cosine against an independent
implementation, the maximum absolute difference, both engines' standard deviations and how
many experts that layer read — all the rows, not a summary of them, so you can sort it and
find the worst one yourself rather than take mine. The second is better, because it is
recomputation rather than reading: `python scripts/analyze_trace.py
evidence/traces/tr_paragraph_L93_2026-09-06` rebuilds every routing number this project
publishes out of the bundle with NumPy in seconds — no checkpoint, no GPU — and if one of
those numbers is wrong, that command is how you catch me.

**Open:** [`evidence/`](evidence/) and [`evidence/README.md`](evidence/README.md); `sha256sum -c SHA256SUMS` from inside it covers all of it.

**Where the checking stops:** the 1.56 TB checkpoint and the C engine's per-layer dump are not published, so the comparison log can be read but not regenerated; the finite-difference numbers have no log behind them (question 3); there is no evaluation result yet (question 11); and the pre-registration rests on this laptop's own clock (question 10).

---

### 2. "How do you know your forward pass is right?"

I do not trust it; I check it against something I did not write.
[`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) is an independent C99
implementation of this model by FareedKhan-dev, it has a per-layer dump hook, and I replay
its dump through my engine and compare layer by layer: all 93 layers agree at cosine
**0.9857 or better** — the lowest row is **0.985744 at layer 71** — and **0.999840** at the
output. Below that sit eight op-level fixtures from the same implementation, now vendored
in this repository under Apache-2.0 with attribution and run on every push: seven match at
1e-5 absolute / 1e-4 relative, and the MoE block at 2e-4 absolute with cosine 1.000000,
which is the MXFP4 decode path's own rounding. The two engines also independently produce
the same massive activations in the last two MLA layers — one token per text at a residual
norm of 1-2 × 10⁴ against a median of 78 — and that is a distinctive enough fingerprint
that agreement is unlikely to be coincidence.

**Open:** [`evidence/cmp93_en34_2026-09-06.log`](evidence/cmp93_en34_2026-09-06.log), [`tests/fixtures/ops/`](tests/fixtures/ops/), [`.github/workflows/quickstart.yml`](.github/workflows/quickstart.yml).

**Where the checking stops:** 34 tokens of one English paragraph, with LoRA B zero-initialised so the adapter contributes exactly nothing and both engines must agree exactly. It is not verified with a trained adapter, at 1024 tokens, or on Turkish or code input. And a shared misreading of the checkpoint format would fool both engines — the op fixtures make that less likely, they do not exclude it.

---

### 3. "How do you know the gradients are right?"

Start with what it is not: this is not a whole-model gradient check. It covers 4 layers of
93 on 4 tokens — 1 (KDA + MoE, one bank entry), 3 (MLA), 12 (a block boundary), 13 (two
bank entries) — by central finite differences on real weights rather than on a toy, in an
fp32 engine with the loss reduced in float64. The worst relative error is **9.1e-3**, at
layer 1, in two directions whose analytic derivative is about 3e-4, where the finite
difference is the noisier of the two estimates; **3.6e-3** on the MLA layer, and 2.0e-3 or
better everywhere else. I quote the 9.1e-3 first because quoting the 3.6e-3 as "the worst"
is a cherry-pick you would catch by opening [DEVAM.md](DEVAM.md) §11, and because getting a
meaningful answer at all needed adaptive epsilon and routing-flip detection: nudge a weight
too far in a top-16 router and a different expert wins, the loss surface steps, and the
difference quotient measures the step instead of the derivative.

**Open:** `scripts/verify_backward.py`, [DEVAM.md](DEVAM.md) §11.

**Where the checking stops:** the harness prints its table to the terminal instead of to a file, so these four numbers are transcribed from my notes and are the one part of the verification with no artefact behind it; reproducing them means re-running that script on the real weights. The end-to-end evidence that the loop is correct is a different thing entirely — the loss actually falls on a fixed sequence, pass after pass, which a subtly wrong gradient does not do for long.

---

### 4. "Your two loss files are the same file, and the same first loss appears twice."

Both true, and the first of them is a real defect — in the logging, not in the runs. There
is one `forward_loss.jsonl`; the trainer appends a line to it every time a forward pass
completes and it has no field saying which run the line belongs to, so
`forward_loss_proof.jsonl` in the bundle is that file as it stood when the proof run ended,
57 lines, and `forward_loss_main.jsonl` is the same file one line later, 58, the extra line
being the main run's step 1. The two rows reading `"step": 1` at loss **0.909084** are the
other half of the question and that one is not a defect: they are 27 h 38 m apart — `time`
1788859586 and 1788959106 — and they agree to six decimals because they are the same
computation. The proof set is the first five records of the 400-example training file, and
LoRA B is zero-initialised in both runs, so the same text through the same frozen weights
with an adapter contributing exactly nothing must give the same number to the last digit;
that makes it a determinism check across two runs a day apart on a machine streaming
1.56 TB off a USB disk, and it also means those five examples are inside the training set,
where they are in no evaluation slice.

**Open:** [`evidence/forward_loss_main.jsonl`](evidence/forward_loss_main.jsonl), [`evidence/forward_loss_proof.jsonl`](evidence/forward_loss_proof.jsonl), `scripts/build_train_set.py`. The fix is a `run` field, and it is on the open list.

**Where the checking stops:** a line is written when a forward pass ends, so nothing in that file marks the end of a backward — the main run's backward and its step total come from the trainer's printed timings and the bundle does not confirm them. The file also carries mock-suite rows at loss ≈ 12.0066 and ≈ 6.9078, which are ln 163840 and ln 1000 and are not measurements of Kimi K3 at all, and its very first line is `"loss": nan`, which a strict JSON parser will refuse.

---

### 5. "Is the LoRA per expert or per layer?"

Per layer. On the routed path there is one rank-16 adapter living in K3's MoE latent space,
3584 → 3072 → 3584, shared by all 896 routed experts of that layer, which is what keeps the
trainable set at 147 M parameters. One adapter per expert would be on the order of 26
billion trainable parameters — about 320 k per expert × 896 × 92 — and roughly 420 GB of
fp32 weights, gradients and Adam moments, which neither this machine nor 400 examples can
support. The consequence is real and worth naming rather than burying: the adapter learns a
correction that applies to whichever expert fires, not per-expert specialisation, so if
routing is as domain-specific as the traces suggest, a per-expert or per-group adapter
might learn something this one structurally cannot.

**Open:** `lazy_lora/trainer/lazy_trainer.py:130`, [Bulgular.md](Bulgular.md) §19.1.

**Where the checking stops:** that ablation is the obvious next question and this hardware cannot run it. It is named as future work, not waved away.

---

### 6. "Is this just inference offloading with extra steps?"

No, and the difference is the whole project. Inference reads each weight once and throws the
activations away; training has to keep, for every layer, the input the backward pass will
need, and it has to read the routed experts a second time on the way back — 896 experts per
layer with top-16 firing per token, so the streaming unit has to be the individual 17.5 MB
expert rather than the decoder layer, because one layer's experts alone are 15.7 GB, twice
this machine's RAM. Then the gradients have to go somewhere correct: K3 writes an
attention-residual bank every 12 layers, so a gradient computed at layer 84 has to be routed
back to the layer that wrote that entry, across a boundary where the layer that produced it
is long gone from memory. Layer-boundary activations live in a ring buffer on NVMe and each
layer is recomputed under autograd when the backward reaches it.

**Open:** [The engine](README.md#the-engine) in the README, and `lazy_lora/streaming/`.

**Where the checking stops:** none of the pieces are new — see question 7 — and the claim here is about the composition and the verification, not about any one of them.

---

### 7. "Isn't this AirLLM?"

It is the same core idea, arrived at independently and concurrently, and AirLLM shipped it in
public in September 2026: layer-streamed LoRA training with frozen weights streaming from
disk and only the adapters resident, 125 B in 6 GB of VRAM. What differs is the
model being 22× larger, the streaming unit having to be the 17.5 MB expert instead of the
decoder layer, the budget being 7.6 GB of *system* RAM with a mostly idle GTX 1050 rather
than a CUDA card with the host RAM free, the weights sitting on a USB hard disk, and the
forward and backward being numerically verified against an independent implementation. None
of the components are mine — gradient checkpointing (2016), layer streaming (AirLLM,
`kimi-k3-in-c`), LoRA and QLoRA, fused low-bit kernels from ggml, expert caches from the
whole MoE serving literature — and the contribution I claim is the composition at this scale
plus the verification, stated as a search result rather than as a fact about the world.

**Open:** [Related work](README.md#related-work), which exists so you can see exactly which line is different.

**Where the checking stops:** "as far as I can find" is the strength of that claim. If you know of prior work, send it and it goes in the table.

---

### 8. "Why not KTransformers or llama.cpp?"

Because they need a machine I do not have, and if you have one you should use them. The
documented recipe for LoRA SFT of a trillion-parameter MoE is KTransformers with
LLaMA-Factory on Kimi K2.5: 2-4 × RTX 4090, an AMX Xeon, about 2 TB of system RAM, ~45
tok/s. This is a 2.8× larger model on roughly 260× less RAM with no usable GPU — and it pays
about **7.44 hours per step** for that, which belongs in the same sentence as the ratio, or
the ratio reads as a speed claim. It is not one.

If the question is "why not Colibri, WARP or BigMoeOnEdge", the answer is shorter: they are
inference engines, and very good ones, and they do not compute a gradient. Colibri in
particular is the right tool for *running* Kimi K3 on this laptop; this engine is not built
to run it, it is built to train an adapter on it, and its forward pass is slow for exactly
that reason (per-sweep cost amortised over a batch, Bulgular.md §16.5). Nothing here competes
with them on inference, and the related-work table says so.

**Open:** [Related work](README.md#related-work) and [Cost, measured](README.md#cost-measured).

**Where the checking stops:** the KTransformers figures are their published ones, not something I measured; so are the star counts and descriptions of the three inference engines, read from their repositories on 10 September 2026.

---

### 9. "Seven hours a step is useless."

Two follow-ups this invites, with the measurements. *Why not more tokens per step, if the
cost is per sweep?* Because the sweep law stops at about 1024 tokens on this CPU: the 6
September profiles over layers 0-11 took 1602 s at 1024 tokens and 3108 s at 2048, while
bytes read grew only from 176 GB to 197 GB. Twice the tokens, 1.94 times the time, 12 %
more disk: beyond 1024 the step is compute-bound, so 4096-token steps would not see more
data per hour here. *Why not prefetch, since the disk is rated 115 MB/s and a sweep gets
61?* The 61 MB/s is one layer sweep in the layers 0-12 profile; over the whole step the
process reads 110 MB/s aggregate, of which the NVMe trunk is about 8 MB/s (218 GB per step
over 26,784 s), so the USB disk is already at about 102 MB/s, 89 % of its benchmark.
Double-buffering the 17.5 MB expert reads could recover at most a tenth of the step.


For production fine-tuning, yes — entirely useless, and nothing here argues otherwise. What
it buys is that the floor for touching a model this size becomes a laptop and patience
rather than a cluster, and that the cost is now a measured number instead of a guess: the
step is **7.44 hours**, the mean of the two intervals between steps 1 and 3 of the main run
(7.26 h and 7.62 h). The step is bandwidth-bound — 14.5 GB read per MoE layer at 1024
tokens — and the bandwidth is not being used well: aggregate over the run's first eight
hours is 110 MB/s across both devices (3.22 TB through `read()`), while inside a single MoE
layer sweep the expert reads come off the USB disk at about 61 MB/s, because the reader
thread idles during compute. A real prefetch pipeline is the obvious next win, and one
useful measurement fell out of the profiling: 8× the tokens costs only 2× the layer time,
because the cost is per sweep and not per token.

**Open:** [Cost, measured](README.md#cost-measured), [Bulgular.md](Bulgular.md) §16.5.

**Where the checking stops:** the ~115 MB/s figure you may see quoted elsewhere is the USB enclosure's own sequential benchmark, not a rate this engine achieves; the two are not the same number and merging them would overstate this engine by nearly a factor of two. The step time is measured on this run, on this disk, and moves with disk health.

---

### 10. "Your pre-registration is just a git commit you made yourself."

Correct, and it is the weakest link in the project. The threshold is in commit `6605306` at
8 September 2026 07:54:49, 29 hours before the run started, and an annotated tag
`preregistration-2026-09-08` points at that commit, so GitHub records when the tag arrived
as well as when the commit did. But git dates come from this laptop's clock, the repository
was private until it went public on 10 September 2026, the tag is mine too, and this
repository's history was rewritten **twice** before publication — so what you have is two
timestamps from one machine and one operator, not an independent witness. Take it for
exactly what it is: a pre-commitment I would have had to plan to fake a month in advance,
and no further.

Both rewrites, since a rewritten history is exactly the thing you should want itemised.
The first dropped two AI-session database files and the tracked bytecode that had been
committed by accident. The second normalised twenty-eight commits that carried a
placeholder author identity, and stripped a `Claude-Session:` line the tool had appended to
forty commit messages — one private URL, repeated identically, that resolves for my account
and for nobody else, which is precisely what this repository asks nobody else to accept.
Neither pass removed an attribution: the `Co-Authored-By: Claude` trailers were left alone,
and they are still there, on every commit from the twenty-ninth on — count them on the day
you read this with `git log --format='%(trailers:key=Co-Authored-By,valueonly)' | grep -c
Claude` against `git log --oneline | wc -l`. Commit hashes quoted anywhere in this repository therefore date from after the
second rewrite, while the dates they carry are the original author dates — which is why
`6605306` is quotable at all.

**Open:** [Evaluation protocol, registered before training](README.md#evaluation-protocol-registered-before-training); `git show 6605306` and `git show preregistration-2026-09-08`.

**Where the checking stops:** there is no external anchor, and I would rather say so than let the word "pre-registered" do work it has not earned. If you know a cheap externally timestamped one, tell me and the next run uses it.

---

### 11. "Did it actually improve Turkish?" — and "does the quickstart work?"

On Turkish: unknown, and that is the honest state — **no evaluation result exists**. The
metric is bits per byte on a 2048-token slice of Turkish news published after the model's
release, baseline 0.455, success at ≤ 0.441, with English Wikipedia as a forgetting control
(0.194, must stay ≤ 0.198); the threshold went into git 29 hours before the run started. At
the measured 7.44 hours per step, 100 steps is about 31 days, so the number is due around
**9-11 October 2026**, and my own expectation is that it may well not move: 100 optimizer
steps at batch size 1 over roughly 51,000 trained tokens is not much signal. A negative
result gets published as a negative result, under a title that says so.

On the quickstart: yes, and not on my word for it. `.github/workflows/quickstart.yml` runs
the whole thing on GitHub Actions on every push, on a machine I do not control — and the
first time it ran, it failed. The finite-difference step failed on the residual-bank
direction at layer 3, relative error 3.1e-2 against a 2e-2 tolerance; the diagnosis was that
the analytic gradient was correct and the *step* was too small for fp32 to resolve against a
tensor of norm 60.8, so the rule is now relative to the perturbed tensor's own norm with a
Richardson extrapolation, and the same direction now agrees to **2.09e-05**. The episode is
written up in [Bulgular.md](Bulgular.md) §20, including the part that matters most: a second
run had passed that same check at 1.83e-2 purely because it drew a different random
direction. A check that has never failed has never been tested. This one has.

**Open:** [Quickstart, without the 1.56 TB checkpoint](README.md#quickstart-without-the-156-tb-checkpoint), [docs/QUICKSTART.md](docs/QUICKSTART.md), [Bulgular.md](Bulgular.md) §20.

**Where the checking stops:** the quickstart proves the code path and the verification harness on a toy model it generates itself; it shows nothing about Kimi K3. The badge, not this paragraph, is the current claim — read it, and if it is green and this file is wrong, the file is wrong.
