# Hacker News — Show HN

**Before posting.** Repository public, `LICENSE`/`NOTICE` in place, the README's step time
and end date correct, and — for beat 2 — the evaluation result written into the README
before anything is posted. The evidence bundle is in the repository now (`evidence/`,
6.2 MB, 25 files, `SHA256SUMS`), which is what makes the "how do I know you didn't make
this up" answer a link instead of an apology: check that `git ls-files evidence/` returns
it on the public clone before you post, because the whole thread leans on it. The
quickstart has never been executed — run it once from a fresh clone on a machine that is
not the training laptop, or say in the thread that its numbers are derived from the code
rather than measured. Read the "your two loss files are the same file" reply below before
you post rather than after: `evidence/forward_loss_proof.jsonl` and
`evidence/forward_loss_main.jsonl` differ by one line, both carry a `"step": 1` row at loss
0.909084, and somebody will run `diff` in the first hour. It is a missing field in a log
writer and it reads like fabrication until it is explained.

Post Tue-Thu, 08:00-10:00 ET (15:00-17:00 TRT). Be at the keyboard for the next four hours.

---

## Title

```
Show HN: Training a LoRA adapter on a 2.78T-parameter model on a 2017 laptop
```

76 characters. Alternates, if the first reads as too pleased with itself:

```
Show HN: Out-of-core LoRA fine-tuning of a 2.78T model in 7.6GB of RAM
```
70 characters.

```
Show HN: LazyLoRA – LoRA training on a 1.56TB model from a USB hard disk
```
72 characters.

Prefer the first. It leads with the verb — inference of this model on a laptop was done a
month ago by someone else, and gradient descent through it is the part that is new.

**URL:** https://github.com/heyobi/LazyLora

---

## First comment

> One MoE layer of this model is 896 experts at 17.5 MB — 15.7 GB, twice the machine's
> RAM. So the streaming unit has to be the individual expert, not the decoder layer, and
> the backward reads the routed ones a second time. Layer-boundary activations go to an
> NVMe ring buffer, each layer is recomputed under autograd when the backward reaches it,
> and gradients cross K3's attention-residual bank back to a layer evicted twelve layers
> ago. Resident throughout: the 590 MB fp32 adapter and its two Adam moments, about
> 1.8 GB. On the routed path it is one rank-16 adapter per *layer*, shared by all 896
> experts of that layer; per-expert adapters would be about 26 B trainable parameters.
>
> I did not want to trust the forward pass, so it is checked against FareedKhan-dev's
> kimi-k3-in-c, an independent C99 implementation with a per-layer dump hook: all 93
> layers agree at cosine 0.9857 or better — the lowest row is 0.985744 at layer 71 — and
> 0.999840 at the output, on 34 tokens with the adapter zero-initialised so the two
> engines must agree exactly. The whole log is in the repo,
> `evidence/cmp93_en34_2026-09-06.log`, so that is all 93 rows rather than the nine I
> would otherwise be asking you to take on trust. The backward is checked by central
> finite differences on real weights — four layers; worst relative error 9.1e-3 in two
> directions whose analytic derivative is about 3e-4, 3.6e-3 on MLA, ≤ 2.0e-3 elsewhere.
>
> The five expert-routing traces are in `evidence/traces/` too, so every routing number in
> the write-up recomputes from the repository:
> `python scripts/analyze_trace.py evidence/traces/tr_paragraph_L93_2026-09-06`. NumPy, no
> checkpoint, no GPU, seconds.
>
> Cost: 6 h 59 m per 1024-token step (forward 3 h 11 m, backward 3 h 48 m). Resident set
> 4.0-4.7 GB plus about 2.4 GB of swap. Measured read throughput 110 MB/s aggregate across
> the USB disk and the NVMe trunk — 3.22 TB through read() in the first eight hours.
>
> Not proven: that the model gets better at anything. The proof run drove the loss on a
> fixed sequence from 0.909 to 0.157 — memorisation of five examples. The threshold for
> the main run went into git 29 hours before it started; the run needs about 29 days at
> the measured step time, so the number is due around 8-9 October and I will post it
> either way.

About 390 words, one long screen. If it has to be shorter, cut the cost paragraph —
someone will ask for it in the thread anyway. Do not cut the two evidence sentences: a
Show HN whose top comment tells the reader exactly which two paths to open is a different
thread from one that asks to be believed.

---

## Prepared replies

Do not paste these unprompted. They are for the questions that will come.

**"Isn't this just AirLLM / what's new?"**

> AirLLM shipped layer-streamed LoRA training in September — frozen weights stream from
> disk, only the adapters stay resident, 125B in 6 GB of VRAM — and it is the same core
> idea, arrived at independently. The differences are the model being 22× larger, the
> streaming unit having to be the 17.5 MB expert instead of the decoder layer because a
> single layer's experts are 15.7 GB, the budget being 7.6 GB of *system* RAM with a GTX
> 1050 that is mostly idle rather than a CUDA card with the host RAM free, the weights
> being on a USB hard disk rather than local NVMe, and the forward and backward being
> numerically verified against an independent implementation. None of the components are
> mine: gradient checkpointing (2016), layer streaming (AirLLM, kimi-k3-in-c), LoRA/QLoRA,
> fused low-bit kernels (ggml has shipped those for years), expert caches (the whole MoE
> serving literature). The composition at this scale, and the verification, are the
> contribution.

**"Why not KTransformers / llama.cpp?"**

> The documented recipe for LoRA SFT of a trillion-parameter MoE is KTransformers with
> LLaMA-Factory on Kimi K2.5: 2-4 × RTX 4090, an AMX Xeon, about 2 TB of system RAM, ~45
> tok/s. That is the right tool if you have the machine. This is a 2.8× larger model on
> roughly 260× less RAM — and it pays seven hours per step for that, so it is not a speed
> result in any sense.

**"Seven hours a step is useless."**

> For production fine-tuning, yes. What it buys is that the floor for touching a model
> this size is a laptop and patience rather than a cluster, and that the cost is now a
> measured number instead of a guess. The step is bandwidth-bound: 14.5 GB read per MoE
> layer at 1024 tokens. Measured aggregate over the first eight hours of the run is
> 110 MB/s across both devices (3.22 TB through read()); inside a single MoE layer sweep
> the expert reads come off the USB disk at about 61 MB/s, because the reader thread idles
> during compute, so a real prefetch pipeline is the obvious next win. The 115 MB/s figure
> you may see quoted is the enclosure's own sequential benchmark, not a rate this engine
> achieves. One useful measurement fell out of it: 8× the tokens costs only 2× the layer
> time, because the cost is per sweep, not per token. Batching is nearly free here; the
> sweep is not.

**"How do I know you didn't make the numbers up?"**

This is the question the whole announcement is built around, and since the evidence bundle
went in it has a good answer. Give the two commands, in this order, and then the
limitations — not the other way round.

> Clone it and check. The layer-by-layer comparison against kimi-k3-in-c is in the
> repository: `evidence/cmp93_en34_2026-09-06.log`, 98 lines — four of header, one row per
> layer for all 93, and a total — each row carrying the cosine, the max absolute
> difference, both engines' standard deviations and how many experts that layer read. The
> header has the 34 token ids it ran on. Minimum 0.985744 at layer 71,
> 0.999840 at the output, 2869 s and 426.59 GB read. All 93 rows, not a summary of them.
> `sha256sum -c SHA256SUMS` inside `evidence/` covers the bundle.
>
> Better than that, because it is recomputation rather than reading: every routing number
> in the write-up comes out of `evidence/traces/`, five traces over all 92 MoE layers in a
> documented little-endian format (`int32 layer, N, K`, then `int16` expert ids and
> `float16` weights). `python scripts/analyze_trace.py evidence/traces/tr_paragraph_L93_2026-09-06`
> reproduces the concentration, entropy and locality tables; pass two directories and it
> prints the cross-language overlap that the "domain over language" claim rests on. NumPy,
> no checkpoint, no GPU, seconds. If one of those numbers is wrong, that command is how
> you catch me.
>
> Also checkable with nothing but the repo: the op fixtures against the reference
> implementation (seven of eight at 1e-5 absolute / 1e-4 relative, the MoE block at 2e-4
> absolute with cosine 1.000000), the engine end to end on synthetic weights, and the raw
> per-step losses with Unix timestamps in `evidence/forward_loss_main.jsonl` and
> `forward_loss_proof.jsonl`. Most of the step timings in the documentation are a
> subtraction of two of those timestamps, so do the subtraction — but let me be exact about
> which ones, because the file does not carry everything. Consecutive rows in the proof run
> are whole steps, forward end to forward end: 20813, 19756, 20756 and 20093 seconds, which
> is the 5.5-5.8 h I quote. The main run's forward comes out of
> `run_manifest.json`'s `started=1788947613.66` against the last row's `time=1788959106`:
> 11492 s, 3 h 11 m 32 s, two seconds off the 3 h 11 m 34 s the trainer printed because the
> manifest timestamps the process starting and the log line timestamps the loss being
> written. The backward and the 6 h 59 m total are *not* in the bundle — the loss is logged
> when the forward ends, so nothing marks the end of a backward pass, and those two numbers
> come from the trainer's own printed timings. Two more warnings about that file: it carries
> lines from the mock suite at loss ≈ 12.0066 and ≈ 6.9078, which are ln 163840 and ln 1000
> and are not measurements of Kimi K3 at all, and its very first line is
> `"loss": nan, "perplexity": inf`, which a strict JSON parser will refuse.
>
> What you still cannot check, which I would rather say than have you discover: the
> 1.56 TB checkpoint is not published and neither is the C engine's per-layer dump, so the
> comparison log can be read but not regenerated; the finite-difference harness prints to
> the terminal instead of to a file, so those four numbers are transcribed from my notes
> and need a re-run on the real checkpoint; the quickstart has never been executed by
> anyone including me; there is no evaluation result yet; and the pre-registration rests
> on this machine's clock plus an annotated git tag on the commit, which gives a second
> timestamp on GitHub but not an independent one. The adapter goes to Hugging Face when
> the run ends, which is the one artefact that makes the central claim checkable end to
> end.

**"Your two loss files are the same file"** / **"the same first loss appears twice"**

Somebody will run `diff` on the bundle within the first hour, and they will be right. Say it
before they do — ideally in the reply above, and certainly the moment it comes up. It looks
like fabrication and it is a missing field in a log writer.

> Both, and you have found a real defect — in the logging, not in the runs. There is one
> `forward_loss.jsonl`; the trainer appends a line to it every time a forward pass completes,
> and it has no field saying which run the line belongs to. So `forward_loss_proof.jsonl` in
> the bundle is that file as it stood when the proof run ended, 57 lines, and
> `forward_loss_main.jsonl` is the same file one line later, 58, the extra line being the
> main run's step 1. The first 57 are byte-identical because they are the same 57 lines.
> That was a poor way to publish it and the fix is a `run` field, which is on the list.
>
> The two rows that read `"step": 1` at loss 0.909084 are the other half of your question,
> and that one is not a defect. They are 27 h 38 m apart — `time` 1788859586 and 1788959106
> — and they agree to six decimals because they are the same computation. The proof set is
> the first five records of the 400-example training file: `scripts/build_train_set.py`
> writes it as `picked[:5]` of the same selection, and I checked it by parsing both files
> and comparing the records rather than trusting the code. Both runs start from a
> zero-initialised adapter, so LoRA B contributes exactly nothing on the first pass. Same
> text, same frozen weights, same zero adapter, same number — a determinism check across
> two runs 27 hours apart on a machine streaming 1.56 TB off a USB disk, which is a thing I
> would rather have than not. It does mean the five proof examples are inside the main run's
> training set; they are outside every evaluation slice.
>
> One consequence for anyone doing arithmetic on that file: the proof run's step time is a
> subtraction of consecutive rows and comes out at 5.49-5.78 h. The main run's step time is
> not — the file has one row for it. Its forward, 3 h 11 m 32 s, is the last row's `time`
> minus `started` in `run_manifest.json`; the backward and the 6 h 59 m total are the
> trainer's printed timings and nothing in the bundle confirms them.

**"Is the LoRA per expert or per layer?"**

> Per layer. On the routed path it is one rank-16 adapter living in K3's MoE latent space,
> 3584 → 3072 → 3584, shared by all 896 experts of that layer —
> `lazy_lora/trainer/lazy_trainer.py:130`. One adapter per expert would be on the order of
> 26 billion trainable parameters — about 320 k per expert × 896 × 92 ≈ 2.6 × 10¹⁰, and
> roughly 420 GB of fp32 weights, gradients and Adam moments — which neither this machine
> nor 400 examples can support. The consequence is real and worth naming: the adapter
> learns a correction that applies to whichever expert fires, not per-expert
> specialisation, so if routing is as
> domain-specific as the traces suggest, a per-expert or per-group adapter might learn
> something this one structurally cannot. That is an ablation I cannot run on this
> hardware and it is listed as future work rather than dismissed.

**"Did it actually improve Turkish?"**

> Unknown, and that is the honest state. The evaluation is bits per byte on a 2048-token
> slice of Turkish news published after the model's release, baseline 0.455, success at
> ≤ 0.441, with English Wikipedia as a forgetting control (0.194, must stay ≤ 0.198). The
> threshold was committed on 8 September at 07:54, 29 hours before the run started. The
> run is 100 steps at about 7 hours each — roughly 29 days, so the number is due around
> 8-9 October. My own expectation is that it may well not move: 100 optimizer steps at
> batch size 1 over about 51,000 trained tokens is not much signal. A negative result gets
> published as a negative result.

**"Your pre-registration is just a git commit you made yourself."**

> Correct, and it is the weakest link in the project. The threshold is in commit `4e9eed1`
> at 8 September 07:54:49, 29 hours before the run's `started=1788947613`, and there is an
> annotated tag on that commit so GitHub has its own record of it. But git dates come from
> this laptop's clock, the repository was private until launch, and the tag is mine too:
> what you have is two timestamps from one machine and one operator, not an independent
> witness. Take it for what it is — a pre-commitment I would have to have planned to fake
> a month ahead — and no further. If somebody wants to suggest a cheap externally
> timestamped anchor for the next run, I will use it.

**"Does the quickstart work?"**

> Honest answer: unrun. `scripts/quickstart.sh`, `make_tiny_model.py`, `demo_generate.py`
> and `export_traces.py` were written by reading the engine while the laptop was busy with
> the training run, so every runtime, memory figure and expected output in their
> documentation is derived from the code and not measured. The first person to run them on
> a clean machine will find whatever I could not. Bug reports on that path are welcome and
> expected. (It builds and runs a tiny random model in a `mktemp -d` sandbox that it
> deletes on exit — it does not write into your clone, and it does not need the 1.56 TB
> checkpoint.) Dependencies are two packages, `numpy>=1.24` and `torch>=2.3`:
> `pip install -r requirements.txt` and `pip install -e .` install the same thing. Minimum
> torch is 2.3 because the loader turns numpy uint16 arrays into bfloat16 with
> `torch.from_numpy(...).view(torch.bfloat16)`.
