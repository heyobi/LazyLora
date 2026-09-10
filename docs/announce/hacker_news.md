# Hacker News — Show HN

**Written** 9 September 2026, revised 10 September 2026.
**Status:** not posted. When it is, this line gets the date and the link to the thread.

Where this draft and [`../numbers.md`](../numbers.md) disagree, that table names the source
and the source settles it.

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
> checkpoint, no GPU, seconds. The reference implementation's fifteen op fixtures are
> vendored in the repo as well, under their own Apache-2.0 licence, so that
> cross-implementation comparison runs in CI on every push instead of only for whoever
> clones two repositories.
>
> Cost: 7.44 h per 1024-token step — the mean of the two intervals between the run's first
> three steps, 7.26 h and 7.62 h. Step 1 measured on its own was 6 h 59 m, forward 3 h 11 m
> and backward 3 h 48 m. Resident set 4.0-4.7 GB plus about 2.4 GB of swap. Measured read
> throughput 110 MB/s aggregate across the USB disk and the NVMe trunk — 3.22 TB through
> read() in the first eight hours.
>
> Not proven: that the model gets better at anything. The proof run drove the loss on a
> fixed sequence from 0.909 to 0.157 — memorisation of five examples. The threshold for
> the main run went into git 29 hours before it started; the run needs about 31 days at
> the measured step time, so the number is due around 9-11 October and I will post it
> either way.

About 455 words, one long screen. If it has to be shorter, cut the cost paragraph —
someone will ask for it in the thread anyway. Do not cut the two evidence sentences: a
Show HN whose top comment tells the reader exactly which two paths to open is a different
thread from one that asks to be believed.

---

## Replies

The questions these drafts expect, and their answers, are in [../../FAQ.md](../../FAQ.md).
