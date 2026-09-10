# To FareedKhan-dev (kimi-k3-in-c)

**Not sent.** As of 10 September 2026 this letter has not gone to FareedKhan-dev; when it
does, this line records the date and the channel it went through.

---

**Subject:** Your kimi-k3-in-c was the oracle for an out-of-core K3 trainer — 93/93 layers
agree

Hi Fareed,

I have spent the last few weeks building LazyLoRA, an out-of-core LoRA trainer for Kimi K3
that runs on one laptop: i7-7700HQ, 7.6 GB of RAM, and the 1.56 TB checkpoint on a USB
hard disk. It streams one layer at a time, keeps layer-boundary activations in a ring
buffer on NVMe, and recomputes each layer under autograd on the way back, so the routed
experts get read a second time. Only the LoRA adapter stays in memory.

None of that would have been checkable without kimi-k3-in-c. I used your per-layer dump
hook as the oracle for my forward pass and compared layer by layer: all 93 layers agree at
cosine 0.9857 or better, with the lowest row at 0.985744 on layer 71 and 0.999840 at the
output, on 34 tokens with the adapter zero-initialised so the two engines have to agree
exactly. Your reference implementation also caught things I would not have found on my own
— the SiTU activation in particular. Mine was g·tanh(4g): unbounded, and it turned negative
inputs positive, so roughly half the channels were being amplified where they should have
been suppressed. After the fix, layer 3's output dropped from 18.88 to 5.27. Of your eight
op fixtures, seven match mine at 1e-5 absolute; the MoE block needed 2e-4, which as far as
I can tell is just the MXFP4 decode path's own rounding (cosine 1.000000). One thing I
should tell you rather than let you find it: I have copied the fixture files — fifteen of
them, covering those eight ops — into my repository at `tests/fixtures/ops/`, unmodified,
with your name, the upstream commit id and the Apache-2.0 notice. That is so the comparison
runs in my continuous integration on every push instead of only for somebody who has cloned
both repositories. If you would rather I fetched them at test time than vendored them, say
so and I will change it.

The whole comparison log is in my repository rather than in a folder on my desk:
`evidence/cmp93_en34_2026-09-06.log` — every layer, with the cosine, the maximum absolute
difference, both engines' standard deviations and the expert count, plus the 34 token ids
it ran on. I am pointing you at it directly because it is, read the other way round, a
per-layer numerical cross-check of *your* implementation by a completely separate one:
annoying to produce for your own code and free for me to hand over. The stretch I would
look at first if I were you is layers 68-72, where the agreement is at its worst
(0.9857-0.9897) before recovering at the output. If any layer looks off to you, or if you
want the comparison re-run with different input, different token count or a dump you
produce yourself, say so and I will run it — that is a few hours of disk time here and
nothing else.

One thing that might interest you independently of my project: both engines reproduce the
same massive activations at the end of the network — in the last two MLA layers, one token
per text reaches a residual norm of 1-2 × 10⁴ against a median of 78. Two implementations
written from different starting points landing on the same outlier is decent evidence it
is the model and not either of our arithmetic.

While I was in there I also recorded the expert routing of five texts across all 92 MoE
layers, and those are in the same directory — `evidence/traces/`, one directory per text,
with the binary format documented in `lazy_lora/monitor/trace.py` and NumPy the only thing
needed to read it. They are what my claims about routing concentration and about expert
choice tracking subject matter rather than language are computed from, and since they need
no checkpoint they may be more useful to you than the comparison log. `sha256sum -c
SHA256SUMS` inside `evidence/` covers everything there.

I am not asking for anything. This is just a heads-up: the repository is public at
github.com/heyobi/LazyLora, and I am about to post about it. The training run itself is
still going — about 31 days at the measured step time, so it finishes around 9-11 October —
and the pre-registered evaluation runs when it does; I will publish that number either way.
kimi-k3-in-c is credited in the README, in the acknowledgements and in the announcement
posts, as the reference implementation the forward pass is validated against — if you would
prefer different wording, a different link, or your name written a particular way, tell me
and I will use yours.

Thank you for writing it in C99 with a dump hook. That decision is the reason anyone can
check my work.

— Ibrahim Polat
github.com/heyobi/LazyLora
