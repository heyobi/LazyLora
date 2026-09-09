# To FareedKhan-dev (kimi-k3-in-c)

Send this before the first public post, not after. The X thread and the Show HN both name
`kimi-k3-in-c` in their first few lines, and the person whose engine is the oracle should
hear it from us rather than from a notification.

**Where.** Whatever contact the repository lists; if there is none, a GitHub Discussion or
a short issue titled the same as the subject line works and has the advantage of being
public and linkable. Keep it to one message. No follow-up if there is no reply — the
credit stands either way.

**Before sending.** The comparison log is in the repository now
(`evidence/cmp93_en34_2026-09-06.log`), so this message links it rather than offering to
attach it. That is a better message than the earlier draft: he does not have to ask, and
nothing about the offer depends on us remembering to send a file. Check that the link
resolves on the public repository before pressing send — a mail to the author of the
oracle pointing at a 404 is the one version of this that does damage.

**The paragraph about timing has to match what is actually true on the day you send it.**
The draft below says the repository is public, because the first LinkedIn and X posts link
it and they go out in the same few days. If the repository is still private when you send
this, then beat 1 is not ready either: hold both, or say "public within days, before I post
anything publicly" and mean it. What must not happen is telling the author of the oracle
"October" while a live link to the repo is already in a public thread.

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
I can tell is just the MXFP4 decode path's own rounding (cosine 1.000000).

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
still going — about 29 days at the measured step time, so it finishes around 8-9 October —
and the pre-registered evaluation runs when it does; I will publish that number either way.
kimi-k3-in-c is credited in the README, in the acknowledgements and in the announcement
posts, as the reference implementation the forward pass is validated against — if you would
prefer different wording, a different link, or your name written a particular way, tell me
and I will use yours.

Thank you for writing it in C99 with a dump hook. That decision is the reason anyone can
check my work.

— [your name]
github.com/heyobi/LazyLora
