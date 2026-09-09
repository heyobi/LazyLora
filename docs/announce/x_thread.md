# X thread

Eleven posts. Character counts are the raw string length; X weights a URL at 23 regardless
of its real length, so post 11 is comfortably inside the limit. Post 1 stands alone — if
nobody reads past it, it is still true and still complete.

**Before posting.**

- The repository must already be public: post 11 links it, and a call to action that 404s
  is worse than no thread. If it is not ready, the thread waits.
- Look up FareedKhan-dev's actual X handle and @-mention them in post 5; do not guess it.
- The figures are regenerated and safe to attach: the square cards now say a LoRA adapter
  was trained rather than that a model learned, and their footers read "5,5-5,8 sa /
  5.5-5.8 h" for the proof-run step and "4,5-4,7 GB" resident set instead of a single
  peak. `run_terminal.svg` is now main-run step 1 alone — real trainer line formats,
  forward 3 h 11 m 34 s, loss 0.9091, backward 3 h 48 m 07 s, 6 h 59 m 41 s in total —
  and it carries a caption saying which three timestamps are measured. Check any image
  against the figure on disk before it goes out; do not reuse an older export.
- Post 6 is the one that changed most. The comparison log and all five routing traces are
  now **in the repository**, under `evidence/`, so the thread may say plainly that a
  reader can go and check — and must, because that is the strongest thing this project
  has. Confirm they are on the public clone first.
- The two loss files in `evidence/` are one append-only log: `forward_loss_proof.jsonl` is
  57 lines, `forward_loss_main.jsonl` is the same 57 plus the main run's step 1, and both
  carry a `"step": 1` row at loss 0.909084. Know the answer before post 8 goes out; the
  note under it says where the full version lives.
- No evaluation result exists. Nothing in the thread may imply one.

Post at 16:00-18:00 TRT on a weekday. Reply to your own thread with the repo link again
after an hour, since the first post is what gets quoted.

---

## 1/11  (267 chars)

```
Kimi K3 is 2.78 trillion parameters and 1.56 TB of weights. I am training a LoRA adapter on it out of core, on a 2017 laptop with 7.6 GB of RAM and the checkpoint on a USB hard disk. The base model is frozen the whole time. The only thing trained is a 590 MB adapter.
```

*Attach:* `docs/figures/proof_loss_en.png` — the square loss card, whose title now names
the adapter.

## 2/11  (268 chars)

```
Why the usual trick does not work here: one MoE layer of K3 is 896 experts at 17.5 MB = 15.7 GB. Twice the machine's RAM. So the unit of streaming has to be the individual expert, not the decoder layer — and the backward pass has to read the routed ones a second time.
```

## 3/11  (278 chars)

```
Layer-boundary activations go to a ring buffer on NVMe. Each layer is recomputed under autograd on the way back. Gradients cross K3's attention-residual bank to a layer evicted twelve layers earlier. Permanently in RAM: a 590 MB adapter plus its two Adam moments — about 1.8 GB.
```

*Attach:* the right panel of `docs/figures/proof_loss_wide.png` — what does not fit in
RAM, drawn to scale.

The 1.8 GB is the number to give, not 590 MB: the adapter is 590 MB of fp32 parameters and
Adam keeps two more copies of it. A checkpoint is the same 1.8 GB on disk.

## 4/11  (275 chars)

```
On the routed path it is one rank-16 adapter per layer, in K3's MoE latent space, shared by all 896 experts of that layer. Not one per expert — that would be ~26B trainable parameters. It learns a correction that applies to whichever expert fires, which is a real limitation.
```

Somebody will ask this within the first ten replies of any MoE fine-tuning thread, so it
is better said here than conceded there. The line in the code is
`lazy_lora/trainer/lazy_trainer.py:130`. If asked what the alternative would look like:
per-expert or per-group adapters, an ablation this machine cannot run.

## 5/11  (270 chars)

```
I did not want to trust the forward pass, so I check it against kimi-k3-in-c, an independent C implementation of the same model. All 93 layers agree at cosine 0.9857 or better — lowest 0.985744, layer 71 — and 0.99984 at the output, on 34 tokens with the adapter zeroed.
```

*Attach:* render the 93-layer cosine table as an image from
`evidence/cmp93_en34_2026-09-06.log`. This is the post that earns the thread; give it the
picture. @-mention FareedKhan-dev here.

## 6/11  (272 chars)

```
And you can check it yourself. The whole comparison log is in the repo: evidence/cmp93_en34_2026-09-06.log, every layer's cosine, max diff and expert count. So are all five expert-routing traces — every routing number I publish recomputes from them with numpy, in seconds.
```

The post that matters most after post 1. Everything else in this thread is a number I am
asking to be believed; this is the one where a stranger with a laptop and no checkpoint can
sit down and verify the claims. If a reply asks how: `git clone`, then
`python scripts/analyze_trace.py evidence/traces/tr_paragraph_L93_2026-09-06`.

## 7/11  (275 chars)

```
Gradients: central finite differences on real weights — four layers, LoRA tensors plus the input and bank directions. Worst 9.1e-3, in two directions whose true derivative is ~3e-4; 3.6e-3 on MLA, ≤2e-3 elsewhere. Nudge a weight, a different expert wins: a step, not a slope.
```

Do not post the 3.6e-3 on its own. It is the worst error on the MLA layer, not the worst
error, and the thread is the form most likely to be screenshotted away from the repo where
the fuller sentence lives. If someone asks for this log: there is none. The harness prints
to the terminal, so these four numbers are the one part of the verification with no
artefact behind it, and saying so costs nothing.

## 8/11  (272 chars)

```
Proof run: five Turkish examples, two packed sequences, loss on answer tokens only. A: 0.909 → 0.500 → 0.157. B: 0.521 → 0.193. That is memorisation of five examples. It proves the forward-backward-AdamW loop is correct. It does not prove the model got better at anything.
```

*Attach:* the loss table as a screenshot (step, sequence, loss, forward finished), or the
left panel of `proof_loss_wide.png`. The raw rows, with Unix timestamps, are the tail of
`evidence/forward_loss_proof.jsonl` in the repo, and consecutive rows there are whole steps
apart — 20813, 19756, 20756 and 20093 s.

If a reply points out that `forward_loss_proof.jsonl` and `forward_loss_main.jsonl` are the
same file plus one row, or that both contain a `"step": 1` row at loss 0.909084: agree at
once, do not argue it in 280 characters, and link the full answer. It is a missing `run`
field in one append-only log, and the repeated 0.909084 is the same text through the same
frozen weights with a zero adapter, 27 h 38 m apart. The reply is in `hacker_news.md` under
Prepared replies; the thread should carry the link, not the paragraph.

## 9/11  (259 chars)

```
What it costs: 6 h 59 m for one 1024-token step — forward 3 h 11 m, backward 3 h 48 m. Resident set 4.0-4.7 GB plus about 2.4 GB of swap; the highest peak recorded anywhere was 6.24 GB. Measured read 110 MB/s aggregate, across the USB disk and the NVMe trunk.
```

*Attach:* `docs/figures/run_terminal.svg` rendered to PNG. It is main-run step 1 and
nothing else now, with a caption inside the image saying that the start, the forward end
and the backward end are measured and the layer lines between them are interpolated at the
measured pace, so it can be attached as-is.

The 6 h 59 m is the main run's step 1 at 1024 tokens; the proof run's 5.5-5.8 h was on
shorter sequences and must not be quoted for this. The 4.0-4.7 GB is a resident set, not a
peak, and the machine is also holding swap — do not let the two merge into "4.7 GB peak on
a 7.6 GB machine", which reads as headroom the machine does not have. The 110 MB/s is
measured: 3.22 TB through `read()` in the first 8 h 07 m of the run, both devices together.

## 10/11  (259 chars)

```
Where the time goes: 14.5 GB read per MoE layer at 1024 tokens. 8x the tokens costs 2x the layer time — cost is per sweep, not per token, so batching is nearly free and caching is not the lever. The 115 MB/s on the enclosure's box is a spec, not a rate I hit.
```

## 11/11  (277 chars)

```
The success threshold went into git on 8 Sep at 07:54, 29 h before the run started: Turkish news bits-per-byte 0.455 → ≤0.441. About 29 days at the measured step time, so around 8-9 Oct. I will post the number either way. github.com/heyobi/LazyLora — tell me where it is wrong.
```

If anyone presses on the pre-registration: it is a commit and an annotated tag, both
timestamped by this laptop's clock on a repository that was private until launch. Two
timestamps, one machine, one operator. Say that rather than letting "pre-registered" do
work it has not earned.

---

## Notes

- The post that used to say "nine rows published, the rest ship later" is gone, and no
  version of it comes back. The log is in the repository; the thread says so.
- The thread has no evaluation result in it and must not acquire one before the run ends.
  If somebody asks "did it work", the answer is post 11 plus "the number is due when the
  run ends, around 8-9 October".
- Never write a date earlier than 8 October. The old "early October" came from the proof
  run's shorter steps.
- Do not quote-tweet yourself with a stronger version of post 1 later. The strong version
  is the one that gets screenshotted, and the strongest true version is already there.
- If somebody asks how to try it, the quickstart is code that has never been run: the
  scripts were written by reading the engine while the machine was busy training, and every
  runtime and memory figure in their documentation is derived, not measured. Say that
  rather than promising a five-minute demo.
- If the thread travels, the four replies worth having ready are the AirLLM comparison, the
  KTransformers one, "how do I know you didn't make this up" and "your two loss files are
  the same file" — all four in `hacker_news.md` under Prepared replies. Keep them to one
  post each; the third is two paths and a command, so it fits, and the fourth does not — a
  reply that concedes the defect in one line and links the long answer is the right shape
  on this platform.
