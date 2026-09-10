# Every number, and where it comes from

One table of every quantity this project quotes, with the log line or file each was read from, and a second list of the numbers that look like measurements and are not. It exists because the worst error this repository has made was a number rather than a claim: for months every document said the forward pass agreed with the C reference at "cosine >= 0.988, minimum at layer 72". That was the lowest of nine spot-checked layers, not the lowest of all 93, and publishing the log settles it at 0.985744 on layer 71 — which is to say the correction was forced by making the evidence checkable, not by anyone noticing. A single sourced table is how that stops happening twice. When README.md, Bulgular.md, docs/measurement_note.md and a post on a social network disagree, this file says which of them to fix; and this file is not the authority either — the log is, and the right-hand column tells you which log.

*Table checked row by row against the working tree on 10 September 2026. Every row below was
re-opened at its cited source on that date; the ones that could not be confirmed from a file
in this repository are marked, and say so in their own words rather than being dropped.*

## How to read the Source column

- A source in `backticks` is a path **in this repository**. Open it. If it does not say what
  the middle column says, the middle column is wrong and that is a defect worth an issue.
- A source marked **(off-repo)** is a file, a process counter or a terminal output **on the
  author's machine that is not in this repository**. Those rows are the ones where you are
  trusting the author rather than checking him, and they are marked so you can tell the two
  apart at a glance. Roughly a quarter of the table is like this, and that is the honest
  shape of a project whose subject weighs 1.56 TB.
- "recomputable from `evidence/traces/`" means the raw records are here and the number is a
  short script away: `scripts/analyze_trace.py`, seconds, no checkpoint, no GPU.

## The table

| Quantity | Value | Source |
|---|---|---|
| Model | Kimi K3, 2.78 T parameters, 93 layers (69 KDA linear-attention + 24 gated MLA), 896 routed and 2 shared experts in each of the 92 MoE layers, top-16 routing, MXFP4 expert weights | `README.md`, "Kimi K3 has 93 layers"; the model card itself **(off-repo)** |
| Checkpoint | 1453.74 GiB = 1.56 TB across 96 safetensors shards, on a 2 TB hard disk in a USB enclosure | `README.md`, "The checkpoint is 1453.74 GiB", the machine banner at the top of `Bulgular.md`; the bytes themselves **(off-repo)** |
| One layer's routed experts | 896 × 17.5 MB = 15.7 GB | arithmetic; the per-expert size is `README.md`, "17.5 MB per expert" |
| Packed trunk | 108,811,952,128 B = 108.8 GB of non-expert weights on NVMe | `Bulgular.md` §5 |
| Machine | i7-7700HQ (4 cores / 8 threads, AVX2), 7.6 GB RAM, GTX 1050 2 GB, 117 GB NVMe of which 108.8 GB is the trunk and about 11 GB is free during a run | `README.md`, "Reference machine:", `README.md`, "about 11 GB free during a run", the machine banner at the top of `Bulgular.md` |
| Adapter | 590 MB fp32, 147 M parameters, rank 16, alpha 32, dropout 0, on `q_proj`, `v_proj` and the expert gate/up/down; with its two Adam moments about 1.8 GB resident, which is also the checkpoint size | `README.md`, "LoRA: rank 16, alpha 32", `DEVAM.md` §17 |
| Adapter, routed path | **one rank-16 adapter per layer in the MoE latent space (3584 → 3072 → 3584), shared by all 896 experts of that layer** — not one per expert, which would be about 26 B trainable parameters (≈320 k per expert × 896 × 92 ≈ 2.6 × 10¹⁰, roughly 420 GB at 16 bytes apiece) | `lazy_lora/trainer/lazy_trainer.py:130`, the comment "Routed-expert LoRA adapters, shared across the 896 experts of this layer", `Bulgular.md` §19.1 |
| Evidence bundle | 25 files, 6.2 MB, `SHA256SUMS` over all of it; comparison log, five routing traces, both loss logs, the run manifest. Paths replaced by placeholders, nothing regenerated. `git ls-files evidence \| wc -l` is 26, which is those 25 plus the checksum file itself | `evidence/README.md`, `evidence/SHA256SUMS`, `Bulgular.md` §19 |
| Vendored op fixtures | 15 files, 8.8 MB under their own `SHA256SUMS` (17 tracked paths, the extra two being `README.md` and `SHA256SUMS`), copied unmodified from `kimi-k3-in-c` at upstream commit `c223f490047e93600f05fbdab06b9742d2b8ef08`, 29 August 2026, Apache-2.0 with attribution | `tests/fixtures/ops/README.md`, "at upstream commit", `tests/fixtures/ops/SHA256SUMS`, `Bulgular.md` §20.3 |
| Traced texts | tr_paragraph 264 tokens, tr_news 261, code_python 167, en_paragraph 159, zh_paragraph 111; all five written for this study, the news-style one about hazelnut production statistics and **not** from any publication | the first line of each `evidence/traces/*/analysis.md`; `evidence/README.md` |
| Forward vs the C reference | 93 layer rows, every one at cosine 0.9857 or better. Lowest row **0.985744 at layer 71**; layer 72 is 0.987909, the fourth lowest; the dip runs 68-72 (0.989709, 0.987459, 0.986975, 0.985744, 0.987909); **0.999840** at the output. 34 tokens, LoRA B = 0, 2869 s, 426.59 GB read. **The whole 98-line log is in the repository** | `evidence/cmp93_en34_2026-09-06.log` |
| Op fixtures, result | 7 of the 8 match at 1e-5 absolute / 1e-4 relative; the composite latent MoE block at 2e-4 absolute with cosine 1.000000, because it chains six matmuls through the MXFP4 decode path | `lazy_lora/tests/test_reference_ops.py` — `ABS_TOL = 1e-5`, `REL_TOL = 1e-4`, and `abs_tol=2e-4` hardcoded for the `moe` case, widened in commit `588ec07`; `Bulgular.md` §15.6 |
| Backward vs central differences, **real weights** | layers 1, 3, 12, 13; 4 tokens; worst **9.1e-3** at layer 1 in two directions whose analytic derivative is about 3e-4; 3.6e-3 on the MLA layer; ≤ 2.0e-3 elsewhere. The bare "≤ 2e-3" is a cherry-pick | `DEVAM.md` §11. The harness prints to the terminal and **the log is not in this repository (off-repo)**; `DEVAM.md` says so in its own list of what is missing |
| Finite differences, **tiny model, in CI** | First run on GitHub Actions **failed**: layer-3 residual bank, analytic 1.086830 against central difference 1.121618, relative error **3.1e-2** on a 2e-2 tolerance. The analytic gradient was correct; the step was too small for fp32. With the step made relative to the perturbed tensor's norm (`REL_STEP = 2e-3`) plus one Richardson extrapolation the same direction agrees to **2.09e-05**, and the worst direction in the run is 3.71e-3 | `Bulgular.md` §20, §20.1, §20.2; the rule is `scripts/quickstart.sh` (`REL_STEP` and the docstring of `check()`). See the note below about a third figure A second failure followed on another runner (`shared_gate_lora.B`, 2.04e-2 against 2e-2, 4.9e-3 on the previous runner with identical code); the step now also has to move the loss by at least 2000 fp32 ulps (`SIGNAL_ULPS`), and that direction agrees to 7.71e-5, worst 1.95e-3 (Bulgular §20.4). |
| Residual-bank norms, tiny model | **60.85 at layer 3**, **0.91 at layer 1** — a factor of 66, which is the whole explanation of the episode above: the same absolute step resolves against the small tensor and drowns in fp32 round-off against the large one | `Bulgular.md` §20.1; regenerate the model with `scripts/make_tiny_model.py` and measure it yourself |
| Proof run | 5 examples → 2 packed sequences, 1082 tokens total, 528 trained answer tokens, packing limit 1024, lr 1e-3, warmup 2 | `Bulgular.md` §18 |
| Proof losses | A: 0.909084 → 0.500335 → 0.156585; B: 0.521090 → 0.193009 | the last five lines of `evidence/forward_loss_proof.jsonl` |
| Proof step time | four measured intervals 20813, 19756, 20756 and 20093 s = 5.78, 5.49, 5.77 and 5.58 h, mean **5.65 h**. It applies to that run only — those two sequences held about 541 tokens each | subtraction of consecutive `time` fields in `evidence/forward_loss_proof.jsonl` |
| Main run, step 1 | **6 h 59 m 41 s** on a full 1024-token packed sequence: forward 3 h 11 m 34 s (123.6 s/layer) + backward 3 h 48 m 07 s (147.2 s/layer) | `Bulgular.md` §18.1; `evidence/run_manifest.json` `note`. The forward is derivable from the bundle; the backward and the total are not — see the last section |
| Main run, step **cadence** | **7.26 h and 7.62 h** between the first three logged forward passes, mean **7.44 h**. This, not the 6 h 59 m above, is the figure to use for the run as a whole: 6 h 59 m is one step measured on its own | the live `forward_loss.jsonl` in the run workspace **(off-repo)**. The snapshot committed as `evidence/forward_loss_main.jsonl` is 58 lines and stops at the main run's step 1, so **these two intervals cannot be derived from the bundle** — say so rather than pointing a reader at a file that does not contain them |
| Main run duration | at 7.44 h a step, 100 steps is about **31 days** from 9 September 12:53, landing around **9-11 October 2026**. The spread is the honest one: 7.26 h gives 30.3 days, 7.62 h gives 31.8 days | arithmetic on the row above. Supersedes the "about 7 h, 29 days, 8-9 October" that came from step 1 alone and still appears in some documents |
| Memory | resident set 4.0-4.7 GB in the main run with swap in use; 4.5-4.7 GB held for 27 h in the proof run; **6.24 GB** on an earlier 256-token step is the highest ever recorded, not the current peak and not a peak of this run | `Bulgular.md` §18, `DEVAM.md` §17, `DEVAM.md` "ŞU AN"; the live counters **(off-repo)** |
| Read throughput, aggregate | **measured 110 MB/s** across the USB disk (routed experts) and the NVMe trunk (non-expert weights) together: 3,219,659,335,955 bytes through the process's read counter after 8 h 06 m 57 s of the main run | `Bulgular.md` §18.1, `DEVAM.md` "ŞU AN"; the `/proc` counter **(off-repo)** |
| Read throughput, inside one sweep | **61 MB/s** effective: 14.5 GB per MoE layer in 238 s, layers 0-12, because the reader thread idles during compute | `Bulgular.md` §16.5 |
| Sweep amortisation | 8× the tokens costs 2× the layer time: 128 → 1024 tokens, 122 s → 238 s per MoE layer, layers 0-12 | `Bulgular.md` §16.5 |
| Main run configuration | 400 Dolly-tr examples → 154 packed sequences ≤ 1024 tokens (78 k trained tokens per epoch), 100 steps at batch 1 = 0.65 epoch, ≈ 260 examples seen once, lr 5e-4 cosine, warmup 5, prompt masked, checkpoint every 5 steps | `DEVAM.md` "ŞU AN"; the arguments are in `evidence/run_manifest.json`. The dataset files are built by `scripts/build_train_set.py` and are **(off-repo)** |
| Baselines (bits per byte) | TR news 0.455, TR Wikipedia 0.311, EN Wikipedia 0.194 | `DEVAM.md` §16.1. Measured into `eval/results.jsonl`, which is **(off-repo)** and does not exist in this repository |
| Threshold | TR news ≤ 0.441 (−3 %) **and** EN Wikipedia ≤ 0.198 (+2 % at most) | `DEVAM.md` §16.1, fixed in commit `6605306` |
| Pre-registration | commit `6605306`, 8 Sep 2026 07:54:49 +0300, **29 h** before the run started at 9 Sep 12:53:33; annotated tag `preregistration-2026-09-08` on that commit | `git log 6605306`, `git show preregistration-2026-09-08`, `evidence/run_manifest.json` (`started=1788947613.66`) |
| Routing: concentration | a batch touches 43-56 % of the experts a uniform router would; **not monotone with depth** — about 430 unique experts at layers 1-36, 304 at 37-48, a trough of 243 at 49-60, 338 at 61-72, about 295 at 73-92 | `Bulgular.md` §17; recomputable from `evidence/traces/` |
| Routing: language | cross-language expert-set Jaccard TR/EN 0.39, TR/ZH 0.35, EN/ZH 0.38 — equal to the 0.34-0.37 between two halves of the same text; prose against Python 0.20-0.21, twice as far apart as two languages | `Bulgular.md` §17; recomputable from `evidence/traces/` |
| Routing: where the language signature is | confined to roughly the first 8 of the 92 MoE layers | **not in `Bulgular.md`.** It is `docs/measurement_note.md` §6, where it carries a **†**: produced by the analysis tooling and never transcribed into the experiment log. Treat it as unaudited until somebody runs `scripts/analyze_trace.py <dir_A> <dir_B> --prefix 55` over two directories in `evidence/traces/` and reports the result. It is the weakest-sourced number in this table |
| Routing: locality | consecutive-token expert-set Jaccard **0.258** against 0.009 for random pairs, falling with distance (0.21 at d=2, 0.14 at d=8, 0.10 at d=128); per-layer LRU hit rate 62 % at 64 experts (1.1 GB), 72 % at 128 (2.2 GB), 80 % at 256 (4.5 GB), single-token decoding | `Bulgular.md` §17.2. One line reproduces the 0.258: `jq -s '[.[]\|.rows[].jac_t]\|add/length' evidence/traces/*/analysis.json` → 0.25796 |
| Routing: batch union | experts a batch must read: 16 (2 %) at one token, 116 (13 %) at N=16, 263 (29 %) at 64, 379 (42 %) at 128, 478 (53 %) at 256, and about 85 % at 1024 — **the 1024 point is layers 0-12 only**, the least concentrated layers, so read it as an upper bound | `Bulgular.md` §17.2 and §16.5; recomputable from `evidence/traces/` |
| Turkish tokenizer tax | the same content costs about 1.7× the tokens and 1.6× the bits per byte of English, and routing is **not** more diffuse — at a matched prefix Turkish uses fewer unique experts in 54 of 66 layers | `README.md`, "The Turkish tax is in the tokenizer"; `docs/measurement_note.md` §6 (the matched-prefix control there carries a **†** on its 92-layer figures) |
| Cross-language perplexity | English paragraph loss 1.776, perplexity 5.90, top-1 53 %; Turkish paragraph loss 0.771, perplexity 2.16, top-1 77 %. **Not comparable across languages** — the comparable measure is bits per byte | `Bulgular.md` §17 |
| Massive activations | in the last two MLA layers (91, 92) a single token's residual norm reaches 10-20 thousand against a median of 78 (in English, the 32nd token, " front"); seen in the English and code texts and **not** in the Turkish or Chinese paragraphs; reproduced by the C reference, so it is the model's behaviour and not an engine bug | `Bulgular.md` §17, §17.1 |

## Numbers that are not measurements and must not be quoted as such

Any per-step time merged across the two runs, and **"5.7 h"** in particular — it came from the
proof run, whose two packed sequences held about 541 tokens each rather than a full 1024, so
it may never be quoted for the main run. **"6 h 59 m" quoted as the run's step time** — that
is step 1 measured on its own; the cadence over the first three steps is 7.44 h and the
duration follows from the cadence. **"29 days" and "8-9 October"**, which were derived from
6 h 59 m and are superseded by about 31 days and 9-11 October; some documents in this
repository still carry the old pair, and where they do, this table settles it.
**"saturates 115 MB/s"**, and 115 MB/s as an achieved rate at all — it is the USB
enclosure's own sequential benchmark, a property of the device. **"perplexity 1.9 on
Python"**, which had no log line behind it. The **LRU points at 16, 32 and 448 experts** in
the draft measurement note. **"128 GB NVMe"** (it is 117 GB). **Every runtime and memory
figure in `docs/QUICKSTART.md` Part 1**: those are derived from the code, and the one
execution that has happened was on a GitHub Actions runner, which is not the author's
laptop and does not measure it. And anything from `forward_loss.jsonl` at loss ≈ 12.0066 or
≈ 6.9078, which are ln(163840) and ln(1000) from mock and broken-pipeline runs. Those mock
rows are visible to everyone, in `evidence/forward_loss_main.jsonl`, sitting directly above
the real ones — so every document that sends a reader to that file also says which lines are
not Kimi K3. Point at the rows below loss 1. Its first line is
`"loss": nan, "perplexity": inf`, which strict JSON parsers reject.

Also not a measurement, and now falsifiable in one command: **"cosine ≥ 0.988, minimum at
layer 72"**. The published log's lowest row is 0.985744 at layer 71. Every document in the
repository has been corrected to 0.9857 / layer 71; nothing may reintroduce the old pair,
including from an old screenshot, an old draft or the nine-row table in `Bulgular.md` §17.1
that it came from. That table is still there, deliberately: it is nine sampled layers and it
is labelled as nine sampled layers.

## One number this table cannot yet settle

The docstring of `check()` in `scripts/quickstart.sh` says the fixed finite-difference check
agrees to **1.6e-4** on the bank direction; `Bulgular.md` §20.2 says **2.09e-05**, and the
continuous-integration
log is the thing that decides. They are not the same run and may both be true — the comment
was written while the fix was being made, the Bulgular figure after it — but until somebody
reads the two numbers off one job, quote **2.09e-05** with §20.2 named, and do not quote the
comment. The same paragraph in the script also points at a section of
`docs/QUICKSTART.md` called "the bank direction" which does not exist in that file. Both are
open defects; they are listed in [`pre_publication_check.md`](pre_publication_check.md) §3 rather than fixed silently
here.

## The identical first loss — confirmed, and say it before someone finds it

The main run's first logged loss is **0.909084**, equal to the proof run's first loss to six
decimals. This is not a hedge: the first five records of `datasets/dolly_tr_400.jsonl` were
parsed and compared against `datasets/dolly_tr_proof.jsonl` and the parsed records are equal,
which is by construction — `scripts/build_train_set.py` writes the proof file as `picked[:5]`
of the same selection. Both runs start from a zero-initialised adapter, so LoRA B contributes
exactly nothing, the first packed sequence is the same text, and the first forward pass is
the same computation. It is a determinism check, not a coincidence: had the two numbers
differed, something in the forward pass would be non-deterministic. Disclose in the same
breath that the five proof examples are inside the 400-example training set and outside every
evaluation slice. The two data files themselves are **not** in this repository —
`build_train_set.py` writes them from the Hugging Face dataset — so a reader who wants to
check this rebuilds them with the same seed rather than opening a committed file. Say
"verified by parsing both files here", not "you can see it in the repo".

## The two loss files are one file, and a reader will diff them

`evidence/forward_loss_proof.jsonl` is 57 lines, `evidence/forward_loss_main.jsonl` is 58,
and the first 57 are byte-identical:

```
diff <(head -57 evidence/forward_loss_main.jsonl) evidence/forward_loss_proof.jsonl
```

prints nothing. They are not two runs' logs. The trainer appends every completed forward pass
to one `forward_loss.jsonl` and records no run identifier; the "proof" file is that file as it
stood when the proof run ended, and the "main" file is the same file one row later, that row
being the main run's step 1. Two rows in it read `"step": 1` at loss 0.909084 — the proof
run's first pass at `time` 1788859586 and the main run's first pass at 1788959106, 27 h 38 m
apart — which is the same identical-first-loss fact as above, except that here it sits in a
public file where it looks exactly like a copy-paste until somebody explains it. Explain it
first. The fix in the code is a missing `run` field, which `lazy_lora/trainer/lazy_trainer.py`
still does not write; it is an open defect, listed as one.

## What the bundle lets a reader derive, and what it does not

Worth knowing precisely, because the invitation to "do the subtraction yourself" has to
survive being taken up.

- *Proof-run step time, fully derivable.* Consecutive rows are whole steps, forward end to
  forward end: 20813, 19756, 20756 and 20093 s — 5.78, 5.49, 5.77 and 5.58 h, mean 5.65.
  That is where "5.5-5.8 h" comes from and a reader gets it from the file alone.
- *Main-run forward, derivable to two seconds.* `evidence/run_manifest.json` has
  `started=1788947613.66` and the last row of `evidence/forward_loss_main.jsonl` has
  `time=1788959106`. The difference is 11492 s = **3 h 11 m 32 s**, against the 3 h 11 m 34 s
  the trainer printed and the figure carries. Say the two-second gap is there before someone
  reports it as a discrepancy: the manifest timestamps the process starting, the log line
  timestamps the loss being written.
- *Main-run backward and the 6 h 59 m total, not derivable.* The loss is logged when the
  forward ends, so nothing in the bundle marks the end of a backward pass. The 3 h 48 m 07 s
  and the 6 h 59 m 41 s come from the trainer's own printed timings, repeated in
  `run_manifest.json`'s `note` field — an assertion in a file, not a subtraction.
- *The 7.44 h cadence, not derivable either, and for a duller reason.* It needs three main-run
  rows and the committed snapshot has one. The bundle was assembled on 9 September and has not
  been re-cut since. Anyone quoting 7.44 h from `evidence/` is quoting a file that does not
  contain it; the fix is to re-snapshot the log, and until that is done the number is the
  author's word.
- *Every routing table, fully recomputable.* `evidence/traces/` holds 5,669,776 bytes of raw
  routing records over all 92 MoE layers for five texts. Every figure in `Bulgular.md` §17 and
  §17.2 comes back out of them with `scripts/analyze_trace.py` or a line of `jq`, in seconds,
  with no checkpoint and no GPU. This is the part of the bundle that is genuinely
  independent of trusting anyone.
- *All 93 cosine rows, fully readable.* `sort -t= -k2 -g` over
  `evidence/cmp93_en34_2026-09-06.log` is what falsified the old 0.988 claim, and it will
  falsify the new one too if it is wrong.

## If a number here is wrong

That is the most useful issue this repository can receive, and
`.github/ISSUE_TEMPLATE/verification_report.md` exists for it. Name the row, the source you
opened and what it actually said. A row that turns out to be unsupported gets marked
unsupported here before it gets fixed anywhere else — the table is not a claim that the
numbers are right, it is a claim that every one of them can be traced to something.
