# Release playbook

Drafts for the public announcement of LazyLoRA, plus the order to post them in and the
things that must be true before any of them goes out.

Files here: `hacker_news.md`, `reddit_localllama.md`, `x_thread.md`, `linkedin.md`,
`email_c_reference_author.md`, plus `../share_post.md`, which is the short LinkedIn copy the
first post is actually pasted from. Each is ready to paste, with the numbers checked against
`evidence/` first and `Bulgular.md`, `DEVAM.md` and the run logs after — where a document
and the bundle disagree, the bundle is the measurement and the document is a transcription
of it. Every draft carries a short **Before posting** block naming the facts to re-verify on
the day, because two of them (the step time and the end date) move while the run is going.

This is the most conservative document in the project, on purpose. A reader who follows a
post to the repository will check it within minutes, so an announcement may say less than
the repository proves but never more — and it may never name a file, a log or a dataset
that a stranger cannot open today.

**What changed when `evidence/` landed.** For most of this project's life the honest answer
to "how do I know you did not make this up" was "you cannot check the strongest claims
yet", and every draft here was written around that concession. It is not the answer any
more. The full 93-layer comparison against the C reference is in the repository
(`evidence/cmp93_en34_2026-09-06.log`) and so are all five expert-routing traces
(`evidence/traces/`), which means every routing number in `docs/measurement_note.md`
recomputes from a clone with NumPy, no checkpoint and no special hardware. That is the
project's strongest single fact and it is now the spine of the pitch rather than a
footnote: lead with the two paths a reader can open, then state what is still unreproducible.
The rule above is unchanged — it just cuts the other way now, because naming a file that
*is* there is the whole point.

---

## 1. The claim

One sentence, and it is the only one that gets made:

> I trained a LoRA adapter on Kimi K3 — 2.78 trillion parameters, 1.56 TB of weights —
> out of core, on one laptop with 7.6 GB of RAM, and I checked the arithmetic layer by
> layer against an independent implementation of the same model.

Turkish:

> 2,78 trilyon parametreli Kimi K3'ün üzerine, 7,6 GB RAM'li tek bir dizüstünde,
> çekirdek-dışı bir LoRA adaptörü eğittim; ileri geçişi bağımsız bir C implementasyonuna
> karşı katman katman doğruladım.

Five sentences that must never appear anywhere, in any language, in any headline,
caption, alt text or reply:

1. **"I trained Kimi K3."** A 590 MB adapter was trained. The 2.78 T base parameters are
   frozen and never change.
2. **"The model learned" / "it got smarter."** The adapter memorised five examples in the
   proof run. That is a test of the mechanism.
3. **"It got better at Turkish."** No evaluation result exists until the main run ends.
   Until then the honest answer to "did it work?" is "the threshold is registered, the
   number is due around 8-9 October, and I will publish it either way."
4. **"First to run K3 on a laptop" / "first to stream layers from disk for training."**
   `kimi-k3-in-c` ran this model on a laptop in August 2026 and is this project's oracle;
   AirLLM shipped layer-streamed LoRA training in September 2026. Both get named above
   the fold, by us, before anyone else does it for us.
5. **"It is in the repository", about anything that is not.** The comparison log and the
   five routing traces *are* now — `evidence/cmp93_en34_2026-09-06.log`,
   `evidence/traces/`, `evidence/forward_loss_{main,proof}.jsonl`,
   `evidence/run_manifest.json`, with `SHA256SUMS` over all of it. What is still **not**:
   the 1.56 TB checkpoint, the C engine's per-layer dump, the packed NVMe trunk, the 1.8 GB
   training checkpoints, and any finite-difference log — that harness prints to the
   terminal, so its four numbers come from `DEVAM.md` §11 and have to be re-run on the real
   checkpoint to reproduce. Say "the log is in `evidence/`" where it is true and "you would
   need the checkpoint" where it is not; do not blur them. Run `git ls-files | grep` before
   writing that a reader can go and look at a file — being caught inventing an artefact
   inside the answer to "how do I know you didn't make this up" is the worst outcome
   available on any of these channels. Run it now, in fact: as of this writing
   `git ls-files evidence/` returns nothing, because the bundle is on disk and not yet in
   git. Every "it is in the repository" in this directory is written for the state after
   that commit, and none of it may be posted before it. §3 lists it first among the
   blocking items.

The defensible novelty, stated as a search result rather than a fact about the world:
*as far as I can find, nobody has published a backward pass through a model this size
inside a single consumer machine.* Invite correction in the same breath.

---

## 2. Two beats

The main run ends around 8-9 October and the evaluation number is the strongest artefact
this project will ever have — positive or negative. Do not spend the one-shot channels
before it exists.

### Beat 1 — proof that the loop is correct (now, ~10-12 September)

**Channels:** LinkedIn (TR + EN, from `../share_post.md` or the longer `linkedin.md`),
X thread. Both are the author's own audience, neither is one-shot, and both can be followed
up by the same account in October without cost.

**Prerequisite:** the repository must already be public when beat 1 goes out, because both
drafts link it. That means the pre-flight checklist's legal, hygiene, claims and
reproducibility sections are done before 10-12 September, and the email to FareedKhan-dev
says "public now" rather than "public in October". If the repository is not ready, beat 1
waits — do not post a link that 404s.

**Message:** the mechanism works, it is verified, and the verification is in the repository
where you can open it. The loss on two fixed sequences fell 0.909 → 0.500 → 0.157. That is
memorisation of five examples, said out loud in the post itself, not in a reply. Both
drafts now end their verification paragraph by naming `evidence/`; that sentence is the
reason to post at all and must not be trimmed for length.

**Not in beat 1:** Hacker News, r/LocalLLaMA top-level post, arXiv, Turkish NLP channels.
HN and r/LocalLLaMA give you one front-page window each; opening it on "the loop is
correct" instead of "here is a pre-registered result" wastes the better story. If a
relevant r/LocalLLaMA or HN thread appears on its own in the meantime, a comment with the
numbers and the repo link is fine and costs nothing.

**Send the email to FareedKhan-dev before beat 1**, not after. The X thread credits
`kimi-k3-in-c` by name; the author of the oracle should hear it from us first. It is a
courtesy that takes ten minutes and buys goodwill for the launch.

### Beat 2 — the launch (when the evaluation number exists)

**Trigger:** step 100 checkpoint written, `eval_perplexity.py --corpus tr_news,tr_wiki,en_wiki`
finished, the result written into `Bulgular.md` and the README *before* anything is
posted. Budget a day for the evaluation itself: six 93-layer forward sweeps, and the news
corpus alone takes about three hours.

**Expected timing:** the run started 9 September 12:53. At the measured 6 h 59 m per
1024-token step, 100 steps is about 29 days, so the checkpoint lands around 8-9 October and
the launch window is roughly 10-17 October. Do not publish a fixed date anywhere; say
"about 29 days at the measured step time" and point at `run_manifest.json` for the live
figure. **Any date earlier than 8 October is wrong**: "~3 October" and "early October" were
computed from the proof run's shorter sequences, which held about 541 tokens each rather
than a full 1024. If any file still carries them, fix it before anything is posted.

**Order on launch day** — one day, in this sequence, so each channel can link the one
before it:

| When | Channel | File |
|---|---|---|
| T-3 days | Email to FareedKhan-dev, if beat 1 did not already go out | `email_c_reference_author.md` |
| T-1 day | Repository made public, `evidence/` and the figures verified on the public clone | checklist below |
| T, 15:00 TRT (08:00 ET), Tue-Thu | Show HN | `hacker_news.md` |
| T + 2 h | r/LocalLLaMA | `reddit_localllama.md` |
| T + 4 h | X thread, quoting the HN link in the last post | `x_thread.md` |
| T + 1 day | LinkedIn, TR and EN, with the result added | `linkedin.md` |
| T + 1 week | Measurement note preprint; the traces it rests on are already public in `evidence/traces/`, a DOI on top of that is a nicety, not a blocker | `docs/measurement_note_draft.md` |
| After the note | Turkish NLP channels (SIGTURK, tdd.ai, Turkish HF circles) | new copy, evaluation-first |

Avoid Friday and the weekend for HN. Be at the keyboard for the four hours after posting;
an unanswered first hour is how a good Show HN dies.

### The before/after transcript

The strongest single moment is when a reader can see the same prompt answered by the base
model and by the adapter. It is also the easiest place in this whole project to make a
claim that cannot be defended. Rules:

- The bits-per-byte number goes **first**, the transcript second, captioned "illustrative,
  not evidence". A hand-picked pair of generations is a screenshot, not a measurement.
- **Never show a before/after on any of the 400 Dolly-tr training examples**, and
  especially not on the five proof-run examples — those are inside the training set and a
  visible improvement on them is memorisation, which is exactly what we already said the
  proof run was.
- Use held-out prompts, say how many you tried and how many you are showing, and show at
  least one where the difference is negligible or worse.
- If the evaluation is negative, the transcript still gets published, with the same
  caption and the honest sentence: the metric did not move, and here is what the outputs
  look like anyway.

---

## 3. Pre-flight checklist

Two lists. The first is what is done, kept because a launch checklist that only shows what
is missing tells you nothing about how far along you are and invites re-doing work. The
second is what is still open, and nothing goes public — not the repository, not a post —
while a line in it that a post depends on is still unticked.

### Done

**Legal and provenance**
- [x] `LICENSE` at the repository root — Apache-2.0.
- [x] `NOTICE`: Kimi K3 is open-weight under Moonshot's custom licence, not Apache — any
      published adapter inherits terms from it; `atasoglu/databricks-dolly-15k-tr`
      CC BY-SA 3.0; Wikipedia CC BY-SA 4.0; `kimi-k3-in-c` credited.
- [x] `CITATION.cff`.
- [x] `docs/LICENSES.md`, including the reasoning about what a routing trace is and why it
      can be released under CC BY 4.0.

**The evidence bundle** — the single biggest change to this playbook. Every tick below means
the file exists and is correct in the working tree. It does **not** mean the file is in git:
`git ls-files evidence/` returns nothing today, and until the orchestrator commits it, every
line in every draft that says "it is in the repository" is a promise rather than a fact. See
the first item under Still open.
- [x] `evidence/` assembled: 6.2 MB, 25 files under `SHA256SUMS` — 26 on disk, the extra one
      being the checksum file itself, which is worth knowing before someone counts — and a
      `README.md` describing every file and, just as importantly, what is *not* there.
- [x] `evidence/cmp93_en34_2026-09-06.log` — the full 98-line comparison against
      `kimi-k3-in-c`: all 93 layer rows with cosine, max absolute difference, both engines'
      standard deviations and the expert count, plus the 34 token ids, 2869 s, 426.59 GB.
- [x] `evidence/traces/` — all five expert-routing traces over 92 MoE layers
      (`code_python`, `en_paragraph`, `tr_news`, `tr_paragraph`, `zh_paragraph`; each with
      `trace.bin`, `trace.json`, `analysis.json`, `analysis.md`).
- [x] `evidence/forward_loss_main.jsonl`, `evidence/forward_loss_proof.jsonl` — the raw
      per-step losses with Unix timestamps, which is where every step timing in the
      documentation comes from. Read the note at the end of §6 before pointing anyone at
      them: they are two snapshots of one append-only file, not two files.
- [x] `evidence/run_manifest.json` — the live run's manifest, with this machine's paths
      replaced by placeholders.
- [x] Provenance stated and not overstated: the five traced texts were written for this
      study, including the Turkish news-style paragraph, which is about hazelnut production
      statistics and is **not** taken from Anadolu Agency, the BBC or any publication.
      Nothing was regenerated for release; the only edit anywhere in the bundle is that
      local filesystem paths became placeholders. Read `evidence/README.md` before writing
      a sentence about any of it.

**Packaging and CI**
- [x] `pyproject.toml`: two runtime dependencies (`numpy>=1.24`, `torch>=2.3`), `[data]`
      and `[plot]` extras, explicit package list, `mxfp4_gemm.c` and `build.sh` shipped as
      package data.
- [x] `requirements.txt` is no longer a pip freeze. It is the same two packages plus a
      header pointing at the CPU wheel index and at the extras, so
      `pip install -r requirements.txt` and `pip install -e .` are equivalent.
- [x] `.github/workflows/quickstart.yml` — runs `scripts/quickstart.sh --fast` on
      `ubuntu-latest`. It has never run either; its first run on a public runner *is* the
      quickstart's first execution anywhere, which is why the file says so in its first line.
- [x] `.github/ISSUE_TEMPLATE/bug_report.md` and `verification_report.md` — the second one
      is the issue this project most wants to receive.
- [x] `CONTRIBUTING.md`.

**Figures**
- [x] Square cards `docs/figures/proof_loss_{tr,en}.png`: titles say a **LoRA adapter** was
      trained, not that a model learned; the proof-run step reads "5,5-5,8 sa / 5.5-5.8 h"
      and the memory line "4,5-4,7 GB" resident rather than a single "peak". `linkedin.md`
      and `../share_post.md` are written against these exact figures and quote both numbers
      the same way; if a card is ever regenerated, re-read both files.
- [x] `docs/figures/run_terminal.svg` regenerated from main-run step 1 alone — no splice of
      the proof run's loss and pace, real trainer line formats only, forward 3 h 11 m 34 s
      at 123.6 s/layer, loss 0.9091, backward 3 h 48 m 07 s at 147.2 s/layer, 6 h 59 m 41 s
      total, no checkpoint line because this run saves every fifth step, and a caption
      inside the image saying which three timestamps are measured and that the layer times
      between them are interpolated at the measured pace.
- [x] `docs/figures/social_preview.png` for the GitHub repository card.
- [x] GitHub topics set on the repository.

**Monitor**
- [x] `lazy_lora/monitor/dashboard.py` no longer prints hardware labels left over from the
      project's first machine: "GPU VRAM (GTX 980 Ti)" on a machine with a GTX 1050,
      "SYSTEM RAM (WSL/Host)", "Disk I/O (D: HDD)" and the "C: DRIVE SAFETY GUARD" line are
      now "GPU VRAM", "SYSTEM RAM", "DISK READ (model)" and "SYSTEM DISK FREE", and
      `metrics.py` probes `/` instead of `/mnt/c`. No screenshot or transcript in any
      document or post may show the old labels.

**Claims corrected in the documents** — all of these were verified in the working tree, not
assumed; if a document is edited again, re-check the line rather than trusting the tick.
- [x] **The minimum cosine, everywhere.** Every document used to say "cosine ≥ 0.988,
      minimum at layer 72", which was the lowest of the nine layers spot-checked in
      `Bulgular.md` §17.1, not the lowest of the 93. Sorting the published log settles it:
      **0.985744 at layer 71**, layer 72 is 0.987909 (fourth lowest), the worst stretch is
      layers 68-72 (0.985744-0.989709), the output row 0.999840 was always right. `README.md`,
      `Bulgular.md`, `docs/measurement_note.md`, `evidence/README.md`, every draft in this
      directory and `docs/share_post.md` now say "cosine 0.9857 or better, lowest row
      0.985744 at layer 71". This is the one correction the evidence bundle forced on the
      project, and it is worth remembering why: publishing the log turned a claim nobody
      could check into a claim anybody can falsify with one `sort`.
- [x] `requirements.txt` described correctly in all four documents that used to call it an
      unusable environment freeze — `README.md`, `docs/QUICKSTART.md` (both places),
      `NOTICE` and `docs/LICENSES.md` now name the two packages and say the two install
      commands are equivalent.
- [x] No "the traces are not released yet" / "ships with the trace dataset" sentence
      survives anywhere. What remains is `docs/measurement_note.md` §"Still not released",
      which is about the checkpoint, the C dump, the trunk, the training checkpoints and the
      `profile_1024` companion trace — all of which genuinely are not here.
- [x] `docs/kanit_kosusu.html` H1 rewritten around the adapter, and the "gerçekten
      öğrendiğini gördük" line replaced by the memorisation sentence.
- [x] Machine banner at the top of `Bulgular.md` (§1-15 the Ryzen desktop, §16 onward the
      laptop, 1453.74 GiB = 1.56 TB); "Tarihte ilk kez" and "%100 doğrulamıştır" gone.
- [x] Cross-language perplexities carry the tokenizer caveat and the Python figure, which
      had no log line behind it, is gone. `README.md` now gives EN 5.90 / TR 2.16 with
      "these perplexities are not comparable across languages" and points at bits per byte.
- [x] "~85 % of experts at 1024 tokens" labelled as measured on layers 0-12, the least
      concentrated layers, and read as an upper bound.
- [x] "a 200 GB static hot set saves 22 %" said as a leave-one-out simulation, with the note
      that this machine has about 11 GB of NVMe free so only the 15 GB point is realisable.
- [x] The fixture test fails loudly instead of skipping when `LAZYLORA_REF_FIXTURES` is set
      but points at nothing.
- [x] The "What you can reproduce" tier table in `README.md` rewritten around the bundle,
      with the `OK (skipped=8)` trap named on the fixture row.
- [x] "5.7 h per 1024-token step" → proof run 5.5-5.8 h on 1082 tokens across two
      sequences; main run **6 h 59 m** measured on step 1 (forward 3 h 11 m + backward
      3 h 48 m) at 1024 tokens. Both attributed, never merged.
- [x] "ends ~3 October" / "early October" → "about 29 days at the measured step time,
      around 8-9 October". No date earlier than 8 October anywhere.
- [x] Backward "relative error ≤ 2e-3" → worst **9.1e-3**, at layer 1, in two directions
      whose analytic derivative is ~3e-4; 3.6e-3 on the MLA layer; ≤ 2.0e-3 elsewhere.
      The 9.1e-3 leads, the 3.6e-3 never stands alone as "the worst".
- [x] Op fixtures "1e-5" → seven of eight at 1e-5 abs / 1e-4 rel, MoE block at 2e-4
      (cosine 1.000000).
- [x] "disk-bound at ~115 MB/s" and "saturates the band" → the 115 MB/s is the USB
      enclosure's own sequential benchmark, not an achieved aggregate. What is **measured**
      is 110 MB/s aggregate across the USB disk (routed experts) and the NVMe trunk
      (non-expert weights): 3,219,659,335,955 bytes through `read()` after 8 h 06 m 57 s of
      the main run. Inside a single MoE layer sweep the effective rate is 61 MB/s (14.5 GB
      in 238 s), because the reader thread idles during compute. Say which is which every
      time; "saturates" is gone.
- [x] Resident set → 4.0-4.7 GB with swap in use, main run. The 6.24 GB is from an earlier
      256-token step and is the highest ever recorded in the project; it is not the current
      peak and the 4.7 GB is not a peak either. Never merge the proof run's 27-hour
      resident set with the main run's step time in one sentence.
- [x] Adapter memory → "590 MB adapter" alone understates what is resident: fp32 parameters
      plus two Adam moments is about 1.8 GB, the size of a checkpoint.
- [x] Adapter placement → one rank-16 adapter per layer in the MoE latent space
      (3584 → 3072 → 3584), shared by all 896 experts of that layer, stated in every draft
      here and in `docs/share_post.md`. No document said it before, which was worse than a
      hostile answer.
- [x] "two fixed 1024-token sequences" → two packed sequences, 1082 tokens in total, 528
      of them trained answer tokens, packing limit 1024. Also in the figure footers.
- [x] Routing concentration → "deeper layers concentrate more" is wrong: ~430 unique
      experts at layers 1-36, a trough of 243 at layers 49-60, a mild widening to ~295 at
      layers 73-92. Not monotone. Now checkable by anyone against `evidence/traces/`.
- [x] The identical first loss confirmed rather than hedged — see the end of §6.

### Still open

Nothing in this list is a document that a post quotes; those are all above. What is left is
the repository going public, the history that goes public with it, one code path, and the
artefacts that do not exist until the run ends. A post is not blocked by anything here
except the first group, which it is blocked by absolutely: every draft in this directory
ends on a link.

**Blocking, because a post either 404s or ships something that cannot be taken back**
- [ ] **`evidence/` and `docs/announce/` committed.** Neither is tracked yet —
      `git ls-files evidence/` and `git ls-files docs/announce/` both return nothing. The
      bundle is assembled and correct on disk, which is not the same thing as being in the
      repository, and every draft in this directory is built on a sentence that is only true
      once it is. This is the first item on the list for that reason.
- [ ] **Repository flipped from private to public**, and `git ls-files evidence/` re-checked
      on the public clone before the first post. Every draft here points a reader at a
      specific path inside `evidence/`; a private repository turns the strongest paragraph in
      each of them into the most embarrassing one.
- [ ] **History rewritten first.** 29 `__pycache__/*.pyc` files are out of the index but
      still in the history — `git log --all --name-only | grep -c pycache` finds 91 touches
      — along with this machine's absolute paths in older commits. Once the repository is
      public, history is public; this is the last moment it can be cleaned. The orchestrator
      does the rewrite; nothing else in this checklist may be committed after it without
      re-checking that `evidence/`, `LICENSE` and the pre-registration commit survived.
- [ ] The annotated tag on `85af2a8` created and pushed with the repository. It does not
      exist yet (`git tag -l` is empty) and it has to survive the history rewrite, so make
      it after the rewrite, not before. `hacker_news.md` (both the post and the
      pre-registration answer), `x_thread.md` and `reddit_localllama.md` all say the tag is
      there; none of those three may be posted until it is. `README.md` states it
      conditionally and is safe either way.
- [ ] The artifact page linked from the README:
      https://claude.ai/code/artifact/04001f6d-b7cb-4d4a-91c4-97b7a3416fdc

**Repository hygiene**
- [ ] `Gorev.txt`, `PlanVeGorev.txt` and `k3_run.json` are still tracked at the repository
      root. Move them into `docs/legacy/` with a one-line note saying they are from the
      first machine.
- [ ] `scripts/watchdog.py` still hardcodes `/mnt/nvme` and `/mnt/disk2tb` in its metrics
      probes. `hdd_reconnect.sh`, `train_lazy_lora.sh` and `run_mock_tests.sh` have their
      `LAZYLORA_*` fallbacks; this one does not.
- [ ] `docs/kanit_kosusu.html`: the step-table column is still headed "Bitiş" where it means
      the end of the *forward* pass — the loss is written when the forward finishes and the
      backward runs about three hours longer. The same page's caption says the proof steps
      took "5,6-5,8 saat"; the four measured intervals are 5.78, 5.49, 5.77 and 5.58 h, so
      the range is 5,5-5,8 like everywhere else.

**Reproducibility**
- [ ] **The whole quickstart run once on a machine that is not this laptop, from a fresh
      clone.** `scripts/quickstart.sh`, `make_tiny_model.py`, `demo_generate.py` and
      `export_traces.py` have never been executed — they were written by reading the engine
      while the machine was occupied — so every runtime, memory figure and expected output
      in their documentation is derived from the code, not measured, and is labelled as such
      until somebody runs them. The CI workflow will be the first execution if nobody beats
      it there. Every draft in this directory says this out loud; none of them may stop
      saying it until it is done.
- [ ] A `run` field in every line `forward_loss.jsonl` gets from here on, and the mock suite
      writing somewhere else. Two consequences are live in the published bundle and every
      draft that sends a reader to those files has to say both: the file carries mock rows
      at loss ≈ 12.0066 and ≈ 6.9078 — ln(163840) and ln(1000) — directly above the real
      ones, and its first line is `"loss": nan, "perplexity": inf`, which strict JSON
      parsers reject. See the end of §6 for the third and larger consequence.

**Artefacts still outside the repository**
- [ ] The final 590 MB adapter published to Hugging Face. It is the one artefact that makes
      the central claim independently checkable end to end, and it does not exist until the
      run finishes.
- [ ] The evaluation itself: step 100 checkpoint, `eval_perplexity.py` over
      `tr_news,tr_wiki,en_wiki`, the result written into `Bulgular.md` and the README before
      a word is posted.
- [ ] `eval/results.jsonl` and the eval corpus manifests, `run_proof.sh`, `run_main.sh` —
      none of these paths exist yet, so no draft may name one.

**Pre-registration, stated so a sceptic accepts it and no further**
- [x] The sentence is in `README.md`: the threshold was committed 8 September 2026 at
      07:54:49 (commit `85af2a8`), 29 hours before the run started on 9 September at
      12:53:33 (`run_manifest.json`, `started=1788947613`). Every draft here carries the
      short version of it.
- [x] The weakness stated in the same breath, in every draft: git dates come from this
      laptop's clock, the repository was private until launch, and the tag will be ours too.
      Two timestamps, one machine, one operator — a pre-commitment, not an independent
      witness. Nowhere does "pre-registered" do work it has not earned. If a cheap externally
      timestamped anchor turns up before the next run, use it.
- [ ] `DEVAM.md` §16 says "6 Eylül 2026, eğitimden önce sabitlendi" and §16.1 "Taban ölçümü
      ve önceden ilan edilen eşik (8 Eylül 2026)", which is now honest about the two dates.
      What is left is one sentence in §16 saying the threshold arrived two days after the
      protocol, so the section cannot be read as claiming the number was fixed on the 6th.
---

## 4. The hardest questions

These will be asked within the first ten comments on every channel. Answer them in the
post where possible, so the thread starts past them.

### "How do I know you did not make this up?"

It used to be the question with no good answer. It now has the best answer in the project,
and the full version lives in `hacker_news.md` under Prepared replies — two paths a reader
can open, in this order, and then the limits:

> `evidence/cmp93_en34_2026-09-06.log` — all 93 comparison rows, so the summary above is
> checkable line by line. Then `python scripts/analyze_trace.py
> evidence/traces/tr_paragraph_L93_2026-09-06` — every routing number I publish, recomputed
> from the repository with NumPy in seconds, no checkpoint and no GPU. That second one is
> recomputation rather than reading, which makes it the strongest thing here. Step timings
> come out of `evidence/forward_loss_*.jsonl` by subtracting two Unix timestamps.
> `sha256sum -c SHA256SUMS` covers the bundle.
>
> Not checkable without the hardware, and I would rather say it than have you find it: the
> 1.56 TB checkpoint and the C engine's dump are not published, so the comparison can be
> read but not regenerated; the finite-difference harness prints to the terminal, so those
> four numbers have no log behind them and want a re-run on the real checkpoint; the
> quickstart has never been executed; there is no evaluation result yet; and the
> pre-registration is a commit timestamped by this laptop's own clock.

Three things to keep straight while answering it. First, "in the repository" and
"reproducible" are different claims — the log is the former, the traces are both. Second,
publishing the log means a reader can now falsify a summary of it, so any number quoted
about that file has to survive a `sort` (see the minimum-cosine line in §3). Third, the
invitation to subtract timestamps will be taken up, so know in advance which subtractions
work: the proof run's step time and the main run's forward do, the main run's backward and
its 6 h 59 m total do not, and the two loss files are one file plus a row. All three are
spelled out at the end of §6, and the reply for the last of them is in `hacker_news.md`.

### "How do you know your forward pass is right?"

> I do not trust it; I check it against something I did not write. `kimi-k3-in-c` is an
> independent C99 implementation of this model by FareedKhan-dev. It has a per-layer dump
> hook, so I replay its dump through my engine and compare layer by layer: all 93 layers
> agree at cosine 0.9857 or better — the lowest row is 0.985744 at layer 71 — and 0.999840
> at the output. The whole log is in the repository at
> `evidence/cmp93_en34_2026-09-06.log`, so you do not have to take the summary: sort it and
> find the worst row yourself.
> Below that, eight op-level fixtures — seven match at 1e-5 absolute / 1e-4 relative, the
> MoE block at 2e-4 absolute with cosine 1.000000, which is the MXFP4 decode path's own
> rounding. The two engines also independently reproduce the same massive activations in
> the last two MLA layers: one token per text at a residual norm of 1-2 × 10⁴ against a
> median of 78. That is a distinctive enough fingerprint that agreement is unlikely to be
> coincidence.
>
> The honest scope: 34 tokens of one English paragraph, with LoRA B zero-initialised so
> the adapter contributes exactly nothing and both engines must agree exactly. It is not
> verified with a trained adapter, at 1024 tokens, or on Turkish or code input. And the
> log is a record, not a recipe: reproducing the comparison rather than reading it needs
> the 1.56 TB checkpoint and a dump from the C engine, neither of which is published.

If they push: yes, a shared misreading of the checkpoint format would fool both engines.
It would not survive the op fixtures, which come from the reference author's own
implementation of each operator, and it would be visible as garbage output. That is the
limit of what cross-implementation agreement can buy, and I would rather say so.

### "How do you know the gradients are right?"

> Central finite differences on real weights, not on a toy. Four representative layers —
> 1 (KDA + MoE, one bank entry), 3 (MLA), 12 (a block boundary), 13 (two bank entries).
> Layers 1 and 3 were swept over all 16 LoRA tensors plus the input and residual-bank
> directions; layers 12 and 13 covered the input and bank directions and that layer's LoRA
> tensors. On 4 tokens, in an fp32 engine with the loss reduced in float64. Worst relative
> error **9.1e-3**, at layer 1, in two directions whose analytic derivative is about 3e-4 —
> at that magnitude the finite difference is the noisier of the two estimates. 3.6e-3 on
> the MLA layer, ≤ 2.0e-3 everywhere else. I quote the 9.1e-3 first, because quoting the
> 3.6e-3 as "the worst" is the cherry-pick a reader will catch by opening DEVAM §11.
>
> Getting a meaningful answer at all needed adaptive epsilon and routing-flip detection:
> nudge a weight too far in a top-16 router and a different expert wins, the loss surface
> steps, and the finite difference measures the step instead of the derivative. Discarding
> those probes is a decision I had to make and it is in the harness where you can see it.
>
> What this is not: a whole-model gradient check. It covers 4 of 93 layers on 4 tokens.
> The end-to-end evidence that the loop is correct is that the loss actually falls on a
> fixed sequence, pass after pass, which is a thing a subtly wrong gradient does not do
> for long. `scripts/verify_backward.py` is in the repository; the log is not, because
> there never was one — the harness prints to the terminal, so these four numbers are
> transcribed from `DEVAM.md` §11 and are the one part of the verification with no
> artefact behind it. Say that before someone asks which file it is in.

### "Is the LoRA per expert or per layer?"

The first question anyone who works on MoE fine-tuning will ask, and no document stated it
before, which is worse than a hostile answer.

> Per layer. On the routed path there is one rank-16 adapter living in K3's MoE latent
> space, 3584 → 3072 → 3584, shared by all 896 experts of that layer —
> `lazy_lora/trainer/lazy_trainer.py:130`. One adapter per expert would be on the order of
> 26 billion trainable parameters — about 320 k per expert × 896 × 92 ≈ 2.6 × 10¹⁰, roughly
> 420 GB of fp32 weights, gradients and Adam moments — which neither this machine nor 400
> examples can support. The consequence is real: the adapter learns a correction that
> applies to whichever expert
> fires, not per-expert specialisation. If routing is as domain-specific as the traces
> suggest, a per-expert or per-group adapter might learn something this one structurally
> cannot — an ablation this hardware cannot run, named as future work rather than waved
> away.

### "Is this just inference offloading with extra steps?"

> No, and the difference is the whole project. Inference reads each weight once and throws
> the activations away. Training has to keep, for every layer, the input the backward pass
> will need, and it has to read the routed experts a second time on the way back — for
> this model that is 896 experts per layer of which top-16 fire per token, so the streaming
> unit has to be the individual 17.5 MB expert, not the decoder layer: one layer's experts
> alone are 15.7 GB, twice the machine's RAM.
>
> Then the gradients have to go somewhere correct. K3 has an attention-residual bank every
> 12 layers, so a gradient computed at layer 84 has to be routed back to the layer that
> wrote the bank entry, across a boundary where the layer that produced it is long gone
> from memory. Layer-boundary activations live in a ring buffer on NVMe and each layer is
> recomputed under autograd when the backward reaches it.
>
> The pieces are known: gradient checkpointing since 2016, layer streaming from AirLLM and
> from `kimi-k3-in-c`, LoRA and QLoRA, fused low-bit kernels from ggml, expert caches from
> the whole MoE serving literature. What I could not find anyone having published is the
> composition at 2.78 T on 7.6 GB of system RAM with the weights on a USB hard disk — and
> the verification that it is arithmetically right. If you know of prior work, send it and
> I will put it in the related-work table.

A useful supporting number when someone says "why not KTransformers or llama.cpp?": the
documented recipe for LoRA SFT of a 1 T MoE (KTransformers + LLaMA-Factory on Kimi K2.5)
wants 2-4 × RTX 4090, an AMX Xeon and about 2 TB of system RAM, and runs at ~45 tok/s.
This is a 2.8× larger model on roughly 260× less RAM with no usable GPU — and it pays
7 hours per step for that. Always quote the cost in the same sentence as the ratio, or it
reads as a speed claim.

---

## 5. If the evaluation is negative

Assume it will be, and say so before it lands. 100 optimizer steps at batch size 1 over
roughly 51,000 trained tokens is a small amount of signal — 100 steps consumes 100 of the
154 packed sequences, so about 260 of the 400 examples are seen once, 0.65 of an epoch.
Publishing that expectation now is what makes the eventual number credible instead of an
excuse; publishing it after a miss is an excuse.

A pre-registered negative result is a better artefact than a marginal positive one, and it
is rarer. It is publishable as-is:

- **Title it plainly.** "…and it did not move the metric" in the title, not in paragraph
  six. Reddit and HN reward this and punish the alternative.
- **Publish the table** — news, TR wiki, EN wiki, before and after — and the diagnostics:
  loss curve over 100 steps, per-layer LoRA gradient norms, and the before/after routing
  trace comparison the protocol already calls for (§16, item 5). "Did the adapter shift
  the routing at all" is an interesting answer whether or not the bits-per-byte moved —
  and the "before" half of it is already public in `evidence/traces/`, so the after-trace
  goes into the same directory in the same format and anyone can diff the two. That is a
  better negative-result artefact than the table.
- **Name the likely causes without picking one:** signal volume (0.65 epoch, batch 1),
  rank 16 on q/v plus the expert projections, lr 5e-4, and instruction data measured with
  a language-modelling metric on news — a known mismatch.
- **Do not quietly re-run with different settings and post that as the result.** If a
  second run happens, it is exploratory and post-hoc, labelled as such, with the
  pre-registered result still in the README above it.

The engineering claim does not depend on the outcome. "The loop is verified and the
adapter moves under gradient descent on this hardware" is true either way, and it is the
claim the launch is built on.

---

## 6. Numbers you may quote

Everything below is sourced. If a number is not on this list, do not put it in a post
without finding the log line first.

| Quantity | Value | Source |
|---|---|---|
| Model | Kimi K3, 2.78 T params, 93 layers (69 KDA + 24 MLA), 896 routed experts/layer, top-16 + 2 shared, MXFP4 | model card, README |
| Checkpoint | 1453.74 GiB = 1.56 TB, 96 safetensors shards, 2 TB USB HDD | Bulgular §2, §10 |
| One layer's experts | 896 × 17.5 MB = 15.7 GB | arithmetic from the above |
| Trunk | 108.8 GB of packed non-expert weights on NVMe | README |
| Machine | i7-7700HQ (4c/8t, AVX2), 7.6 GB RAM, GTX 1050 2 GB, 117 GB NVMe (~11 GB free) | README, status.txt |
| Adapter | 590 MB fp32, rank 16, alpha 32, on q_proj, v_proj and the expert gate/up/down; with its two Adam moments about 1.8 GB resident, which is also the checkpoint size | README, DEVAM §12 |
| Adapter, routed path | **one rank-16 adapter per layer in the MoE latent space (3584 → 3072 → 3584), shared by all 896 experts of that layer** — not one per expert, which would be ~26 B trainable parameters (about 320 k per expert × 896 × 92 ≈ 2.6 × 10¹⁰, roughly 420 GB at 16 bytes apiece) | `lazy_lora/trainer/lazy_trainer.py:130` |
| Evidence bundle | 25 files, 6.2 MB, `SHA256SUMS` over all of it; comparison log, five routing traces, both loss logs, the run manifest. Paths replaced by placeholders, nothing regenerated | `evidence/README.md` |
| Traced texts | tr_paragraph 264 tokens, tr_news 261, code_python 167, en_paragraph 159, zh_paragraph 111; all five written for this study, the news-style one about hazelnut production statistics and **not** from any publication | `evidence/README.md`, each `analysis.md` |
| Forward vs C reference | all 93 layers, cosine 0.9857 or better (lowest row **0.985744 at layer 71**; worst stretch 68-72), 0.999840 at output, 34 tokens, LoRA B = 0, 2869 s, 426.59 GB read. **The whole 98-line log is in the repository** | `evidence/cmp93_en34_2026-09-06.log` |
| Op fixtures | 7 of 8 at 1e-5 abs / 1e-4 rel; MoE block 2e-4 abs, cosine 1.000000 (`test_reference_ops.py` hardcodes `abs_tol=2e-4`; commit 91964c6 widened it) | `test_reference_ops.py`, Bulgular §15.6 |
| Backward | layers 1, 3, 12, 13; 4 tokens; worst **9.1e-3** at layer 1 in two directions whose analytic derivative is ~3e-4; 3.6e-3 on MLA; ≤ 2.0e-3 elsewhere | DEVAM §11, §268-276 |
| Proof run | 5 examples → 2 packed sequences, 1082 tokens total, 528 trained answer tokens (limit 1024), lr 1e-3, warmup 2 | Bulgular §18 |
| Proof losses | A: 0.909 → 0.500 → 0.157; B: 0.521 → 0.193 (0.909084, 0.521090, 0.500335, 0.193009, 0.156585 to six decimals) | `evidence/forward_loss_proof.jsonl`, last five lines |
| Proof step time | 5.5-5.8 h (four measured intervals, mean 5.65 h) | log timestamps |
| **1024-token step** | **6 h 59 m: forward 3 h 11 m (123 s/layer) + backward 3 h 48 m (147 s/layer)** | `run_manifest.json`, step 1 |
| Memory | resident set 4.0-4.7 GB in the main run with swap in use (~2.4 GB); 4.5-4.7 GB held for 27 h in the proof run; 6.24 GB on an earlier 256-token step is the highest ever recorded, not the current peak | Bulgular §18, DEVAM §17, status.txt |
| Read throughput, aggregate | **measured 110 MB/s** across the USB disk (routed experts) and the NVMe trunk (non-expert weights): 3,219,659,335,955 bytes through `read()` after 8 h 06 m 57 s of the main run | process read counters, main run |
| Read throughput, inside a sweep | 61 MB/s effective (14.5 GB per MoE layer in 238 s, layers 0-12); the ~115 MB/s is the enclosure's own sequential benchmark, **not** an achieved aggregate | Bulgular §16.5 |
| Main run duration | 100 steps at about 7 h = about 29 days from 9 September 12:53, so around 8-9 October. No earlier date is defensible | arithmetic on the step time |
| Sweep amortisation | 8× the tokens costs 2× the layer time (128 → 1024 tokens, 122 s → 238 s, layers 0-12) | Bulgular §16.5 |
| Main run | 400 Dolly-tr examples → 154 packed sequences ≤ 1024 tokens (78k trained tokens/epoch), 100 steps at batch 1 = 0.65 epoch, ≈ 260 examples seen once, lr 5e-4 cosine, warmup 5, prompt masked, checkpoint every 5 steps | DEVAM "ŞU AN" |
| Baselines (bits/byte) | TR news 0.455, TR wiki 0.311, EN wiki 0.194 | DEVAM §16.1 |
| Threshold | news ≤ 0.441 (−3 %) **and** EN wiki ≤ 0.198 (+2 % max) | DEVAM §16.1 |
| Pre-registration | commit `85af2a8`, 8 Sep 2026 07:54:49, 29 h before the run (9 Sep 12:53:33) | git log, run_manifest |
| Routing: concentration | a batch touches 43-56 % of the experts a uniform router would; **not monotone with depth** — ~430 unique experts at layers 1-36, a trough of 243 at 49-60, ~295 at 73-92 | Bulgular §17, measurement_note §5; recomputable from `evidence/traces/` |
| Routing: language | cross-language expert-set Jaccard 0.35-0.39, equal to within-language different-content; prose vs Python 0.20-0.21; language signature only in layers 1-8 of 92 | Bulgular §17; recomputable from `evidence/traces/` |
| Routing: locality | consecutive-token Jaccard 0.258 vs 0.009 random; per-layer LRU 62-72 % at 64-128 experts (1.1-2.2 GB) in decoding | Bulgular §17.2; recomputable from `evidence/traces/` |
| Routing: batch union | 42 % of all experts at 128 tokens, 53 % at 256, ~85 % at 1024 — **the 1024 point is layers 0-12 only** | Bulgular §16.5, §17.2; recomputable from `evidence/traces/` |
| Turkish tokenizer tax | 1.7× the tokens and 1.6× the bits per byte of English; routing not more diffuse | Bulgular §17 |
| Massive activations | last two MLA layers, one token per text at residual norm 1-2 × 10⁴ vs median 78; reproduced by the C reference | Bulgular §17 |

**Numbers that are not measurements and must not be quoted as such:** any per-step time
merged across the two runs, and "5.7 h" in particular — it came from the proof run, whose
two packed sequences held about 541 tokens each rather than a full 1024, so it may never be
quoted for the main run; "saturates 115 MB/s", and 115 MB/s as an achieved rate at all;
"perplexity 1.9 on Python"; the LRU points at 16, 32 and 448 experts in the draft note;
"128 GB NVMe" (it is 117 GB); every runtime, memory figure and expected output in the
quickstart documentation, because those scripts have never been executed and the figures
are derived from the code; and anything from `forward_loss.jsonl` at loss ≈ 12.0066 or
≈ 6.9078, which are ln(163840) and ln(1000) from mock and broken-pipeline runs. Those mock
rows are now visible to everyone, in `evidence/forward_loss_main.jsonl`, sitting directly
above the real ones — so every draft that sends a reader to that file also says which lines
are not Kimi K3. Point at the rows below loss 1.

Also not a measurement, and now falsifiable in one command: **"cosine ≥ 0.988, minimum at
layer 72"**. The published log's lowest row is 0.985744 at layer 71. Every document in the
repository has been corrected to 0.9857 / layer 71; nothing may reintroduce the old pair,
including from an old screenshot, an old draft or the nine-row table in `Bulgular.md` §17.1
that it came from.

**The identical first loss — confirmed, and say it before someone finds it.** The main run's
first logged loss is 0.909084, equal to the proof run's first loss to six decimals. This is
no longer a hedge: the first five records of `datasets/dolly_tr_400.jsonl` were parsed and
compared against `datasets/dolly_tr_proof.jsonl` and the parsed records are equal, which is
by construction — `scripts/build_train_set.py` writes the proof file as `picked[:5]` of the
same selection. Both runs start from a zero-initialised adapter, so LoRA B contributes
exactly nothing, the first packed sequence is the same text and the first forward pass is
the same computation. It is a determinism check, not a coincidence, and both `README.md`
and `Bulgular.md` §18.1 should state it that way, naming how it was verified. Disclose in
the same breath that the five proof examples are inside the 400-example training set and
outside every evaluation slice. The two data files themselves are not in the repository —
`build_train_set.py` writes them from the Hugging Face dataset — so a reader who wants to
check this rebuilds them with the same seed rather than opening a committed file. Say
"verified by parsing both files here", not "you can see it in the repo".

**The two loss files are one file, and a reader will diff them.** This is the sharpest thing
the evidence bundle did to this playbook and no draft may go out without an answer to it.
`evidence/forward_loss_proof.jsonl` is 57 lines, `evidence/forward_loss_main.jsonl` is 58,
and the first 57 are byte-identical — `diff <(head -57 evidence/forward_loss_main.jsonl)
evidence/forward_loss_proof.jsonl` prints nothing. They are not two runs' logs. The trainer
appends every completed forward pass to one `forward_loss.jsonl`; the "proof" file is that
file as it stood when the proof run ended, and the "main" file is the same file one row
later, that row being the main run's step 1. Two rows in it read `"step": 1` at loss
0.909084 — the proof run's first pass at `time` 1788859586 and the main run's first pass at
1788959106, 27 h 38 m apart — which is the same identical-first-loss fact as above, except
that now it is sitting in a public file where it looks exactly like a copy-paste until
somebody explains it. Explain it first. The fix in the code is the missing `run` field; the
fix in the announcement is one sentence, and `hacker_news.md` carries the full version under
Prepared replies.

**What the bundle lets a reader derive, and what it does not.** Worth knowing precisely,
because the invitation to "do the subtraction yourself" has to survive being taken up:

- *Proof-run step time, fully derivable.* Consecutive rows are whole steps, forward end to
  forward end: 20813, 19756, 20756 and 20093 s — 5.78, 5.49, 5.77 and 5.58 h, mean 5.65.
  That is where "5.5-5.8 h" comes from and a reader gets it from the file alone.
- *Main-run forward, derivable to two seconds.* `run_manifest.json` has
  `started=1788947613.66` and the last row of `forward_loss_main.jsonl` has
  `time=1788959106`. The difference is 11492 s = **3 h 11 m 32 s**, against the
  3 h 11 m 34 s the trainer printed and the figure carries. Say the two-second gap is there
  before someone reports it as a discrepancy: the manifest timestamps the process starting,
  the log line timestamps the loss being written.
- *Main-run backward and the 6 h 59 m total, not derivable.* The loss is logged when the
  forward ends, so nothing in the bundle marks the end of a backward pass. The 3 h 48 m 07 s
  and the 6 h 59 m 41 s come from the trainer's own printed timings, repeated in
  `run_manifest.json`'s `note` field — which is an assertion in a file, not a subtraction.
  It is a small thing and it is exactly the kind of small thing that costs a thread its
  credibility when a reader finds it before we say it.
