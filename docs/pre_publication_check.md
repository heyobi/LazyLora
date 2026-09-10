# What was checked before this repository was made public

What was verified before this repository went public, what the checking found wrong, and what is still wrong today. The middle section is the one worth your time: about twenty-five numbers and sentences in this repository that were incorrect, what each of them used to say, and what settled it — a minimum cosine that was the minimum of nine sampled layers rather than of all 93, hardware labels left over from a Windows machine that no longer exists, a step time from one run quoted for another, a read throughput that was the disk enclosure's own benchmark rather than a rate this engine achieves. None of those were found by a reader, which is the only reason they can be listed in one place instead of in the issue tracker. The third section is the standing list of what is still broken: it is not a to-do list kept for tidiness, it is the part of this document that stays true after the launch, and nothing is removed from it because a post went out.

*Last checked against the working tree on 10 September 2026. Every tick below was re-verified
on that date rather than carried forward from an earlier pass; where an item had gone stale
it was corrected rather than re-ticked. Numbers quoted here are the ones in
[`numbers.md`](numbers.md); where this document and that table disagree, that table names the
source and the source settles it.*

**Publication happened. The repository became public on 10 September 2026**, at
<https://github.com/heyobi/LazyLora>, with GitHub Pages serving `docs/` from `main`. This
document is sealed on that date, which changes what each section is for:

- **Sections 1 and 2 are closed records.** They say what was checked before the repository
  was public and what the checking found wrong, as of the moment it went public. They are
  not edited again — not to add a later fix, not to append a defect found afterwards, and
  above all not to soften a row in section 2 once somebody has read it. A checklist that
  keeps being improved after the fact stops being evidence of anything.
- **Section 3 is the living part**, and it still is not tidied when a post goes out. An item
  leaves it only by being fixed, and when one does it moves to
  [Closed since publication](#closed-since-publication) below — with the date, the commit and
  the check that settles it — rather than being deleted. What was wrong is the most valuable
  thing in this file; nothing here is ever quietly removed.
- Anything found wrong **after** 10 September 2026 belongs in section 3 or in the issue
  tracker, never in sections 1 and 2.

---

## 1. Checked before publication

*Closed record, sealed 10 September 2026. Every tick was verified on the day the repository
became public and is not re-ticked, amended or extended afterwards.*

### Legal and provenance

- [x] `LICENSE` at the repository root — Apache-2.0.
- [x] `NOTICE`: Kimi K3 is open-weight under Moonshot's custom licence, not Apache — any
      published adapter inherits terms from it; `atasoglu/databricks-dolly-15k-tr`
      CC BY-SA 3.0; Wikipedia CC BY-SA 4.0; `kimi-k3-in-c` credited, including for the op
      fixtures now vendored under `tests/fixtures/ops/`.
- [x] `CITATION.cff`.
- [x] `docs/LICENSES.md`, including the reasoning about what a routing trace is and why it
      can be released under CC BY 4.0.
- [x] The five traced texts were written for this study, including the Turkish news-style
      paragraph, which is about hazelnut production statistics and is **not** taken from any
      publication. An earlier warning in `scripts/export_traces.py` that called them
      copyrighted news text was written without knowing their provenance and was wrong.

### The evidence bundle

Every tick here means the file exists, is correct, **and is in git** — which is the change
since the first version of this checklist, where it was only true of the working tree.
`git ls-files evidence | wc -l` returns 26.

- [x] `evidence/` committed: 6.2 MB, 25 files under `SHA256SUMS` — 26 tracked, the extra one
      being the checksum file itself, which is worth knowing before someone counts — and a
      `README.md` describing every file and, just as importantly, what is *not* there.
- [x] `evidence/cmp93_en34_2026-09-06.log` — the full 98-line comparison against
      `kimi-k3-in-c`: all 93 layer rows with cosine, maximum absolute difference, both
      engines' standard deviations and the expert count, plus the 34 token ids, 2869 s and
      426.59 GB at the bottom.
- [x] `evidence/traces/` — all five expert-routing traces over 92 MoE layers
      (`code_python`, `en_paragraph`, `tr_news`, `tr_paragraph`, `zh_paragraph`; each with
      `trace.bin`, `trace.json`, `analysis.json`, `analysis.md`).
- [x] `evidence/forward_loss_main.jsonl`, `evidence/forward_loss_proof.jsonl` — the raw
      per-step losses with Unix timestamps, which is where every step timing in the
      documentation comes from. They are two snapshots of one append-only file, not two
      files; read the note in [`numbers.md`](numbers.md) before pointing anyone at them.
- [x] `evidence/run_manifest.json` — the live run's manifest, with this machine's paths
      replaced by placeholders.
- [x] Nothing was regenerated for release; the only edit anywhere in the bundle is that
      local filesystem paths became placeholders.

### The external reference

- [x] Fifteen op fixtures vendored at `tests/fixtures/ops/` (8.8 MB under their own
      `SHA256SUMS`), copied unmodified from `kimi-k3-in-c` at upstream commit
      `c223f490047e93600f05fbdab06b9742d2b8ef08`, Apache-2.0, with attribution in
      `tests/fixtures/ops/README.md` and in `NOTICE`. Before this, the only external ground
      truth in the project ran solely for someone who had also cloned the C repository.
- [x] The fixture comparison therefore runs in continuous integration on every push, on a
      machine neither implementation's author controls: seven of the eight ops at 1e-5
      absolute / 1e-4 relative, the latent MoE block at 2e-4 absolute with cosine 1.000000.
- [x] The fixture test fails loudly instead of skipping when `LAZYLORA_REF_FIXTURES` is set
      but points at nothing.

### Packaging and continuous integration

- [x] `pyproject.toml`: two runtime dependencies (`numpy>=1.24`, `torch>=2.3`), `[data]` and
      `[plot]` extras, explicit package list, `mxfp4_gemm.c` and `build.sh` shipped as
      package data.
- [x] `requirements.txt` is no longer a pip freeze. It is the same two packages plus a header
      pointing at the CPU wheel index and at the extras, so `pip install -r requirements.txt`
      and `pip install -e .` are equivalent.
- [x] `.github/workflows/quickstart.yml` runs `scripts/quickstart.sh --fast` on
      `ubuntu-latest` — **and it has now run.** Its first execution was the quickstart's first
      execution anywhere. Five of seven steps passed on the first attempt; the whole episode,
      including the check that failed and why, is `Bulgular.md` §20.
- [x] `.github/ISSUE_TEMPLATE/bug_report.md` and `verification_report.md` — the second one is
      the issue this project most wants to receive.
- [x] `CONTRIBUTING.md`.

### Figures

- [x] Square cards `docs/figures/proof_loss_{tr,en}.png`: the titles say a **LoRA adapter**
      was trained, not that a model learned; the proof-run step reads "5,5-5,8 sa / 5.5-5.8 h"
      and the memory line "4,5-4,7 GB" resident rather than a single "peak". The LinkedIn and
      short-form drafts are written against these exact figures and quote both numbers the
      same way; if a card is regenerated, re-read those files.
- [x] `docs/figures/run_terminal.svg` regenerated from main-run step 1 alone — no splice of
      the proof run's loss and pace, real trainer line formats only, forward 3 h 11 m 34 s at
      123.6 s/layer, loss 0.9091, backward 3 h 48 m 07 s at 147.2 s/layer, 6 h 59 m 41 s
      total, no checkpoint line because this run saves every fifth step, and a caption inside
      the image saying which three timestamps are measured and that the layer times between
      them are interpolated at the measured pace.
- [x] `docs/figures/social_preview.png` for the GitHub repository card; topics set.

### Monitor

- [x] `lazy_lora/monitor/dashboard.py` no longer prints hardware labels left over from the
      project's first machine: "GPU VRAM (GTX 980 Ti)" on a machine with a GTX 1050,
      "SYSTEM RAM (WSL/Host)", "Disk I/O (D: HDD)" and the "C: DRIVE SAFETY GUARD" line are
      now "GPU VRAM", "SYSTEM RAM", "DISK READ (model)" and "SYSTEM DISK FREE", and
      `metrics.py` probes `/` unless `/mnt/c` actually exists. No screenshot or transcript in
      any document or post may show the old labels.

### Git history and pre-registration

- [x] **The history rewrite is done.** The two AI-session database files and every tracked
      bytecode file are gone from every commit, not merely out of the index:
      `git log --all --diff-filter=A --name-only --format='' | grep -c '__pycache__'` returns
      **0**. An earlier version of this checklist listed 29 `.pyc` files as still in history;
      that is no longer true and the item is closed, not pending.
- [x] The annotated tag `preregistration-2026-09-08` exists on commit `6605306` and is
      pushed. It gives a second timestamp; it is **not** an independent one, and every
      document that mentions it says so.
- [x] The pre-registration sentence is in `README.md`: the threshold was committed
      8 September 2026 at 07:54:49 (commit `6605306`), 29 hours before the run started on
      9 September at 12:53:33 (`evidence/run_manifest.json`, `started=1788947613.66`).
- [x] The weakness stated in the same breath, in every draft: git dates come from this
      laptop's clock, the repository was private until launch, and the tag is this account's
      too. Two timestamps, one machine, one operator — a pre-commitment, not an independent
      witness. Nowhere does "pre-registered" do work it has not earned.

### Repository hygiene

- [x] `Gorev.txt` and `PlanVeGorev.txt` are no longer at the repository root. They are
      `docs/origin/gorev.txt` and `docs/origin/plan_ve_gorev.txt`, under a `README.md` that
      says what they are and that they are from the first machine. `k3_run.json` is gone.
- [x] Three abandoned scripts are in `docs/attic/` under a README that says why each was
      abandoned and that they are frozen.

---

## 2. Corrected in the checking

*Closed record, sealed 10 September 2026, for the same reason as section 1: a list of one's
own errors is worth something only if it stops growing and shrinking after publication. Later
corrections go to section 3 and then to* Closed since publication, *not into this table.*

Every row is a thing this repository asserted and that turned out to be wrong. None of them
was found by a reader.

| What it said | What it says now | What settled it |
|---|---|---|
| "cosine ≥ 0.988, minimum at layer 72", in five documents at once | cosine 0.9857 or better across all 93 layers; lowest row **0.985744 at layer 71**; layer 72 is 0.987909, fourth lowest; worst stretch 68-72; the output row 0.999840 was always right | Publishing `evidence/cmp93_en34_2026-09-06.log` and sorting it. The old figure was the lowest of the **nine** layers spot-checked in `Bulgular.md` §17.1, not of the 93. This is the correction the evidence bundle forced, and the reason for the whole bundle: it turned a claim nobody could check into one anybody can falsify with `sort` |
| "5.7 h per 1024-token step" | proof run 5.5-5.8 h on 1082 tokens across two sequences; main run step 1 **6 h 59 m** at 1024 tokens (forward 3 h 11 m, backward 3 h 48 m). Both attributed, never merged | Timestamp subtraction in `evidence/forward_loss_proof.jsonl` against the trainer's printed step-1 timings |
| "about 7 h a step, 29 days, ends 8-9 October" | **7.44 h** a step — the mean of the 7.26 h and 7.62 h intervals between the main run's first three steps — so about **31 days** and around **9-11 October 2026** | Two more steps of the live run. 6 h 59 m was one step measured on its own; the cadence is what the duration follows from. Some documents still carry the old pair; `numbers.md` settles it |
| "ends ~3 October" / "early October" | see above; no date earlier than 9 October is defensible | the same |
| "relative error ≤ 2e-3" for the backward pass | worst **9.1e-3**, at layer 1, in two directions whose analytic derivative is about 3e-4; 3.6e-3 on the MLA layer; ≤ 2.0e-3 elsewhere. The 9.1e-3 leads; the 3.6e-3 never stands alone as "the worst" | `DEVAM.md` §11's own table, read instead of summarised |
| Op fixtures "match at 1e-5" | seven of eight at 1e-5 absolute / 1e-4 relative; the composite MoE block at 2e-4 absolute with cosine 1.000000, and the test hardcodes that tolerance | `lazy_lora/tests/test_reference_ops.py`, `abs_tol=2e-4` on the `moe` case; commit `588ec07` is where the tolerance was widened |
| The quickstart's finite-difference check "passes" | it **failed** on its first CI run: layer-3 residual bank, relative error 3.1e-2 on a 2e-2 tolerance. The analytic gradient was right; the step was too small for fp32 against a tensor of norm 60.85. The step is now relative to the perturbed tensor's norm with one Richardson extrapolation, and the same direction agrees to **2.09e-05** | Running it. `Bulgular.md` §20 and `scripts/debug_fd_bank.py`. A second pre-fix run had passed at 1.83e-2 by drawing a luckier random direction — so the check was also *flaky*, and CI caught both the error and its intermittency |
| "disk-bound at ~115 MB/s", "saturates the band" | **110 MB/s aggregate** is the measured end-to-end rate (3,219,659,335,955 bytes after 8 h 06 m 57 s); **61 MB/s** is the effective rate inside one layer sweep (14.5 GB in 238 s), because the reader idles during compute; **115 MB/s** is the USB enclosure's own sequential benchmark and not a rate this engine achieves | The process read counter against `Bulgular.md` §16.5. "Saturates" is gone; the three numbers are never merged |
| "peak memory 6.24 GB" quoted for this run | resident set 4.0-4.7 GB in the main run with swap in use; 4.5-4.7 GB for 27 h in the proof run; 6.24 GB is from an earlier 256-token step and is the highest ever recorded in the project, not this run's peak | `Bulgular.md` §18, `DEVAM.md` §17. The proof run's 27-hour resident set is never put in one sentence with the main run's step time |
| "a 590 MB adapter" as the memory cost | 590 MB of fp32 parameters plus two Adam moments is about **1.8 GB** resident, which is also the checkpoint size | `README.md` §"Adapter", `DEVAM.md` §17 |
| Nothing at all about where the adapter sits | **one rank-16 adapter per layer in the MoE latent space (3584 → 3072 → 3584), shared by all 896 experts of that layer** — not one per expert, which would be about 26 B trainable parameters | `lazy_lora/trainer/lazy_trainer.py:130`. No document said this before, which was worse than a hostile answer would have been |
| "two fixed 1024-token sequences" in the proof run | two packed sequences, **1082 tokens in total**, 528 of them trained answer tokens, packing limit 1024 | `Bulgular.md` §18; also corrected in the figure footers |
| "deeper layers concentrate routing more" | not monotone: about 430 unique experts at layers 1-36, 304 at 37-48, a trough of 243 at 49-60, 338 at 61-72, about 295 at 73-92 | `Bulgular.md` §17, and now checkable by anyone against `evidence/traces/` |
| "~85 % of experts at 1024 tokens" | measured on layers 0-12, the least concentrated layers, and to be read as an upper bound | `Bulgular.md` §16.5 |
| "a 200 GB static hot set saves 22 %" | said as a leave-one-out simulation, with the note that this machine has about 11 GB of NVMe free so only the 15 GB point (2 %) is realisable | `Bulgular.md` §17's simulation table |
| "perplexity 1.9 on Python" | gone. It had no log line behind it | Looking for the log line |
| Cross-language perplexities quoted side by side | EN 5.90 / TR 2.16 with "these perplexities are not comparable across languages", pointing at bits per byte instead | `Bulgular.md` §17; the tokenizer costs Turkish 1.7× the tokens |
| "128 GB NVMe" | 117 GB, of which 108.8 GB is the packed trunk | `df` |
| `requirements.txt` described as an unusable environment freeze, in four documents | the two packages named, and the two install commands stated to be equivalent | Reading the file. `README.md`, `docs/QUICKSTART.md` (both places), `NOTICE` and `docs/LICENSES.md` were all corrected |
| "the traces are not released yet" / "ships with the trace dataset" | neither sentence survives anywhere. What remains is `docs/measurement_note.md` §"Still not released", about the checkpoint, the C dump, the trunk, the training checkpoints and the `profile_1024` companion trace — all of which genuinely are not here | `git ls-files evidence/traces` |
| `docs/kanit_kosusu.html` said the model "gerçekten öğrendiğini gördük" | rewritten around the adapter, with the memorisation sentence in its place | The proof run proves the loop, not generalisation, and says so |
| `Bulgular.md` said "Tarihte ilk kez" and "%100 doğrulamıştır" | both gone; the machine banner at the top now says which sections are the Ryzen desktop and which the laptop, and gives 1453.74 GiB = 1.56 TB | Reading it with the question "is this exactly as strong as the evidence" |
| The "What you can reproduce" tier table promised more than the tiers deliver | rewritten around the bundle, with the `OK (skipped=8)` trap named on the fixture row | Running the tiers |
| The identical first loss was hedged as a probable coincidence | confirmed as a determinism check, with the method named: both dataset files parsed and their first five records compared, equal by construction because `build_train_set.py` writes the proof file as `picked[:5]` of the same selection | `Bulgular.md` §18.1 |
| The op fixtures were described as an external check anyone could run | they were an external check only for someone who had also cloned `kimi-k3-in-c`. Now vendored, so the comparison runs in CI on every push | `Bulgular.md` §20.3 |
| `scripts/export_traces.py` warned that the traced texts were copyrighted news | corrected; the texts were written for this study | `DEVAM.md`, and the texts themselves, which are in the trace manifests |
| The evidence bundle was described as "in the repository" while `git ls-files evidence/` returned nothing | it is in the repository: 26 tracked paths | `git ls-files` |
| This checklist itself claimed 29 `.pyc` files were still in git history, that the tag did not exist, that `Gorev.txt` was still at the root, and that the quickstart had never been executed | all four were false by the time anyone read them. Corrected above | `git log --all --diff-filter=A --name-only`, `git tag -l`, `git ls-files`, and the CI run recorded in `Bulgular.md` §20 |

---

## 3. Known defects, still open

**This section is not tidied when a post goes out.** Nothing is removed from it because it
has become inconvenient, because a thread is live, or because a launch looks better without
it. An item leaves this list when it is fixed and the fix is checkable, and in no other
circumstance — and when one is fixed it moves down to
[Closed since publication](#closed-since-publication) with the date and the commit, so that
the list of what was wrong keeps growing even as the list of what is wrong shrinks. If you
are reading this after the announcement and the list looks short, that is the thing to be
suspicious of.

### The measurements a reader cannot reproduce

- **The finite-difference checks on the real checkpoint cannot be reproduced by anyone
  else.** The worst-case 9.1e-3 at layer 1, the 3.6e-3 on the MLA layer and the ≤ 2.0e-3
  elsewhere all come from `scripts/verify_backward.py` run against the 1.56 TB checkpoint on
  the author's machine. Two things are wrong with that from a reader's side: the checkpoint
  is 1.56 TB, and **the harness prints to the terminal, so the log is not in the repository
  either.** What is reproducible is the same check on a tiny generated model, which
  `scripts/quickstart.sh` step 7 runs in two minutes — that check is real, and it is not the
  same check.
- **The quickstart's timings are still derived from the code, not measured on the author's
  hardware.** The scripts have now been executed — on a GitHub Actions runner — which
  establishes that they run and that the loop is correct. It does not establish that any
  runtime, memory figure or byte count in `docs/QUICKSTART.md` Part 1 is right for this
  laptop, or for yours. Those figures stay labelled as derived until somebody measures them.
- **The evaluation has not run.** The central question — whether the adapter improved the
  model's Turkish — is unanswered and will stay unanswered until step 100. `scripts/eval_perplexity.py`
  over `tr_news,tr_wiki,en_wiki` needs the step-100 checkpoint, which does not exist yet. Until
  then, everything this repository demonstrates is that the machinery is correct, which is a
  different and smaller claim. The threshold is fixed in advance (`DEVAM.md` §16.1) and a
  negative result will be published as a negative result. Since 10 September 2026 the harness
  is at least exercised on a runner once `.github/workflows/tools.yml` has a green run (it was committed on 10 September and its first run is the test): it exercises `scripts/eval_perplexity.py`,
  `scripts/demo_generate.py` and `scripts/export_traces.py` on a GitHub runner against the
  tiny generated checkpoint, because until then none of the three had been executed anywhere
  and an evaluation script that broke on 9 October would have cost days. That establishes
  that they run. It says nothing about the answer.
- **`eval/results.jsonl`, the eval corpus manifests, `run_proof.sh` and `run_main.sh` are not
  in this repository.** The baseline bits-per-byte figures (0.455 / 0.311 / 0.194) rest on a
  file the reader cannot open. No document may cite one of these paths as though a reader
  could follow it.
- **The final 590 MB adapter is not published.** It is the one artefact that would make the
  central claim independently checkable end to end, and it does not exist until the run
  finishes.

### The pre-registration

- **It rests on this account's own commits and its own tag.** Commit `6605306` is dated by
  this laptop's clock; the annotated tag `preregistration-2026-09-08` was made by the same
  person on the same machine; and the repository was private while both were made, so the
  earliest third-party record of either is GitHub's receipt of the push, which is later than
  both. Two timestamps, one machine, one operator. This is a pre-commitment, not an independent witness, and no
  document may let "pre-registered" do work it has not earned. If a cheap externally
  timestamped anchor turns up before the next run, use it.
- **`DEVAM.md` §16 is headed "6 Eylül 2026, eğitimden önce sabitlendi" while §16.1 is dated
  8 September.** The protocol was fixed on the 6th and the numeric threshold two days later.
  Both dates are in the file, but §16's heading can still be read as claiming the number was
  fixed on the 6th. One sentence in §16 would close this; it is not there yet.

### Numbers in this repository that are not fully sourced

- **The evidence snapshot is one step behind the run.** `evidence/forward_loss_main.jsonl` is
  58 lines and stops at the main run's step 1, so the 7.26 h and 7.62 h intervals — and
  therefore the 7.44 h step and the ~31-day duration that every document now quotes — cannot
  be derived from the bundle. They are read from the live trainer log, which is not in this
  repository. Re-cutting the snapshot fixes it; until then the number is the author's word,
  and [`numbers.md`](numbers.md) marks it as such.
- **The "language signature is confined to layers 1-8" figure is unaudited.** It is in
  `docs/measurement_note.md` §6 with a **†** on it: produced by the analysis tooling and never
  transcribed into the experiment log. It is not in `Bulgular.md`. It is recomputable in
  seconds — `scripts/analyze_trace.py <dir_A> <dir_B> --prefix 55` over two directories in
  `evidence/traces/` — and until somebody does that and writes the result down, the layer-8
  boundary rests on an output nobody kept. The matched-prefix concentration control in the
  same section carries the same mark.
- **Two figures disagree about the fixed finite-difference check.** The docstring of
  `check()` in `scripts/quickstart.sh` says the bank direction agrees to 1.6e-4;
  `Bulgular.md` §20.2 says 2.09e-05. Quote 2.09e-05 with §20.2 named until one
  continuous-integration job settles both.
- **A cross-reference in `scripts/quickstart.sh` points at nothing.** Its comment sends the
  reader to a section of `docs/QUICKSTART.md` called "the bank direction"; that section does
  not exist in that file.

### Code and repository defects

- **`forward_loss.jsonl` still has no `run` field.** `lazy_lora/trainer/lazy_trainer.py`
  appends every completed forward pass to one file with no run identifier, which is why the
  two "loss files" in `evidence/` are one file and why two rows in it read `"step": 1` at
  0.909084. The mock suite writes to the same file, which is why it carries rows at loss
  ≈ 12.0066 and ≈ 6.9078 — ln(163840) and ln(1000) — directly above the real ones, and why
  its first line is `"loss": nan, "perplexity": inf`, which strict JSON parsers reject. Every
  document that sends a reader to that file has to say all of this first.
- **`scripts/watchdog.py` hardcodes `/mnt/nvme` and `/mnt/disk2tb`** in its metrics probes.
  `hdd_reconnect.sh`, `train_lazy_lora.sh` and `run_mock_tests.sh` have their `LAZYLORA_*`
  fallbacks; this one does not, so it works on exactly one machine.
- **`.github/workflows/quickstart.yml` still introduces itself as a workflow that has never
  run**, and its comment above the last step still predicts that "Step 1 will report SKIP:
  the op fixtures live in the reference C repository and are not vendored here." Both
  sentences were true when the file was written and neither is true now — the workflow has
  run on every push since 10 September 2026 and the fixtures are committed at
  `tests/fixtures/ops/`. It is the last place in the tree that describes the old
  arrangement: `grep -n 'SKIP' .github/workflows/quickstart.yml` is the check, and an empty
  result closes this item.

### Closed since publication

Items that were open in section 3 and have since been fixed. They are moved here rather than
deleted, with what was wrong stated first and the fix second, because a reader who wants to
know how careful this project is learns more from the defect than from its absence.

- **`README.md` linked the Turkish walkthrough to a `claude.ai` artifact URL** that resolved
  for one person, in a repository where every other reference points at something a reader
  can open. **Resolved 10 September 2026** (commit `a1b694b`, "Point the Turkish results page
  at its published address"): GitHub Pages serves `docs/` from `main`, the page is live at
  <https://heyobi.github.io/LazyLora/kanit_kosusu.html> and returns 200, and the Turkish
  paragraph in `README.md` links there with the source file named beside it. The vendor URL
  did not survive publication: `grep -rn 'claude\.ai' .` now matches nothing but this
  sentence.
- **The op fixtures were an external check only for a reader who had also cloned
  `kimi-k3-in-c`**, and three places in the tree said so. **Resolved 10 September 2026**
  (commit `be8c458`, "Vendor the reference op fixtures so the external check runs for
  everybody"): the fifteen fixtures are committed at `tests/fixtures/ops/`, 8.8 MB under
  their own `SHA256SUMS`, Apache-2.0, attributed in `NOTICE` and in
  `tests/fixtures/ops/README.md`. Step 1 of the quickstart now passes on a plain clone
  instead of reporting SKIP, so the comparison against an independent implementation runs in
  continuous integration on every push, on a machine neither author controls;
  `README.md`'s reproduction tiers and its quickstart table were corrected to say so, and
  `LAZYLORA_REF_FIXTURES` is now documented as the override for a reader who would rather
  trust their own clone than this copy. What is still wrong is only the workflow file's own
  header comment, which is listed above as its own item.
- **`docs/kanit_kosusu.html` carried three wrong numbers or labels**: the step table's last
  column was headed "Bitiş" where it meant the end of the *forward* pass only — the loss is
  written when the forward finishes and the backward runs about three hours longer; the page
  said the proof steps took "5,6-5,8 saat" where the four measured intervals are 5.78, 5.49,
  5.77 and 5.58 h, so the range is 5,5-5,8 as it is everywhere else; and it still quoted the
  main run's step as "7 saate" after the measured cadence had become 7.44 h. **Resolved
  10 September 2026**, the day the page went live on GitHub Pages: the column now reads
  "İleri geçiş bitti" with a caption explaining why the difference of two such stamps is
  still a whole step, the four intervals are printed individually beside the 5,5-5,8 range,
  and every step figure on the page is now the 7,44 h cadence with the 6 h 59 m step-1 total
  named as what it is. The page also now says, in Turkish, that the 7.26 h and 7.62 h
  intervals were read from the live log and are not in the `evidence/` snapshot — the same
  caveat this file keeps under *Numbers in this repository that are not fully sourced*.
- **`docs/share_post.md` and `docs/announce/linkedin.md` carried the same two paragraphs**,
  with a note telling the reader to keep them in sync — exactly the pattern that let "cosine
  ≥ 0.988, minimum at layer 72" survive in five documents at once. **Resolved before
  publication:** `share_post.md` was merged into `linkedin.md` and deleted. There is now one
  copy.

- **The cosine dip at layers 68-72 is unexplained.** `evidence/cmp93_en34_2026-09-06.log`
  shows the maximum absolute difference jumping to 5.26 at layer 71 against 0.21 at layer 72
  while the cosine falls to 0.985744 and recovers by the output. The likeliest cause is a
  top-16 routing flip on one token, where the two engines pick a different sixteenth expert
  from near-tied scores; a bf16 accumulation difference is the other candidate. The check
  that settles it — per token, per layer, how many of the 16 chosen experts differ between
  the two engines at layers 68-72 — needs the C dump and a forward pass, so it waits for the
  machine. Until then the number is reported, not explained. (Raised by an external review,
  10 September 2026.)
- **The run reads about 3 TB per step from a consumer hard disk, ~295 TB over 100 steps,**
  far beyond any published workload rating for such a drive. Added 10 September: the
  watchdog alerts on any increase of the SMART reallocated, pending, uncorrectable and CRC
  counters (all 0 today) and copies every new checkpoint to the root SSD, a different
  physical disk, keeping the two newest. Not done: a powered USB hub for the enclosure,
  which may be behind the hourly bridge resets; that is a hardware change for the author.
- **History rewrites, for the record.** Before publication the history was rewritten twice.
  The first pass dropped two AI-session database files and 91 tracked bytecode objects. The
  second normalised twenty-eight commits that carried a placeholder author identity and
  stripped a `Claude-Session:` line the tool had appended to forty commit messages — one
  private URL, repeated identically, resolving for one account only. Neither pass touched a
  `Co-Authored-By` trailer. Commit hashes quoted in the documents date from after the second
  rewrite; the dates they carry are the original author dates.
- **A secondary evaluation metric was declared on 10 September 2026** (README, "Secondary
  metric"): masked answer loss on 100 held-out Dolly-tr examples, manifest in `evidence/`.
  Declared a month before the evaluation and before any number exists; it is secondary and
  the primary threshold stands unchanged.

### The standing risk

The list above is what is known to be wrong. The list of what is wrong is longer, because
this repository is one person's work checked by one person, and the corrections in section 2
are all things that survived several passes before somebody opened the file the number was
supposed to have come from. That is the argument for
`.github/ISSUE_TEMPLATE/verification_report.md`: the most valuable thing a reader can do here
is open one file, check one number, and report that it does not say what this repository
claims it says.
