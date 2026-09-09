# Attic — scripts kept for history, not for use

Three scripts in this repository belong to directions the project took and then left. They
are kept because deleting them would hide what was tried, and because a reader who finds
them in the git history should be able to see what they were for without having to
reconstruct it. Nothing in the engine, the tests, the quickstart or the CI workflows calls
any of them; that was checked with a repository-wide grep, and the only mentions outside
the files themselves are prose in `NOTICE`, `README.md`, `docs/LICENSES.md` and
`Bulgular.md`, listed under each entry below.

They are not maintained. They still carry `/mnt/d/...` defaults from the Windows/WSL
machine the project started on, which no longer exists; they were written against a
GTX 980 Ti that this machine does not have; and two of the three import packages
(`transformers`, `tiktoken`) that the project's install list — now just `numpy>=1.24` and
`torch>=2.3` — deliberately does not pull in. Expect them to fail if you run them, and do
not treat anything they print as a measurement.

## Where the files are

They are here, in `docs/attic/`, moved with `git mv` so each file's history follows it:
`docs/attic/patch_k3.py`, `docs/attic/build_speculative_tree.py`,
`docs/attic/continuous_deep_tree_engine.py`. The headings below name them at that path.
`README.md`, `NOTICE`, `docs/LICENSES.md` and `Bulgular.md` were updated in the same commit.

## `docs/attic/patch_k3.py` — 40 lines, patcher for the reference C engine

**What it did.** Opened `src/cli/k3_run.c` in a sibling checkout of
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) at a hardcoded
`/mnt/d/hamza/...` path and rewrote one function, `mem_available_bytes()`, so that it added
`SwapFree` to `MemAvailable` when reporting how much memory was free. The C engine refused
to start when it thought there was not enough RAM; on the WSL machine, with swap doing the
work, this made it start.

**Why it is out.** Two reasons, and the second is the important one. First, it is a
find-and-replace against an absolute path on a machine that is gone, and it silently does
nothing at all when that path does not exist — the worst failure mode a script can have.
Second, kimi-k3-in-c is now used in this project as an *independent reference*: the
layer-by-layer comparison in `evidence/cmp93_en34_2026-09-06.log` is worth something only
because the C implementation was not adjusted to agree with ours. A script whose whole
purpose is to modify that checkout is the opposite of what a reference needs. LazyLoRA's
own streaming path never leans on swap either — `StreamingConfig` caps resident memory at
4.5 GB precisely so it does not have to.

**Licence note, which survives the move.** This file contains the only code in the
repository copied from another project: `mem_available_bytes()` from kimi-k3-in-c, Apache
2.0, modified here. `NOTICE` and `docs/LICENSES.md` record that attribution and are written
to hold whether or not the file is present ("if `docs/attic/patch_k3.py` is present in the
tree you have"). If the file moves, the path in those two documents and in `README.md`
should move with it; if it is deleted outright, the `NOTICE` entry should say the vendored
function is no longer present rather than disappear silently.

---

## `docs/attic/build_speculative_tree.py` — 116 lines, static speculation tree

**What it did.** Ideas 4 and 5 in `Fikirler.md`: since a single sweep of the 108.8 GB non-expert
trunk costs the same whether you verify one token or fifteen, feed the big model a *tree*
of candidate continuations and let one disk pass verify all of them. This script built that
tree — loaded Kimi K3's tiktoken vocabulary (163,840 entries), tokenised a handful of
candidate continuations, truncated each to a fixed depth, and wrote the branches to JSON
for the inference engine to check in one batch.

**Why it is out.** The candidates are a hardcoded list of five English greeting
continuations (`", how can I help you"`, `"! How are you doing today"`, …). It is a
tokeniser exercise dressed as speculation: it cannot propose anything it was not told in
advance, so the branch it verifies is right only when the prompt happens to be a greeting.
It was superseded by `continuous_deep_tree_engine.py`, which does the same thing with a
real draft model — and that one is in this attic too.

Nothing outside the file references it.

---

## `docs/attic/continuous_deep_tree_engine.py` — 300 lines, neural speculation tree

**What it did.** Ideas 4 and 5 in `Fikirler.md`, done properly. A small draft model
(Qwen2.5-0.5B-Instruct or SmolLM2-360M) sits in GPU VRAM and runs real autoregressive
inference in a background thread while the big model is busy reading the trunk off disk
(the file's own docstring budgets twenty-seven minutes for that pass). Each node
expands into its top-k tokens, branches are scored by cumulative log-probability in a
min-heap, and the live tree is flushed to JSON once a
second; when the big model reports which tokens it accepted, the dead branches are pruned
and generation continues from the new root.

**Why it is out.** The project changed subject. Speculative decoding accelerates
*generation*: it wins by getting more than one accepted token out of one expensive pass.
LazyLoRA is now about out-of-core *training*, and a training step has no such slack — the
forward must visit all 93 layers and the backward must visit them again, and no amount of
guessing about future tokens removes a single layer from either. The measured step is
6 h 59 m 41 s of exactly that work. On top of which, the engine needs `transformers` and a
GPU-resident draft model, neither of which the project installs any more.

**Referenced in prose at `Bulgular.md` §9**, which records the environment it ran in —
PyTorch 2.13.0+cu130, transformers 5.16.1, a GTX 980 Ti with 6 GB of VRAM. Those are the
numbers of the earlier machine, not of the one that produced any measurement in the
measurement note. If the file moves, that section's link should move with it, and the
section is worth marking as history for the same reason this directory exists.

---

## What "kept for history" means here

These files are not deleted, not rewritten and not tested. They stay in the git history
with the commits that created them, so the record of what the project tried stays honest —
including the parts that did not work out. If you want to know what LazyLoRA actually does,
read `README.md` and `docs/measurement_note.md`; if you want to know what it stopped doing,
read this directory.
