# Evidence

The raw measurements behind the claims in the README and in `docs/measurement_note.md`.
Everything here is a file produced by a run on the machine described in the README, copied
in unmodified except that this machine's own paths were replaced with placeholders. Nothing
here was regenerated, rounded or reconstructed for publication.

`SHA256SUMS` covers every file in this directory. Verify with `sha256sum -c SHA256SUMS`
from inside `evidence/`.

## `cmp93_en34_2026-09-06.log` (98 lines)

The layer-by-layer comparison of this engine's forward pass against
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c), on 34 tokens, over all 93
layers. Each line records the cosine similarity, the maximum absolute difference and both
implementations' standard deviations after a layer, plus how many experts that layer read.
Every one of the 93 layers is at cosine 0.9857 or better; the lowest row is 0.985744 at
layer 71, the worst stretch is layers 68-72, and the last layer is 0.999840. Documents
written before this log was published quoted "0.988 at layer 72", which was the minimum of
the nine layers sampled in `Bulgular.md` §17.1, not of the 93; sorting this file is what
corrected it.
The run took 2869 s and read 426.59 GB. Produced by `scripts/compare_with_c_dump.py`
against a dump from the C engine; reproducing it needs the 1.56 TB checkpoint and the C
engine's per-layer dump, so this log is the only form in which most readers will see it.

## `forward_loss_main.jsonl`, `forward_loss_proof.jsonl`

One JSON object per completed **forward** pass: step, loss on the assistant tokens,
perplexity, and a Unix timestamp. These are two snapshots of one rolling log rather than
one file per run: the proof file ends with the five steps of the proof-of-learning run of
8-9 September 2026 (`Bulgular.md` §18), and the main file is the same log after the run
that started on 9 September appended to it.

Read the timestamps for what they are. Each marks the moment a forward pass ended, because
that is when the loss is written. So the proof run's step-to-step intervals, and the main
run's forward duration, are honest subtractions of two lines. The main run's backward pass
(3 h 48 m 07 s) and its 6 h 59 m 41 s total are **not** in this file at all; they come from
the trainer's own printed timings, and nothing here lets you check them.

## `run_manifest.json`

What the watchdog believes is running: the data file, the step count, the hyper-parameters,
the start time and the measured step time. Paths to this machine are replaced by
placeholders.

## `traces/`

Five expert-routing traces, one directory each, recorded over all 92 MoE layers by
`scripts/measure_routing.py`. Format, exactly, in `lazy_lora/monitor/trace.py`:

- `trace.bin` - a sequence of records, each `int32 layer, int32 N, int32 K`, then
  `int16[N*K]` expert ids and `float16[N*K]` combining weights, little-endian.
- `trace.json` - the manifest: the text, its token ids, per-layer timings and byte counts,
  peak resident set, thread count, and the layers recorded.
- `analysis.json`, `analysis.md` - the statistics `scripts/analyze_trace.py` computed from
  the trace: unique experts per layer, entropy, top-100 share, consecutive-token overlap.

The five texts are: a paragraph about a morning in a coastal town, written for this study
and rendered in Turkish (264 tokens), English (159) and Chinese (111) with the same
meaning; a Turkish news-style paragraph about hazelnut production statistics, also written
for this study rather than taken from a publication; and a short Python snippet. Because
routing is causal, one trace yields the unique-expert curve for every prefix length, which
is what the concentration and locality numbers in the measurement note are computed from.

Read one with `read_trace` from `lazy_lora/monitor/trace.py`, or with NumPy directly:

```python
import numpy as np, struct
with open("evidence/traces/tr_paragraph_L93_2026-09-06/trace.bin", "rb") as f:
    blob = f.read()
off = 0
while off < len(blob):
    layer, n, k = struct.unpack_from("<iii", blob, off); off += 12
    ids = np.frombuffer(blob, np.int16, n * k, off).reshape(n, k); off += n * k * 2
    w = np.frombuffer(blob, np.float16, n * k, off).reshape(n, k); off += n * k * 2
    print(layer, ids.shape, len(np.unique(ids)))
```

## What is not here

The 1.56 TB checkpoint, the C engine's per-layer dump, the packed NVMe trunk and the
training checkpoints (1.8 GB each). The finite-difference verification writes its result to
the terminal rather than to a file, so `scripts/verify_backward.py` has to be re-run on the
real checkpoint to reproduce those numbers; the values quoted in the README come from
`DEVAM.md` §11, which was written as those runs finished.

## Licence

The routing arrays themselves - every `trace.bin` - are released under CC0-1.0, because a
record of which expert a router selected is a functional measurement of a computation
rather than an authored work, and a share-alike condition on it would only get in the way
of the cache and scheduling research the file exists to enable. Everything else here (the
manifests, the analyses, the comparison log, the loss files, this page) is CC BY 4.0, like
the rest of the project's documentation. `docs/LICENSES.md` sets out the reasoning and the
parts of it that are not legally settled.
