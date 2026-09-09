# Expert-routing traces: `evidence/traces/`

Expert routing decisions recorded from forward passes of **Kimi K3** (Moonshot AI;
2.78 T parameters, 93 layers, 896 routed experts per layer, top-16, 2 shared experts,
MXFP4 expert weights) over all 92 MoE layers, for five texts in three natural languages
and one programming language.

One record answers, for one layer and one token: *which 16 of 896 experts did the router
choose, and with what combining weights.*

**They are in this repository**, under `evidence/traces/`, committed on 9 September 2026 —
5 669 776 bytes of routing records plus manifests and analyses. There is no separate
dataset, no Zenodo record and no DOI; the traces are versioned with the code that produced
them and the note that reports them. Earlier versions of this file were written as a card
for a dataset that would be released elsewhere, and said the traces were on one machine and
nowhere else. That is no longer true and the rest of this file describes what is actually
there.

- **Companion note:** [`docs/measurement_note.md`](../measurement_note.md). Every routing
  number in its Sections 5–9 is recomputable from these files; its Section 12 lists, result
  by result, which command does it and what it costs.
- **The bundle these sit in:** [`evidence/README.md`](../../evidence/README.md) — the
  comparison log, the raw per-step losses, the run manifest, and `SHA256SUMS` over all of
  it. That file describes the bundle; this one documents the trace format. Neither repeats
  the other.
- **Producer:** LazyLoRA — `lazy_lora/monitor/trace.py` (format), `scripts/measure_routing.py`
  (the run), `scripts/analyze_trace.py` (the statistics).
- **Integrity:** `cd evidence && sha256sum -c SHA256SUMS`.

---

## 1. Why this exists

Every published system for running a large MoE on a small machine — expert caches,
prefetchers, hot-set placement, mixed-precision fallbacks, token reordering — is a policy
over the sequence of expert requests a model makes. Evaluating such a policy normally
requires the model: 1.56 TB of Kimi K3 weights, a machine that can hold a layer at a time,
and hours per text. It does not have to. The policy only ever sees the request sequence,
and the request sequence is these files.

With a trace and twenty lines of NumPy you can measure, at 896-expert / 92-layer scale and
on a laptop:

- the hit rate of any cache size and replacement policy, per layer or globally;
- the hit rate of any prefetch policy that depends only on past routing;
- how many distinct experts a batch of N tokens forces you to read (the number that governs
  a training step, and the one the inference literature does not report);
- how much a static residency budget saves, and how well a hot set chosen on one text
  transfers to another;
- expert-usage skew, entropy, and how all of it changes with depth.

What you cannot do with a trace is run the model. No weights, no activations and no
recoverable model parameters are here.

---

## 2. Contents

```
evidence/traces/
  zh_paragraph_L93_2026-09-06/
      trace.bin       654 672 B   routing records, 92 of them          (§3)
      trace.json       15 070 B   manifest: text, ids, per-layer timing (§4)
      analysis.json    27 896 B   per-layer statistics and prefix curves (§5)
      analysis.md       7 473 B   the same, as a table
  en_paragraph_L93_2026-09-06/
      trace.bin       937 296 B
      trace.json      173 271 B   (larger: this run also stored per-token residual norms)
      analysis.json    30 693 B
      analysis.md       7 522 B
  code_python_L93_2026-09-06/
      trace.bin       984 400 B
      trace.json       15 592 B
      analysis.json    30 676 B
      analysis.md       7 532 B
  tr_news_L93_2026-09-06/
      trace.bin     1 537 872 B
      trace.json       16 444 B
      analysis.json    33 422 B
      analysis.md       7 574 B
  tr_paragraph_L93_2026-09-06/
      trace.bin     1 555 536 B
      trace.json      276 606 B   (per-token residual norms, and 264 tokens of them)
      analysis.json    33 432 B
      analysis.md       7 581 B
```

Sizes are the committed files, not estimates. Each `trace.bin` is exactly
`92 · (12 + 64·N)` bytes — 92 records of 16 experts per token (§3) — which is a cheap
integrity check on top of the checksums: 654 672 at N=111, 937 296 at 159, 984 400 at 167,
1 537 872 at 261, 1 555 536 at 264.

`analysis.json` and `analysis.md` are derived: `scripts/analyze_trace.py <dir>` regenerates
them from `trace.bin`. They are committed so that a reader can check a rerun against them
rather than having to trust it.

---

## 3. Binary format (`trace.bin`)

Defined by `lazy_lora/monitor/trace.py` and deliberately trivial, so that NumPy alone can
read it. The file is a bare concatenation of variable-length records with **no header and
no footer**. It is written append-only as the forward pass proceeds and flushed after every
layer, so a crashed run leaves a valid prefix.

One record per recorded layer:

| offset | field | type | count | notes |
|---:|---|---|---:|---|
| 0 | `layer` | `int32` | 1 | layer index as reported by the engine |
| 4 | `N` | `int32` | 1 | tokens in this forward pass |
| 8 | `K` | `int32` | 1 | experts selected per token (16 for Kimi K3) |
| 12 | `expert_ids` | `int16` | `N*K` | row-major `[N, K]`; expert index in `[0, 896)` |
| 12 + 2·N·K | `weights` | `float16` | `N*K` | row-major `[N, K]`; the router's combining weights |

- **Endianness: little-endian throughout.** The three `int32` header fields are packed with
  Python `struct.Struct("<iii")`; the two arrays are written with `ndarray.tobytes()` on a
  little-endian host and are not byte-swapped. On a big-endian machine you must swap.
- **Record length** is `12 + 4·N·K` bytes; all 92 records of one file share the same `N`.
- `expert_ids` is `int16`, which is signed. 896 experts fit comfortably; a negative value
  would mean corruption.
- `weights` is `float16`, as the engine used them. They are *not* renormalised to sum to
  one; `analyze_trace.py` normalises per row before computing the effective-expert count.
- Rows of `expert_ids` are in the router's own top-k order, which is by descending weight.
  Do not assume they are sorted by expert index.
- Records are in execution order, ascending. Layer 0 of Kimi K3 is dense and contributes no
  record, so the recorded layers are 1–92, and every file here has exactly 92 records. The
  authoritative list is `trace.json["layers"]`.

Minimal reader, no dependency on this repository:

```python
import struct, numpy as np

def read_trace_bin(path):
    """-> {layer: (ids int16 [N,K], weights float16 [N,K])}"""
    head = struct.Struct("<iii")
    out = {}
    with open(path, "rb") as f:
        while True:
            b = f.read(head.size)
            if len(b) < head.size:
                break
            layer, n, k = head.unpack(b)
            ids = np.frombuffer(f.read(n * k * 2), dtype="<i2").reshape(n, k)
            w   = np.frombuffer(f.read(n * k * 2), dtype="<f2").reshape(n, k)
            out[layer] = (ids, w)
    return out
```

The repository's own reader is `lazy_lora.monitor.trace.read_trace(trace_dir)`, which
returns `(manifest, layers)` and is what `scripts/analyze_trace.py` uses.

---

## 4. Manifest (`trace.json`)

UTF-8 JSON, rewritten atomically after every layer, so it is valid even mid-run.

| key | type | meaning | present in |
|---|---|---|---|
| `tag` | str | short name of the run, e.g. `tr_paragraph` | all |
| `created` | str | local time the run started, `%Y-%m-%d %H:%M:%S` | all |
| `text`, `ids` | str / list[int] | the input text and its token ids, BOS first | all five — nothing is withheld (§6) |
| `n_tokens` | int | `N`, including BOS | all |
| `layers_requested` | int | how many layers the run was asked for (93) | all |
| `layers` | list | one entry per recorded layer, in execution order (below) | all |
| `compute_dtype` | str | the engine's compute dtype (`torch.bfloat16` for all five) | all |
| `model_dir`, `cpu`, `threads` | str/int | environment; `model_dir` is a placeholder in the committed copies | all |
| `total_seconds`, `bytes_read`, `peak_rss_gb` | num | totals for the whole pass | all |
| `token_norms` | dict | `{layer_as_string: [per-token L2 norm of the residual stream]}` | **`en_paragraph` and `tr_paragraph` only** |
| `loss`, `perplexity`, `top1_acc`, `loss_seconds` | num | next-token metrics over the prompt | **the same two**, which were run with `--loss` |

Each entry of `layers`:

| key | meaning |
|---|---|
| `layer` | layer index, 1–92 |
| `n`, `k` | tokens and experts-per-token for this record |
| `unique_experts` | distinct experts this layer touched over the whole batch |
| `seconds` | wall time of this layer |
| `bytes_read` | bytes read from disk for this layer |
| `is_kda` | true for a KDA linear-attention layer, false for a gated MLA layer (3:1 interleave) |

**Why two of the five carry more.** `token_norms` was added to `scripts/measure_routing.py`
on the morning of 6 September 2026, between the third and fourth run of the day; the
Chinese, Python and Turkish-news traces were already finished. The same two later runs were
the ones invoked with `--loss`. Nothing was backfilled. The consequence for the note is
stated there: the massive-activation spike of its Section 9 is checkable here for English
(and its absence checkable for Turkish), but the Python figure in that section came from
that run's terminal output and no file preserves it.

---

## 5. Derived statistics (`analysis.json`, `analysis.md`)

`scripts/analyze_trace.py <dir>` writes both. `analysis.json` has two objects:

- `rows`: per layer, `n`, `unique` experts, `expected_uniform` (= `896·(1−(1−16/896)^N)`),
  `top100_share`, `H_use_bits` (usage entropy; 896 experts uniform = 9.81 bits), `eff_k`
  (mean effective experts per token, `exp` of the entropy of the 16 normalised weights),
  `jac_t` (mean Jaccard of consecutive tokens' expert sets) and `jac_L` (the same between
  this layer and the next; `null` at layer 92, which has no next layer).
- `prefix_curves`: per layer, `{N: unique experts in the first N tokens}` for
  N ∈ {1, 8, 16, 24, 32, 64, 96, 127, 128, 192, 256, 384, 512} up to the text's length,
  plus the full length. Because routing is causal, one run yields the whole curve.

`analysis.md` is the same content as a table, plus two summaries: the mean
`unique / expected_uniform` ratio over the 92 layers, and the unique-experts-vs-N curve
averaged over layers.

**One caution.** Run without `--prefix`, `analyze_trace.py` rewrites `analysis.json` in the
trace directory — that is how the committed copies were made. With `--prefix N` it only
prints. If you want to compare a rerun against the committed file, copy the directory first
or use `git diff`.

---

## 6. The five texts

| tag | content | language | tokens (incl. BOS) | recorded | wall time | read | peak RSS |
|---|---|---|---:|---|---:|---:|---:|
| `zh_paragraph` | one paragraph, Chinese rendering | Chinese | 111 | 06-09 07:03 | 4 711 s | 651.5 GB | 2.62 GB |
| `code_python` | a Python snippet | Python | 167 | 06-09 08:22 | 6 230 s | 848.7 GB | 2.69 GB |
| `tr_news` | a news-style Turkish paragraph, unrelated content | Turkish | 261 | 06-09 10:06 | 6 989 s | 906.8 GB | 2.72 GB |
| `en_paragraph` | the same paragraph, English rendering | English | 159 | 06-09 12:03 | 5 875 s | 786.8 GB | 2.64 GB |
| `tr_paragraph` | the same paragraph, Turkish original | Turkish | 264 | 06-09 13:41 | 6 716 s | 870.5 GB | 2.65 GB |

Every column after `tokens` is read out of that trace's own `trace.json`.

**Provenance: all five texts were written for this study.** None is copied from a
publication, a corpus or a website.

- The paragraph is about a morning in a coastal town — street lamps going out, a queue at
  the bakery, gulls over the boats, tomatoes and cucumbers at the market. The **Turkish**
  version is the original; the English and Chinese are translations of it, which is a
  limitation (§8).
- `tr_news` is written in the register of a Turkish news report and is about **hazelnut
  production statistics**. It is *not* an Anadolu Agency or BBC Türkçe article, and not from
  any other outlet. Earlier versions of this card, of `docs/LICENSES.md` and of the
  measurement note called it a copyrighted news paragraph whose text would have to be
  replaced by a hash before release. That was wrong about where it came from. Nothing is
  withheld from any of the five manifests, and there is no `sources.json`, because there is
  no third-party source to record.
- `code_python` is a short Python snippet written for the same purpose.

(The Turkish *evaluation* corpus is a different thing entirely: `scripts/build_eval_news.py`
fetches real news from Anadolu Agency and BBC Türkçe at run time for the perplexity
measurements, and that text is not redistributed and is not in any trace.)

**How they were chosen.** Three axes and the smallest set that separates them:

1. *Same meaning, three languages* (`tr_paragraph`, `en_paragraph`, `zh_paragraph`) —
   isolates language from content.
2. *Same language, different content* (`tr_paragraph` vs `tr_news`) — the control that makes
   cross-language overlap interpretable. Without it, a cross-language Jaccard of 0.39 means
   nothing.
3. *Different domain* (`code_python` vs the prose texts).

Lengths were not matched deliberately; they are what the same content costs in each
language, which is itself one of the note's measurements (Turkish is 1.66× English).
Comparisons across texts are therefore made at matched prefixes (55 or 110 tokens), which
one trace supports for every prefix because routing is causal.

**How they were produced.** One invocation per text, all 93 layers, model tokenizer, BOS
prepended:

```bash
python scripts/measure_routing.py \
    --text-file <text>.txt \
    --layers 93 \
    --tag <tag> \
    --out-dir <workspace>/traces/<tag>_L93_2026-09-06
```

`--loss` was additionally passed for `en_paragraph` and `tr_paragraph`. No other flags;
`--max-tokens` stayed at its default of 512 and no text reached it. The engine ran with the
fused MXFP4 kernel (`lazy_lora/native/mxfp4_gemm.c`) and **an untrained adapter**: LoRA B
initialised to zero, so the adapter contributed exactly nothing and these are the base
model's routing decisions. Machine: i7-7700HQ (4 cores / 8 threads, AVX2), 7.6 GB RAM,
GTX 1050 (2 GB), 117 GB NVMe holding the packed non-expert trunk, the 1.56 TB checkpoint on
a 2 TB USB hard disk. The committed manifests have this machine's filesystem paths replaced
by placeholders; nothing else was changed.

---

## 7. Worked examples

Three numbers from the measurement note, reproduced from these files. None needs the
checkpoint, a GPU or a network connection.

**(a) Consecutive-token expert-set overlap, 0.258** (note §7, "temporal locality is real and
short-ranged"). It is the mean of `jac_t` over 92 layers × 5 traces, and `jac_t` is already
in the committed `analysis.json`, so this is one line of `jq`:

```console
$ jq -s '[.[] | .rows[].jac_t] | add/length' evidence/traces/*/analysis.json
0.2579586956521737
```

Per trace: Python 0.208, Chinese 0.242, English 0.243, Turkish news 0.295, Turkish
paragraph 0.302 — prose repeats its experts from token to token more than code does.

**(b) The batch-union curve** (note §7): how many distinct experts a batch of N tokens
forces a layer to read. Each `analysis.md` ends with that curve averaged over its 92 layers;
averaging across the traces that reach each N gives the note's table exactly:

| N | zh | code | en | tr_news | tr | mean | note §7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 16 | 16 | 16 | 16 | 16 (2 %) |
| 16 | 111 | 119 | 126 | 112 | 110 | 116 | 116 (13 %) |
| 64 | 254 | 286 | 274 | 255 | 248 | 263 | 263 (29 %) |
| 128 | — | 411 | 383 | 367 | 355 | 379 | 379 (42 %) |
| 256 | — | — | — | 491 | 464 | 478 | 478 (53 %) |

Recomputing it from the arrays rather than the summaries is five lines:

```python
layers = read_trace_bin("evidence/traces/tr_paragraph_L93_2026-09-06/trace.bin")  # §3
for N in (1, 16, 64, 128, 256):
    per_layer = [len(np.unique(ids[:N])) for ids, _ in layers.values() if ids.shape[0] >= N]
    print(N, round(float(np.mean(per_layer))), f"{np.mean(per_layer) / 896:.0%}")
```

**(c) The massive activation at the end of the network** (note §9), from `token_norms` in
the English manifest:

```console
$ jq '.token_norms["92"] | max, (to_entries | max_by(.value) | .key)' \
     evidence/traces/en_paragraph_L93_2026-09-06/trace.json
21269.332
"32"
```

Token 32 of the English paragraph carries a residual norm of 21 269 at layer 92 and 10 916
at layer 91, against a layer-92 median of 78.036 and a BOS token at 275.348; at layer 90 the
maximum is still 271.392. The same query on `tr_paragraph` returns 275.348 at index 0 — the
BOS token and no spike. This works for those two traces only (§4).

**A cache simulation**, which this repository does *not* ship, is about fifteen lines: walk
`ids` row by row, maintain an ordered set per layer, count misses. That is the intended use
of these files, and the reason they are committed rather than described.

---

## 8. Known limitations

- **One model, one checkpoint.** Kimi K3 as released. Nothing here generalises to other MoEs
  by itself.
- **Five short texts**, 111–264 tokens. Every aggregate is a mean over 92 layers of five
  runs, not over a corpus.
- **Translations, not independent samples.** The English and Chinese texts are translations
  of the Turkish original, so translation artefacts cannot be separated from language
  effects.
- **`float16` weights**, not renormalised. Adequate for entropy and effective-k statistics,
  not for reconstructing the model's arithmetic.
- **`int16` expert ids** are per-layer indices into that layer's 896 routed experts. They are
  not comparable across layers: expert 42 of layer 3 has nothing to do with expert 42 of
  layer 4, and the cross-layer Jaccard (`jac_L`) sits at the random level throughout.
- **Shared experts are not in the trace.** Kimi K3 activates 2 shared experts per layer
  unconditionally; only the 16 routed experts are recorded, because only they are a routing
  decision. Any bytes-read model must add the shared experts separately.
- **`token_norms` and the loss metrics exist for two of the five traces** (§4).
- **The `profile_{128,512,1024}_2026-09-06` companion traces are not here.** They cover
  layers 0–12 only, at 128 / 512 / 1024 tokens, and the note's N=1024 union point (~85 % of
  all experts) comes from them — measured on the 13 shallowest and least concentrated
  layers, so that point is an upper bound and, unlike everything else in the note's §5–§9,
  it is not reproducible from this repository.
- **These are measurements of routing, not of expert semantics.** Because a router is a
  linear map, expert-set overlap largely restates hidden-state similarity (Wang, Hayou and
  Nalisnick, arXiv:2604.09780). Overlap numbers computed from these files characterise
  routing geometry; they are not evidence about what an expert has learned.

---

## 9. Licence and attribution

| part | licence |
|---|---|
| `trace.bin` — expert ids and combining weights | **CC0-1.0**. They are functional measurements of a computation. |
| `trace.json`, `analysis.json`, `analysis.md`, this file | **CC BY 4.0**, © 2026 Ibrahim Polat |
| the texts carried in the manifests | written for this study; released with the manifests under **CC BY 4.0** |
| the producing code (`lazy_lora/`, `scripts/`) | Apache-2.0, <https://github.com/heyobi/LazyLora> |

The reasoning behind the split — in particular why routing arrays are treated as facts
about a computation rather than as a derivative of the model — is in
[`docs/LICENSES.md`](../LICENSES.md).

**Relationship to the model.** These are measurements of Kimi K3's behaviour, not model
weights and not a distillation of them. No Kimi K3 parameters are contained in or
recoverable from these files. Kimi K3 is © 2026 Moonshot AI and is released under the
[Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE), which places
no restriction on model outputs; using the model itself requires obtaining the weights from
Moonshot AI under that licence.

**This project is not affiliated with, sponsored by, or endorsed by Moonshot AI.** "Kimi"
and "Kimi K3" are used descriptively to identify the model that was measured. No trademark
rights are claimed.

**Verification oracle.** `evidence/cmp93_en34_2026-09-06.log` compares this engine against
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (FareedKhan-dev, Apache-2.0)
layer by layer over all 93 layers, which is why the routing decisions here can be attributed
to the model rather than to the engine that recorded them. The 34 token ids in that log's
header are the first 34 ids of `en_paragraph_L93_2026-09-06/trace.json`.

---

## 10. Citation

No DOI has been minted. Cite the repository and the commit you read.

```bibtex
@misc{lazylora_k3_routing_traces_2026,
  title        = {Kimi K3 expert-routing traces: 92 MoE layers, five texts},
  author       = {Polat, Ibrahim},
  year         = {2026},
  howpublished = {\url{https://github.com/heyobi/LazyLora/tree/main/evidence/traces}},
  note         = {Evidence bundle for \emph{Expert Routing in a 2.78-Trillion-Parameter
                  MoE, Measured on a Laptop}. Routing arrays CC0-1.0; manifests and
                  analyses CC BY 4.0.}
}

@misc{lazylora_measurement_note_2026,
  title        = {Expert Routing in a 2.78-Trillion-Parameter MoE, Measured on a Laptop},
  author       = {Polat, Ibrahim},
  year         = {2026},
  howpublished = {\url{https://github.com/heyobi/LazyLora/blob/main/docs/measurement_note.md}}
}
```

If you use these traces to evaluate a cache or scheduling policy, the number most worth
reporting back is the one they were committed to make cheap: how your policy behaves as the
batch grows, not at batch size one.
