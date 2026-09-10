# Expert Routing in a 2.78-Trillion-Parameter MoE, Measured on a Laptop

**Version 1.2, 10 September 2026.** Supersedes `docs/measurement_note_draft.md`. Version
1.0, on 9 September, said that the traces and the comparison log were on one machine and
nowhere else, and that every routing number here was therefore an assertion a reader could
not check; both were committed to `evidence/` hours later, and v1.1 the same day said so.
This version completes the bibliography — entries 28-41 and the verification notes were
missing from the file as first published — and corrects one piece of arithmetic in Section
11: one adapter per expert would cost roughly 420 GB of fp32 weights, gradients and Adam
moments, not the 315 GB stated there. No measurement has changed in any version.

Where this document and [docs/numbers.md](docs/numbers.md) disagree, that table names the source and the source settles it.

Every number in this note comes from a run recorded in `Bulgular.md` §16–§18 or
`DEVAM.md` §11, §15–§17, and each table says which. The raw material of those runs is in
`evidence/`: the five routing traces, the 93-layer comparison log, the per-step losses and
the run manifest, with `SHA256SUMS` over all of it. Section 12 states, result by result,
what each one needs — most need the traces alone, two need the 1.56 TB checkpoint, and one
is not reproducible today at all.

**Notation.**

- **†** — the number is produced by the analysis tooling and quoted here, but it was not
  transcribed into the experiment log at the time of the run. It is recomputable from
  `evidence/traces/` with `scripts/analyze_trace.py`; treat it as unaudited until somebody
  does that and reports the result.
- **‡** — a bibliographic field (venue, page range, author list) that could not be
  verified; see the verification notes after the references.

---

## Abstract

We instrument the forward pass of Kimi K3 [1] (2.78 T parameters, 93 layers, 896 routed
experts per layer, top-16, 2 shared experts, MXFP4 expert weights) with an out-of-core
engine that streams the 1.56 TB checkpoint from a USB hard disk into 7.6 GB of RAM, and
record every routing decision for five texts in three natural languages and one
programming language across all 92 MoE layers. The engine's forward pass matches an
independent C implementation layer by layer over all 93 layers (every layer at cosine 0.9857
or better, lowest row 0.985744 at layer 71,
0.99984 at the output), so the routing decisions recorded are the model's and not the
engine's.

We report four measurements.

1. **Concentration.** A batch touches 43–56 % of the experts a uniform router would.
   Concentration deepens with depth but not monotonically: 555–705 unique experts at
   layer 1, a minimum of 243 in layers 49–60, ~295 in layers 73–92.
2. **Domain over language.** Expert-set overlap between Turkish, English and Chinese
   renderings of one paragraph (Jaccard 0.35–0.39) equals the overlap between two
   unrelated passages in a single language (0.34–0.37), while prose and Python code are
   twice as far apart (0.20–0.21). A language signature appears only in the first layers.
   This restates for a 2.78 T model what Mixtral [27] reported at 47 B, and the mechanism
   OpenMoE [29] proposes — routing keyed on token identity — predicts it; it also
   disagrees with Bandarkar et al. [32], who find language-specific routing in early
   *and late* layers of smaller multilingual MoEs (Section 6).
3. **Locality is per token, not per batch.** Consecutive tokens share 26 % of their
   experts against 0.9 % at random, and a per-layer LRU of 1.1–2.2 GB hits 62–72 % in
   single-token decoding, which is the assumption inference-offloading systems
   [3, 4, 8, 12] are built on and it holds here. It does not survive a training batch:
   the union of expert sets reaches 42 % of all experts at 128 tokens and 53 % at 256,
   so a cache is the wrong primitive for a training step and a bandwidth-optimal
   sequential sweep is the right one. That batching negates MoE sparsity is already
   published [9, 15, 16]; the contribution here is the saturation curve measured at 896
   experts across 92 layers, and the finding that a 200 GB static hot set — nearly twice
   this machine's entire NVMe — saves only 22 % of expert reads.
4. **The Turkish tax is in the tokenizer, not the router.** The same content costs 1.7×
   the tokens and 1.6× the bits per byte of English, a disparity well inside the
   published cross-lingual range [39, 40], while at equal token count Turkish routes
   *more* concentrated than English, not less.

The traces — layer, token, expert id and combining weight, over all 92 MoE layers of the
five texts — are in this repository under `evidence/traces/`: 5 669 776 bytes of routing
records, plus the manifests and the analyses computed from them. Cache and scheduling
policies can therefore be evaluated at frontier scale without the 1.56 TB checkpoint or
this hardware, and every routing number below is recomputable by a reader with a laptop, in
seconds, with no checkpoint and no network (Sections 4, 12 and 14). The layer-by-layer
comparison log behind Table 1 is in `evidence/` as well. What is not there is the
checkpoint; Section 12 says which results need it.

---

## 1. Setting

**Model.** Kimi K3 [1]: `kimi_linear` hybrid, 69 KDA linear-attention layers and 24 gated
MLA layers in a 3:1 interleave [2]; latent MoE (7168 → 3584 → experts 3584→3072→3584 →
7168), 896 routed experts per layer with top-16 selection plus 2 shared experts [28];
attention-residual bank refreshed every 12 layers. Routed expert weights are stored as
MXFP4, 17.5 MB per expert. Layer 0 is dense; layers 1–92 are MoE.

**Machine.** i7-7700HQ (4 cores / 8 threads, AVX2), 7.6 GB RAM, GTX 1050 (2 GB), 117 GB
NVMe holding the packed non-expert trunk (108.8 GB), and a 2 TB hard disk in a USB
enclosure holding the routed experts.

**Throughput: three numbers that are not the same number.** (i) *110 MB/s aggregate*, the
only one measured on this engine end to end: the main training run's `/proc` read counter
stood at 3 219 659 335 955 bytes after 8 h 06 m 57 s (`Bulgular.md` §18.1), which is the
USB disk and the NVMe trunk together, seen by one process. (ii) *61 MB/s effective inside
one layer sweep*: 14.5 GB per MoE layer in 238 s at N=1024 (`Bulgular.md` §16.5) — lower
than the aggregate because the reader thread idles while the layer computes. (iii)
*115 MB/s*, the USB enclosure's own sequential benchmark: a property of one device, not a
measurement of this engine, and not used here as one. Earlier figures of 118.5 MB/s are
from a different machine with a different disk. **We make no claim that the engine
saturates the bus.**

**Engine.** LazyLoRA (this work). One layer resident at a time; layer-boundary activations
to an NVMe ring buffer; routed experts streamed per expert, because a single layer's 896
experts are 15.7 GB and the streaming unit therefore cannot be the decoder layer. The
adapter is LoRA rank 16, alpha 32, and one part of its shape is worth stating because the
opposite is easy to assume: the routed-expert adapter is **one** adapter per layer, in the
MoE latent space (3584 → 3072 → 3584), **shared by all 896 experts of that layer**
(`lazy_lora/trainer/lazy_trainer.py:130`). It is not one adapter per expert — that would be
about 2.6 × 10¹⁰ trainable parameters and a different machine (Section 11). Attention and
the two shared experts carry their own adapters, also one per layer. The traces below were
recorded with an untrained adapter (LoRA B initialised to zero, so the adapter contributes
exactly nothing to the forward pass).

---

## 2. Verification of the instrument

A routing measurement is only worth as much as the forward pass that produced it. The
engine's forward pass was checked against `kimi-k3-in-c` (FareedKhan-dev), an independent
C99 implementation of the same model, on the first 34 tokens of the English paragraph
(BOS included; the 32nd token is `" front"`). The C engine ran all 93 layers in 61 min at
5.2 GB peak RSS; LazyLoRA replayed the same token sequence and compared layer by layer in
48 min, reading 427 GB (`Bulgular.md` §17.1, log `evidence/cmp93_en34_2026-09-06.log`).

**Table 1.** Cosine between the LazyLoRA and C-reference residual streams, 34 tokens,
untrained adapter. Nine sampled layers of the 93 compared; all 93 rows are in
`evidence/cmp93_en34_2026-09-06.log`, and the minimum over them is 0.985744 at layer 71.

| layer | 12 | 24 | 48 | 72 | 84 | 88 | 90 | 91 | 92 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine | 0.99978 | 0.99919 | 0.99797 | 0.98791 | 0.99773 | 0.99700 | 0.99889 | 0.99966 | 0.99984 |
| our std | 0.054 | 0.0062 | 0.0068 | 0.040 | 0.094 | 0.330 | 0.835 | 22.28 | 43.28 |
| C std | 0.054 | 0.0062 | 0.0068 | 0.040 | 0.094 | 0.331 | 0.834 | 22.18 | 43.24 |

Scope of this check, stated precisely: one English paragraph, 34 tokens, an untrained
adapter, and the full 93-layer stack. It has not been repeated with a trained adapter, at
the 1024-token lengths the training run uses, or on Turkish or code input where routing
differs. Nine rows are sampled above, but all 93 are in the repository: the log has one
line per layer with the cosine, the maximum absolute difference, both standard deviations
and the number of experts that layer read, and it ends `total 2869s, bytes read 426.59 GB`.
The minimum of 0.985744 at layer 71 and the 0.999840 at layer 92 are rows in that file, not
assertions about a file nobody else has. The 34 token ids it replays are printed in its
header and are the first 34 ids of
`evidence/traces/en_paragraph_L93_2026-09-06/trace.json`, so the two files check each
other. What the log does not give a reader is the ability to *recompute* it: that needs the
1.56 TB checkpoint and a `kimi-k3-in-c` per-layer dump, neither of which is in this
repository (Section 12).

Op-level fixtures: seven of the eight ops (`rmsnorm`, `situ_glu`, `shortconv`,
`kda_decay`, `router`, `attnres`, `mla`) match the reference at 1e-5 absolute / 1e-4
relative; the eighth, `moe`, matches at 2e-4 absolute with cosine 1.000000, which is the
MXFP4 decode path's own rounding (`Bulgular.md` §15.6).

The backward pass is not exercised by the measurements in this note, but it is what the
engine exists for, so its check is stated here too. LoRA, input and residual-bank
gradients were compared against central finite differences on four representative layers
— 1 (KDA + MoE, one bank entry), 3 (MLA), 12 (block boundary), 13 (two bank entries) —
over all 16 LoRA tensors of the layer plus the input and bank directions, on 4 tokens, in
an fp32 engine with float64 loss reduction, adaptive epsilon and routing-flip detection.
Worst relative error 3.6e-3 on the MLA layer and ≤ 2.0e-3 elsewhere, except two
directions whose analytic derivative is ~3e-4, where it reaches 9.1e-3 (`DEVAM.md` §11).
This is four layers of 93, not a whole-model gradient check.

---

## 3. Related work

**Expert offloading and caching for inference.** The line from Mixtral-offloading [3]
through MoE-Infinity [4], Fiddler [5], HOBBIT [6], MoE-Lightning [7], DAOP [8],
Diff-MoE [9]‡ and FreeToken [10] shares one premise: expert activation has enough
temporal locality that a cache or a predictor can hide the cost of holding experts off
the accelerator. LLM in a flash [11] makes the I/O side of the same argument — read in
large contiguous chunks and the storage device stops being the bottleneck. Section 7
confirms the premise for single-token decoding of K3 and quantifies where it stops
holding. FreeToken [10], three weeks older than this note, is the closest system: a 753 B
MoE served on a single workstation GPU by abandoning fixed offloading policies for
bandwidth-adaptive execution. Its framing — bandwidth, not hit rate, as the governing
quantity — is the one our training-side measurement independently arrives at.

**Predicting routing rather than caching it.** Pre-gated MoE [12] changes the
architecture so the next layer's experts are known one layer early; ProMoE [13] and
ExpertFlow [14] learn to predict them; PROBE [15] prefetches in real time and reports
expert hotspots migrating abruptly under continuous batching. Our "prefetch the previous
token's 16 experts" baseline (38 %, Section 7) is the naive floor such predictors must
beat. Lynx [16] states the tension this note measures outright — batching, needed for
throughput, forces activation of many experts and negates MoE sparsity — and Diff-MoE [9]
opens on the same problem, that prefetch and cache systems are built for batch size one
and break at larger batches. **The qualitative claim in Section 7 is therefore not new.**
What is new is the curve at this scale: 896 experts, top-16, 92 layers, with the
saturation points measured rather than argued.

**Training-time offload and parameter-efficient fine-tuning.** ZeRO-Offload [18] and
ZeRO-Infinity [17] are the reference designs for spilling optimizer state, parameters and
activations to CPU and NVMe; LoHan [21] (system name Fuyou) does SSD-backed activation
swapping for single-GPU fine-tuning of a 175 B dense model; ES-MoE [19] is the nearest
prior work, offloading expert parameters to host memory with pipelined per-expert
processing for MoE *training*, on server GPUs. LoRA [22] is the adapter, QLoRA [23] keeps
the frozen base quantized in memory, and MEFT [24] keeps adapters and their updates on
the CPU. The engine used here occupies the same design space as [17] and [19] with three
orders of magnitude more parameters, no usable GPU, and a USB disk instead of an NVMe
array. MELINOE [25] is the constructive counter to Section 7: it fine-tunes an MoE so
that it prefers fewer experts per sequence, which makes a small cache viable — the
locality we report as absent under batching can be induced rather than cached around.

**Measurements of routing behaviour.** ST-MoE [26] found experts organising around
shallow and syntactic features rather than topics; Mixtral [27] reported a lack of domain
specialization together with strong consecutive-token repetition, which is both halves of
our Findings 2 and 3 at 47 B; DeepSeekMoE [28] is the fine-grained-plus-shared-expert
design K3 uses; OpenMoE [29] proposes the mechanism — context-independent specialization,
routing driven by token identity and fixed early in pretraining — that would produce our
Finding 2 exactly, since translations of one paragraph share meaning but almost no
tokens; Lo et al. [30] and OLMoE [31] provide the layerwise and router-saturation
methodology our trace analysis follows. Bandarkar et al. [32] disagree with our layer
profile and are discussed in Section 6. Wang et al. [33] argue that because a router is a
linear map, expert-set overlap largely restates hidden-state similarity; that caution
applies directly to every Jaccard number in this note and is repeated in Section 13.

**Massive activations, and the tokenizer.** Section 9's observation is an instance of the
phenomenon defined by Sun et al. [37], with the difference in layer position stated
there; LLM.int8() [34], Bondarenko et al. [35] and attention sinks [36] are the
quantization consequence, the mechanism and the closest analogue respectively. Bits per
byte, the metric Section 8 and the evaluation protocol rest on, is defined in The
Pile [38]; the cross-lingual token-count disparity our 1.7× figure sits inside is
documented by Ahia et al. [39] (up to 5×) and Petrov et al. [40] (up to 15×).

---

## 4. Data and instrument

Five texts, each pushed through all 93 layers with every routing decision recorded:

| tag | text | language | tokens | trace directory under `evidence/traces/` |
|---|---|---|---:|---|
| `zh_paragraph` | one paragraph, Chinese rendering | Chinese | 111 | `zh_paragraph_L93_2026-09-06/` |
| `en_paragraph` | the same paragraph, English rendering | English | 159 | `en_paragraph_L93_2026-09-06/` |
| `tr_paragraph` | the same paragraph, Turkish original | Turkish | 264 | `tr_paragraph_L93_2026-09-06/` |
| `tr_news` | a news-style Turkish paragraph, unrelated content | Turkish | 261 | `tr_news_L93_2026-09-06/` |
| `code_python` | a Python snippet | Python | 167 | `code_python_L93_2026-09-06/` |

**Provenance of the texts.** All five were written for this study. The paragraph is about a
morning in a coastal town; the Turkish rendering is the original and the English and
Chinese are translations of it, which is a limitation and is restated in Section 13. The
fourth text is written in the register of a news report and is about hazelnut production
statistics; it is **not** taken from Anadolu Agency, BBC Türkçe or any other publication.
Earlier versions of this note called it a copyrighted news paragraph that would have to be
withheld from any release. That was wrong about its provenance, and it is why all five
manifests carry their text and token ids with nothing replaced by a hash.

**Instrument.** `lazy_lora/monitor/trace.py`, driven by `scripts/measure_routing.py`. For
each layer it appends a record — `int32 layer, int32 N, int32 K`, then `int16[N·K]` expert
ids and `float16[N·K]` combining weights, little-endian — and updates a JSON manifest with
per-layer wall time, bytes read and unique-expert count, plus totals and peak RSS at the
end. Statistics are computed by `scripts/analyze_trace.py`. Because routing is causal, one
recorded pass yields the unique-expert curve for every prefix length up to the text's own
length without a second run.

**Where the traces are: `evidence/traces/`, in this repository.** Recorded 6 September 2026
on the machine of Section 1 at 1.3–1.9 h per text, committed on 9 September 2026. Each
directory holds `trace.bin` (the routing records; 654 672 to 1 555 536 bytes, 92 records),
`trace.json` (the manifest, including the input text and its token ids) and the
`analysis.json` / `analysis.md` that `analyze_trace.py` produced from it.
`evidence/SHA256SUMS` covers all of them, `docs/traces/README.md` documents the format
field by field, and `evidence/README.md` describes the bundle they sit in. Every routing
table in Sections 5–9 is therefore recomputable by a reader with a laptop and no
checkpoint; Section 12 gives the command for each and what it costs. Version 1.0 of this
note, written on the morning of 9 September, said the traces were on the machine that
recorded them and nowhere else. Deleting that sentence is what this release is for.

One field is not uniform across the five. `token_norms`, the per-layer, per-token L2 norm
of the residual stream that Section 9 rests on, was added to `measure_routing.py` on the
morning of 6 September, between the third and fourth run; only `en_paragraph` and
`tr_paragraph` carry it. Those two were also run with `--loss`, so they alone carry `loss`,
`perplexity` and `top1_acc`.

---

## 5. Finding 1: concentration

**Table 2.** Unique experts per layer against the uniform expectation
896·(1−(1−16/896)^N); top-100 share of activations; usage entropy (uniform = 9.81 bits).
`Bulgular.md` §17.

| text | N | unique/uniform (mean, 92 layers) | layer 1 | layer 46 | layer 92 | top-100 share | entropy (bits) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chinese | 111 | 0.43 | 555 | 223 | 200 | 0.76 | 7.27 |
| Python | 167 | 0.54 | 705 | 450 | 533 | 0.68 | 7.70 |
| Turkish news | 261 | 0.56 | 667 | 326 | 458 | 0.71 | 7.50 |
| English | 159 | 0.50 | 690 | 295 | 453 | 0.71 | 7.54 |
| Turkish | 264 | 0.53 | 664 | 306 | 453 | 0.74 | 7.37 |

**Depth profile** at a common prefix of N=110, averaged over the five texts: ~430 unique
experts in layers 1–36, 304 in 37–48, 243 in 49–60 (the most concentrated band), 338 in
61–72, ~295 in 73–92. Concentration is not monotone in depth: there is a trough around
layers 49–60 and a mild widening after it. Lo et al. [30] report a comparable layerwise
trend with a discontinuity at the final layer.

Combining weights are close to flat: the mean effective number of experts per token,
exp(H) over the 16 normalised routing weights, is 12–16 of 16. Pruning a token's
computation to its heaviest experts would therefore discard real mass, and the shortcut
"compute only the top few of the top-16" is not available in this model.

The practical reading for an offloading system: in the deep layers two thirds of the
experts are never read for a given text, so if any residency cache is worth building it
belongs in layers 37–92 rather than spread uniformly. Section 7 shows how little that is
worth in a training regime.

---

## 6. Finding 2: domain over language

Jaccard overlap of unique expert sets in equal 55-token windows, mean over the 92 MoE
layers (`Bulgular.md` §17):

| pair | Jaccard |
|---|---:|
| Turkish vs English (same meaning) | 0.39 |
| Turkish vs Chinese (same meaning) | 0.35 |
| English vs Chinese (same meaning) | 0.38 |
| same text, first half vs second half | 0.34–0.37 |
| prose vs Python (English / Turkish) | 0.21 / 0.20 |
| Turkish paragraph vs Turkish news | 0.30 |

Cross-language overlap for identical content (0.35–0.39) is indistinguishable from
within-language overlap for different content (0.34–0.37). Code sits at roughly half that
(0.20–0.21) — twice as far from prose as the languages are from one another — and two
different Turkish texts (0.30) are further apart than a Turkish paragraph and its English
translation (0.39). Expert selection tracks content domain, not language.

**Mechanism.** OpenMoE [29] reports context-independent specialization: routing keyed
predominantly on token identity, fixed early in pretraining. If routing keys on token
identity, translations of one paragraph — which share meaning but almost no tokens — must
land on different experts, and cross-language overlap must collapse to cross-content
overlap. That is what the table shows. Mixtral [27] reports the same absence of domain
specialization at 47 B, which makes this a property of the routing design and not of
K3's scale.

**Where the language signature is.** The effect is confined to the first layers. In the
earlier 66-layer measurement with 80-token windows (`Bulgular.md` §16.3), cross-language
overlap at layers 1 and 5 (0.44, 0.36) falls clearly below within-language overlap at the
same layers (0.54, 0.51), and there is no such gap deeper in the stack. In the 92-layer
traces the gap does not persist past roughly layer 8.† The per-layer table behind that
sentence was not transcribed into the experiment log at the time. It is
`analyze_trace.py <dir_A> <dir_B> --prefix 55` over two of the directories in
`evidence/traces/`, which anyone can now run in seconds — and should, because until
somebody does, the layer-8 boundary rests on an output nobody wrote down.

**Disagreement.** Bandarkar et al. [32] (ICLR 2026) find language-specific routing in
early *and late* decoder layers, with cross-lingual alignment concentrated in the middle;
related work reports language-exclusive experts for low-resource languages. We observe no
late-layer language signature in K3. The measurements are not comparable in three
respects and we do not claim to have refuted theirs: different model and scale, three
languages here against many there, and 55-token windows over five texts here. We report
the difference rather than leave it to be found.

**Equal-length control.** Concentration differences between languages could be an artefact
of token count, since the Turkish rendering is 1.66× the English. At a matched prefix
they are not: in the 66-layer measurement at N=159 for both, Turkish uses fewer unique
experts in 54 of the 66 layers and has lower usage entropy (7.38 vs 7.63 bits,
`Bulgular.md` §16.3). The 92-layer traces give the same ordering at a common prefix —
Turkish 347 unique experts, English 367, Chinese 362; entropy 7.26 / 7.50 / 7.44 bits.†
The hypothesis that a lower-resource language is routed more diffusely is false for this
pair; the effect runs the other way.

**Caveat that applies to this whole section.** Wang et al. [33] show that because the
router is a linear map, similar hidden states must select similar experts, so expert-set
overlap is close to a restatement of representation similarity rather than independent
evidence of semantic specialization. These Jaccard values characterise **routing
geometry**, not expert semantics. The prose-versus-code gap survives that caveat as a
statement about routing: whatever the experts mean, code and prose are routed to
substantially disjoint sets, and an offloading policy sees that difference regardless of
how it is interpreted.

---

## 7. Finding 3: locality is per token, not per batch

All figures from `Bulgular.md` §17.2 unless marked, computed over the five texts × 92
layers with no additional forward pass.

**Temporal locality is real and short-ranged.** Mean Jaccard between the expert sets of
consecutive tokens is 0.258, against 0.009 for random token pairs — about 29× chance. It
decays with distance: 0.21 at d=2, 0.14 at d=8, 0.12 at d=32, 0.10 at d=128. (This
supersedes the "no predictability" heading of `Bulgular.md` §16.2, which was too strong;
the accurate statement is that locality exists, is short-ranged, and is a property of
adjacent tokens rather than of a text.)

**In the decoding regime a cache works.** With tokens arriving one at a time, per-layer
LRU hit rates are:

| cache size (experts / layer) | resident bytes / layer | hit rate |
|---:|---:|---:|
| 64 | 1.1 GB | 62 % |
| 128 | 2.2 GB | 72 % |
| 256 | 4.5 GB | 80 % |

The policy "prefetch the previous token's 16 experts" hits 38 %. The premise that
MoE-Infinity [4], Pre-gated MoE [12], HOBBIT [6] and DAOP [8] build on therefore holds for
this model in the single-token regime, at 128 of 896 experts — 14.3 % of the layer —
buying 72 %. (An earlier draft of this note also listed hit rates for cache sizes 16, 32
and 448; those points are not in the experiment log and have been removed rather than
reproduced.)

The one published study that measures the same quantity agrees. "Cacheable by Design?"
[41]‡ reports, for a 235 B MoE decoded on an 8 GB GPU, adjacent-token expert reuse at
2.0x chance and an LRU over 13.4 % of experts serving 66 % of requests — against 14.3 %
serving 72 % here, one order of magnitude up in scale. It is an independent measurement of
the decode-time locality this subsection confirms; it does not measure the training-batch
union that the next one is about.

**In the training regime it collapses.** The quantity that matters for a training step is
not the hit rate but the union of expert sets over the batch, because every expert in the
union must be read at least once:

| tokens in the batch | experts read per layer | share of 896 |
|---:|---:|---:|
| 1 | 16 | 2 % |
| 16 | 116 | 13 % |
| 64 | 263 | 29 % |
| 128 | 379 | 42 % |
| 256 | 478 | 53 % |
| 1024 | ~760 | ~85 % |

**The 1024-token point is measured on layers 0–12 only** (`Bulgular.md` §16.5: mean 759
unique experts per layer over 13 layers), which are the *least* concentrated layers in
the model — layer 1 alone reaches 555–705 unique experts while layers 49–60 sit at 243
(Section 5). The whole-model figure at N=1024 is therefore expected to be lower than
85 %, and the point should be read as an upper bound until it is remeasured over all 92
layers. The argument does not depend on it: 53 % at 256 tokens already means that more
than half of every layer is read.

**Static residency does not rescue it.** Taking the (layer, expert) pairs that cover 80 %
of activations gives 140 experts at layer 56 to 461 at layer 1, median 251, and 25,157
pairs in total — 440 GB, roughly four times the machine's entire NVMe. A leave-one-out
simulation (hot set chosen on four texts, evaluated on the fifth; baseline 436 experts
read per layer from the hard disk) gives:

| NVMe budget | experts pinned | experts still read / layer | saving |
|---:|---:|---:|---:|
| 15 GB | 857 | 428 | 2 % |
| 50 GB | 2 857 | 411 | 6 % |
| 100 GB | 5 714 | 386 | 11 % |
| 200 GB | 11 428 | 340 | 22 % |

This machine has 117 GB of NVMe, 108.8 GB of it already the packed non-expert trunk. The
15 GB row is the budget the log assumed from the free space at the time; about 11 GB is
free during the training run, and the 200 GB row is nearly twice the whole drive. The
finding is that within-text concentration is strong but hot sets move from text to text —
the domain effect of Section 6 — so a static residency cache is not the lever on this
hardware.

**Framing.** That batching negates MoE sparsity is stated by Lynx [16], is the opening
problem of Diff-MoE [9]‡, and appears as migrating hotspots in PROBE [15]. This note does
not claim the observation. It contributes the saturation curve at 896-expert, top-16,
92-layer scale, the static-residency curve against a real NVMe budget, and the conclusion
that follows for a training step on a bandwidth-bound device: read every expert of the
layer once in disk order and amortise it over the batch. The engine does exactly that,
and the amortisation is measured — eight times the tokens costs twice the time
(`Bulgular.md` §16.5: 128 → 1024 tokens raises the per-layer time from 122 s to 238 s).
MoE-Lightning [7] reaches the same primitive from the serving side by pipelining a
bandwidth-bound sweep; FreeToken [10] generalises it to bandwidth-adaptive execution;
MELINOE [25] is the alternative that this note does not take, namely training the missing
locality into the model instead of scheduling around its absence.

---

## 8. Finding 4: where the Turkish tax lives

The same paragraph costs 111 Chinese, 159 English and 264 Turkish tokens — Turkish is
1.66× English. On 2048-token Wikipedia slices covering the same eleven topics in both
languages, the model spends 0.194 bits per byte on English and 0.311 on Turkish, a factor
of 1.6; on a 2048-token slice of Turkish news published after the model's release, 0.455
(`DEVAM.md` §16.1). Bits per byte [38] is used precisely because it is
tokenizer-independent and therefore comparable across the two languages; per-token
perplexity is not, and Section 9 states the numbers that are not comparable as such.

A 1.7× token-count disparity is unremarkable as a tokenizer result: Ahia et al. [39]
report up to 5× across languages and Petrov et al. [40] up to 15×. That is what makes the
second half of the measurement worth stating: **the router shows no matching penalty.**
At equal token count Turkish is routed more concentrated than English, not less
(Section 6). The cost of Turkish for this model sits in tokenization and per-token
modelling, not in expert allocation — which also means that no amount of expert-side
engineering will recover it.

A reader who accepts the late-layer language routing of Bandarkar et al. [32] might
expect a routing-side language tax to appear; we do not find one, in either the overlap
tables or the equal-length concentration control.

---

## 9. Massive activations at the end of the network

In the last two MLA layers (91 and 92) a single token develops a residual norm of
10 000–20 000 against a median of 78 across tokens (the BOS token reaches 275). It is the
32nd token, `" front"`, in the English paragraph, and `):\n` in the Python snippet. It was
**not** observed in the Turkish or Chinese paragraphs; the Turkish news trace is not
reported in the log either way (`Bulgular.md` §17).

That is readable out of the bundle for the English trace. In
`evidence/traces/en_paragraph_L93_2026-09-06/trace.json`, `token_norms["91"]` and
`token_norms["92"]` peak at 10 916.049 and 21 269.332, both at token index 32, against a
layer-92 median of 78.036 and a BOS token at 275.348; at layer 90 the maximum is still
271.392, so the spike is confined to the last two layers. The same field in
`tr_paragraph`'s manifest peaks at 274.76 and 275.348 — the BOS token, and no spike. The
Python observation is *not* checkable in the bundle: that run predates the `token_norms`
field, and the figure was read from the terminal (Section 13).

Two things make it worth reporting. First, the C reference reproduces it: at layer 91 the
residual standard deviation is 22.28 in our engine and 22.18 in the C engine, at layer 92
43.28 against 43.24 (Table 1). It is the model's behaviour, not a bug in either
implementation. Second, the final per-token RMSNorm absorbs it, so next-token quality is
unaffected.

Sun et al. [37] define the phenomenon and locate it largely in early-to-middle layers,
with values that stay largely constant across inputs. Here it appears in the last two
layers, and the carrying token differs by text (`" front"`, `):\n`). Both differences are
stated as differences; we have not established which of scale, the KDA/MLA hybrid, or the
input set accounts for them. The closest analogue is the attention sink [36] — a
semantically thin token absorbing a large share of attention mass — with the mechanism
proposed by Bondarenko et al. [35], attention heads emitting a no-op. The practical
consequence belongs to quantization: outliers of this magnitude are what LLM.int8() [34]
was written for, which matters for a model whose expert weights are MXFP4.

---

## 10. Cost of the measurement

All on the machine of Section 1.

| what | measured |
|---|---|
| One L93 trace (111–264 tokens, 93 layers) | 1.3–1.9 h per text, ~8 h for the five (§17) |
| Per-layer read volume | 9.1 GB (N=128), 13.6 GB (N=512), 14.5 GB (N=1024), layers 0–12 (§16.5) |
| Per-MoE-layer time, layers 0–12 | 122 s (N=128), 197 s (N=512), 238 s (N=1024) (§16.5) |
| Peak RSS during tracing | 2.89 / 2.98 / 3.39 GB at N=128 / 512 / 1024 (§16.5) |
| Peak RSS during the five L93 traces | 2.62–2.72 GB (`peak_rss_gb` in each `trace.json`) |
| 66-layer traces (earlier run) | 584 GB read at 159 tokens, 655 GB at 264 tokens (§16.4) |
| Effective disk throughput during a sweep | 61 MB/s (14.5 GB / 238 s) (§16.5) |
| 93-layer cross-implementation comparison | C engine 61 min at 5.2 GB RSS; LazyLoRA replay 48 min, 427 GB read (§17.1) |

The cost is per sweep, not per token: eight times the tokens costs twice the time. The
fused MXFP4 kernel (`lazy_lora/native/mxfp4_gemm.c`, OpenMP/AVX2) is what moved the
bottleneck from dequantisation back to the disk — per-expert cost fell from ~289 ms of
dequantisation to 55 ms at 22 rows, and layer 12 at 1024 tokens fell from 200 s to 108 s
(`DEVAM.md` §15).

Every figure in the first block of that table is in the bundle rather than only in the
log: each `evidence/traces/<tag>_L93_2026-09-06/trace.json` carries `total_seconds`,
`bytes_read` and `peak_rss_gb` for the whole pass and `seconds`, `bytes_read` and
`unique_experts` for each of the 92 layers, and `evidence/cmp93_en34_2026-09-06.log` ends
with the line `total 2869s, bytes read 426.59 GB` for the comparison run.

For scale, the training step the engine exists for, on the same machine: forward
3 h 11 m 34 s (123.6 s per layer) plus backward 3 h 48 m 07 s (147.2 s per layer) =
**6 h 59 m 41 s** for one 1024-token packed sequence, measured on step 1 of the main run
(`evidence/run_manifest.json`). The forward half of that is checkable in the bundle: the
last line of `evidence/forward_loss_main.jsonl` is the main run's step 1 at Unix time
1 788 959 106, `started` in `run_manifest.json` is 1 788 947 613.66, and the difference is
11 492 s — the 3 h 11 m 34 s of `Bulgular.md` §18.1, to within the seconds between process
launch and the trainer's first line. The backward half finishes after the last line either
file holds and is timed from the run's terminal output. At that rate 100 steps is about 29
days at step 1's pace; the cadence over the first three steps is about 7.4 h, so about 31 days,
finishing around 9-11 October 2026. An earlier 256-token step took 4 h 31 m at
6.24 GB peak RSS (`DEVAM.md` §17), which is the highest resident set this engine has ever
recorded; the main run sits at 4.0–4.7 GB. The five-step proof run of 8–9 September used
two packed sequences of about 541 tokens each and took 5.5–5.8 h per step at 4.5–4.7 GB
resident (`Bulgular.md` §18, losses in `evidence/forward_loss_proof.jsonl`). Figures of
"5.7 h per 1024-token step" in earlier documents came from that run's shorter sequences and
are superseded by the 6 h 59 m 41 s measurement.

---

## 11. What the traces are for

The union curve in Section 7 is the number an offloading system needs and the inference
literature does not measure, because it is a question about a training step rather than a
decode. A trace turns that question into arithmetic over five megabytes: given a candidate
cache size, replacement policy, prefetch depth, batch size or token ordering, the hit rate
and the bytes read per layer follow from the recorded expert ids without a model, a GPU or
1.56 TB of weights. That is why the traces are in `evidence/traces/` rather than in a
sentence promising them: the policies below can be evaluated today, by anyone, against the
same request sequence this note measured.

**Directly testable on the traces, and not evaluated here.** Token reordering to
consolidate expert usage within a batch (ExpertFlow [14]); mixed-precision substitution for
cache misses (HOBBIT [6]); any replacement policy other than LRU; and the batch size at
which a given cache stops paying for itself, which is a short function of the union curve
and the per-expert read cost.

**Future work the traces cannot answer.** The routed-expert adapter trained by this engine
is **one** rank-16 adapter per layer in the MoE latent space, shared by all 896 experts of
that layer (Section 1). The obvious ablation is one adapter per expert, so that the update
can specialise the way the router does. It needs the checkpoint and a training run rather
than a trace, and the arithmetic says why it was not done here: 896 experts × 92 layers ×
0.32 M parameters per expert (three rank-16 adapters over 3584 → 3072 → 3584) is about
2.6 × 10¹⁰ trainable parameters — roughly 420 GB at 16 bytes apiece (fp32 weight,
gradient and the two AdamW moments, `lazy_lora/trainer/optimizer.py`), against the 590 MB
the shared adapter actually costs. That is a different machine's experiment, not one for
7.6 GB of RAM. Whether the shared adapter is a limitation or a regulariser at this scale
is untested. So is the middle ground: one adapter per band of layers, or per group of
co-activated experts, for which the concentration and overlap tables above are the natural
way to choose the grouping.

---

## 12. Reproducibility

Honest tiers. Most of this note needs the traces only, and the traces are in this
repository. Two results need the 1.56 TB checkpoint, one needs a second engine as well,
and one is not reproducible today at all.

**What the "traces" column means.** `evidence/traces/<tag>_L93_2026-09-06/`, committed on
9 September 2026: 5 669 776 bytes of expert ids and combining weights for the five texts
over 92 MoE layers, with `evidence/SHA256SUMS` to check them against. Every routing number
in Sections 5–9 follows from them on any laptop — no checkpoint, no GPU, no network.
`scripts/analyze_trace.py <trace_dir>` reads a few hundred kilobytes and does set
arithmetic over 92 × N × 16 int16 values; it takes seconds on the machine of Section 1 and
has not been benchmarked on any other. Each trace directory also ships the output it
produced — `analysis.json` (per-layer rows and the prefix curves) and `analysis.md` — so a
reader can diff against them instead of trusting a rerun. Note that `analyze_trace.py`
rewrites `analysis.json` in place when run without `--prefix`, which is how the committed
copies were made; with `--prefix` it only prints.

| Result | What it needs | How | Cost |
|---|---|---|---|
| Table 2, concentration, top-100 share, usage entropy, effective experts per token | traces | `scripts/analyze_trace.py <trace_dir>` | seconds |
| Depth profile (Section 5) | traces | `analyze_trace.py --prefix 110` over the five dirs | seconds |
| Unique-experts-vs-N curve, every prefix length up to the text length | traces | `prefix_curves` in `analysis.json`, or `analyze_trace.py` | already committed |
| Batch union curve to N=256 (Section 7) | traces | the same curves, averaged over the traces that reach N; worked through in `docs/traces/README.md` §7 | already committed |
| Jaccard tables (Section 6) | traces | `analyze_trace.py <dir_A> <dir_B> --prefix 55` | seconds per pair |
| Equal-length concentration control (Section 6, †) | traces | `analyze_trace.py --prefix <N>` on the three language traces | seconds |
| Consecutive-token Jaccard 0.258 | traces | `jac_t` column of `analyze_trace.py`, committed in `analysis.json` | already committed |
| Jaccard decay at d=2, 8, 32, 128 | traces | ~10 lines over the expert-id arrays; not in `analyze_trace.py` | seconds to run |
| LRU hit rates, prefetch baseline (Section 7) | traces | a cache simulation over the expert-id arrays; ~15 lines, not in `analyze_trace.py` | seconds to run |
| Static hot-set simulation (Section 7) | traces | leave-one-out over the five sets of (layer, expert) pairs; not in `analyze_trace.py` | seconds to run |
| Per-layer time, bytes read, unique experts, peak RSS (Section 10) | traces | `layers` entries in `trace.json` | already committed |
| Next-token loss, perplexity and top-1 for the two paragraphs (Sections 9, 13) | traces | `loss`, `perplexity`, `top1_acc` in the `en_paragraph` and `tr_paragraph` manifests | already committed |
| Massive activations in English, and their absence in Turkish (Section 9) | traces | `token_norms` in those two manifests: layer 92 max 21 269.332 at token 32, median 78.036, BOS 275.348; Turkish layer 92 max 275.348 | seconds with `jq` |
| Massive activation in the Python snippet (Section 9) | **nothing in the bundle records it** — that run predates `token_norms` | re-run `measure_routing.py` on the snippet | 1.56 TB checkpoint, ~1.7 h |
| Step timings of Section 10 | `evidence/forward_loss_{main,proof}.jsonl` | Unix timestamps, one line per completed forward pass | already committed |
| Table 1, all 93 cosine rows | `evidence/cmp93_en34_2026-09-06.log` | read it; the nine rows above are sampled from it | already committed |
| Recomputing Table 1 rather than reading it | 1.56 TB checkpoint **and** a `kimi-k3-in-c` per-layer dump | `scripts/compare_with_c_dump.py` | 61 min for the C dump, 48 min and 427 GB for the replay |
| The N=1024 union point (~85 %) | the `profile_1024` companion trace, which is **not** in the repository | `prefix_curve()` on it; over all 92 layers it needs the checkpoint | not reproducible today |
| Backward finite differences (Section 2) | 1.56 TB checkpoint | `scripts/verify_backward.py --layer 1 --param-probes 16` | hours, and it prints to the terminal: no log file exists to read instead |
| Bits per byte, Turkish and English (Section 8) | 1.56 TB checkpoint + the eval corpora, which the scripts fetch | `scripts/eval_perplexity.py` | hours |
| Op-level fixtures (Section 2) | a `kimi-k3-in-c` clone (no checkpoint) | `python -m unittest lazy_lora.tests.test_reference_ops` — note that this module **skips** rather than fails when the fixture directory is absent; check for `OK` and not `skipped` | minutes |
| The engine runs at all (no numerics) | this repository only | `bash scripts/run_mock_tests.sh`, synthetic weights | minutes |

---

## 13. Limitations

- **One model, one checkpoint, five short texts** (111–264 tokens). Every number is a
  measurement of Kimi K3 as released, not of MoE routing in general.
- **The English and Chinese texts are translations of the Turkish original.** Translation
  artefacts cannot be separated from language effects with this design.
- **Concentration figures depend on N** and are reported per N throughout; comparisons
  across texts of different lengths are made at matched prefixes or not at all.
- **The N=1024 union point is measured on layers 0–12** — the least concentrated layers —
  and is an upper bound for the whole model (Section 7).
- **The layer-1-to-8 language-signature boundary and the equal-length expert counts are
  marked †**: produced by the analysis tooling and not transcribed into the experiment log
  at the time of the run. They stay marked until someone recomputes them and reports it,
  which anyone can now do in seconds from `evidence/traces/`.
- **Per-token residual norms exist for two of the five traces.** `token_norms` was added to
  `scripts/measure_routing.py` on the morning of 6 September, after the Chinese, Python and
  Turkish-news runs had finished; only `en_paragraph` and `tr_paragraph` carry it. Section
  9's English spike and its absence in Turkish are therefore checkable in the bundle; the
  Python `):\n` spike is not, because it was read from that run's terminal output and no
  file preserves it.
- **The routed-expert adapter is one rank-16 adapter per layer, shared by all 896 experts**
  (Section 1). Nothing here measures what per-expert adapters would do to either routing or
  learning; that ablation is named as future work in Section 11 and was not run.
- **Jaccard overlap characterises routing geometry, not expert semantics** [33]. Nothing
  here shows what an expert has learned.
- **The LRU and static-residency numbers are simulations** over recorded traces, not
  running systems. No cache was built; the 200 GB configuration cannot be built on this
  machine.
- **The forward-pass verification covers 34 tokens of one English paragraph with an
  untrained adapter**, and the backward check covers four layers on four tokens
  (Section 2).
- **Perplexities are not comparable across languages.** Next-token loss on held-out
  paragraphs is 1.776 (perplexity 5.90, top-1 53 %) for English and 0.771
  (perplexity 2.16, top-1 77 %) for Turkish (`Bulgular.md` §17), and Turkish looks better
  only because it is cut into more, shorter pieces. Bits per byte is the comparable
  measure and is what Section 8 uses. A perplexity figure for the Python snippet
  circulated in earlier drafts; it is not in the experiment log and has been removed.
- **No claim is made here about training outcomes.** This note measures routing. The
  engine's learning demonstration (`Bulgular.md` §18) is memorisation of five examples,
  and the pre-registered evaluation of whether the adapter improves Turkish
  (`DEVAM.md` §16.1, threshold committed 8 September 2026 at 07:54:49, commit `6605306`,
  29 hours before the main run started) had not returned a result when this note was
  written.

---

## 14. Data, code and licences

**Traces.** `evidence/traces/<tag>_L93_2026-09-06/` for the five texts — `trace.bin`,
`trace.json`, and the `analysis.json` / `analysis.md` that `scripts/analyze_trace.py`
wrote from them. 5 669 776 bytes of expert ids and combining weights in total, 92 records
per trace. The format is defined by `lazy_lora/monitor/trace.py` and documented field by
field in `docs/traces/README.md`; `evidence/README.md` describes the bundle as a whole.
Earlier drafts and v1.0 of this note said the traces were not in the repository. They are,
as of 9 September 2026, together with the comparison log
`evidence/cmp93_en34_2026-09-06.log` (all 93 rows of Table 1), the raw per-step losses
`evidence/forward_loss_{main,proof}.jsonl` and the run manifest
`evidence/run_manifest.json`. `evidence/SHA256SUMS` covers every one of them. No DOI has
been minted; the identifier goes here when it exists.

**Still not released.** The 1.56 TB checkpoint, the `kimi-k3-in-c` per-layer dump that
Table 1 was computed against, the packed NVMe trunk, the 1.8 GB training checkpoints, and
the `profile_{128,512,1024}_2026-09-06` companion traces behind the N=1024 union point.
The finite-difference harness (Section 2) prints to the terminal rather than to a file, so
its numbers are quoted from `DEVAM.md` §11 and have to be re-run on the real checkpoint to
be reproduced.

**Text in the traces.** A trace manifest carries the input text and its token ids, which
reconstruct the text exactly, and all five carry theirs — nothing is withheld and no hash
substitution was needed. All five texts were written for this study: the paragraph about a
morning in a coastal town in Turkish, English and Chinese, the Python snippet, and the
news-style Turkish paragraph, which is about hazelnut production statistics and is **not**
taken from Anadolu Agency, BBC Türkçe or any other publication. Earlier versions of this
note and of `docs/LICENSES.md` described that fifth text as a copyrighted news article and
planned to withhold it; that was wrong about its provenance. (The Turkish *evaluation*
corpus of Section 8 is a different thing and does come from news outlets; it is fetched at
run time and is not redistributed.)

**Nothing was regenerated for publication.** The files in `evidence/` are the ones the
runs wrote, copied in unmodified except that this machine's own filesystem paths were
replaced with placeholders (`<model dir>`, `<workspace>`).

**Code.** `github.com/heyobi/LazyLora`, Apache-2.0. Reference engine: `kimi-k3-in-c`
(FareedKhan-dev), Apache-2.0, which is both the oracle for Table 1 and the source of the
op fixtures. Model weights: Kimi K3, © 2026 Moonshot AI, released under the Kimi K3
License; no weights are contained in this repository, in `evidence/`, or in anything
derivable from a trace. This work is not affiliated with, sponsored by or endorsed by
Moonshot AI.

**Licences.** Routing arrays (`trace.bin`): CC0-1.0. Manifests, analyses, logs and the
rest of `evidence/`: CC BY 4.0. Code: Apache-2.0. The reasoning is in `docs/LICENSES.md`.

---

## References

Every entry was checked against arXiv, the publisher's page, or the ACL/PMLR proceedings
on 9 September 2026. Identifiers that were wrong or incomplete in the earlier draft are
listed in the verification notes at the end; nothing here is cited from memory, and the
two fields that could not be verified are marked ‡ at their point of use.

**Model and architecture**

1. Kimi Team. *Kimi K3: Open Frontier Intelligence.* arXiv:2607.24653, July 2026.
   <https://arxiv.org/abs/2607.24653> — the model measured here: 2.8 T parameters,
   104 B active, 896 routed experts, KDA/MLA hybrid, 1 M context.
2. Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture.*
   arXiv:2510.26692, October 2025. <https://arxiv.org/abs/2510.26692> — Kimi Delta
   Attention and the 3:1 KDA/MLA interleave that K3 inherits; source for the layer
   taxonomy in Section 1.

**Expert offloading and caching for MoE inference**

3. A. Eliseev, D. Mazur. *Fast Inference of Mixture-of-Experts Language Models with
   Offloading.* arXiv:2312.17238, December 2023. <https://arxiv.org/abs/2312.17238>
   — the original LRU-plus-speculation offloading argument for single-token decoding;
   our per-token LRU curve (Section 7) reproduces its premise at far larger scale.
4. L. Xue, Y. Fu, Z. Lu, L. Mai, M. Marina. *MoE-Infinity: Efficient MoE Inference on
   Personal Machines with Sparsity-Aware Expert Cache.* arXiv:2401.14361, January 2024
   (v3, March 2025). <https://arxiv.org/abs/2401.14361> — sequence-level activation
   tracing and prefetching; assumes the temporal locality we confirm for decoding and
   quantify as insufficient for a training batch. Cited as a preprint: no venue is listed
   on its arXiv record.‡
5. K. Kamahori, T. Tang, Y. Gu, K. Zhu, B. Kasikci. *Fiddler: CPU-GPU Orchestration for
   Fast Inference of Mixture-of-Experts Models.* ICLR 2025; arXiv:2402.07033.
   <https://arxiv.org/abs/2402.07033> — computes on the CPU instead of moving experts;
   the alternative to a sequential sweep when a usable GPU is present.
6. P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, M. Guo. *HOBBIT: A Mixed
   Precision Expert Offloading System for Fast MoE Inference.* arXiv:2411.01433, November
   2024. <https://arxiv.org/abs/2411.01433> — replaces cache-miss experts with
   low-precision copies; a policy directly testable on the traces in `evidence/traces/`.
   Cited as a preprint: no venue is listed on its arXiv record.‡
7. S. Cao, S. Liu, T. Griggs, P. Schafhalter, X. Liu, Y. Sheng, J. E. Gonzalez,
   M. Zaharia, I. Stoica. *MoE-Lightning: High-Throughput MoE Inference on
   Memory-constrained GPUs.* ASPLOS 2025, pp. 715-730; arXiv:2411.11217.
   <https://arxiv.org/abs/2411.11217> — the closest prior art to Section 7's conclusion:
   at batch scale the win comes from pipelining a bandwidth-bound sweep, not from caching.
8. Y. Zhang, S. Aggarwal, T. Mitra. *DAOP: Data-Aware Offloading and Predictive
   Pre-Calculation for Efficient MoE Inference.* DATE 2025; arXiv:2501.10375.
   <https://arxiv.org/abs/2501.10375> — per-sequence expert placement plus one-layer-ahead
   prediction; a per-sequence policy of exactly the kind Section 7 shows saturating under
   batching.
9. K. Li, W. Huang, Q. Wang, L. Zheng, X. Liao, H. Jin, J. Xue. *Diff-MoE: Efficient
   Batched MoE Inference with Priority-Driven Differential Expert Caching.* SC '25, ACM,
   2025. DOI 10.1145/3712285.3759903. — states the problem Section 7 measures (prefetch
   and cache systems are built for batch size one and break at larger batches) and answers
   it with a global/local hot-set hierarchy; our static-residency curve is the
   frontier-scale version of its global cache. Page range not verified.‡
10. S. Yang, X. Fan, M. Pan, H. Xi, Z. Wang, S. Sun, K. Keutzer, S. Han, M. Zaharia,
    C. Xu, I. Stoica. *FreeToken: Efficient Edge-Native MoE Serving with
    Bandwidth-Adaptive Execution.* arXiv:2608.16157, August 2026.
    <https://arxiv.org/abs/2608.16157> — current state of the art for MoE serving on one
    personal machine (753 B on a single workstation GPU); the inference-side counterpart
    to this engine, and evidence that the bandwidth-adaptive framing rather than the
    cache-hit framing is where the field has moved.
11. K. Alizadeh, I. Mirzadeh, D. Belenko, K. Khatamifard, M. Cho, C. C. Del Mundo,
    M. Rastegari, M. Farajtabar. *LLM in a flash: Efficient Large Language Model Inference
    with Limited Memory.* arXiv:2312.11514, December 2023.
    <https://arxiv.org/abs/2312.11514> — reading weights from flash in large contiguous
    chunks; the same I/O economics that make a sequential sweep the right primitive at
    61 MB/s effective throughput.

**Predicting and prefetching expert selection**

12. R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, M. Yang. *Pre-gated MoE: An
    Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.*
    ISCA 2024; arXiv:2308.12066. <https://arxiv.org/abs/2308.12066> — makes routing
    predictable by changing the architecture; the strongest counter-position to Section 7,
    since it removes the need to predict rather than predicting better.
13. X. Song, Z. Zhong, R. Chen, H. Chen. *ProMoE: Fast MoE-based LLM Serving using
    Proactive Caching.* arXiv:2410.22134, October 2024 (v3, September 2025).
    <https://arxiv.org/abs/2410.22134> — learned prediction of the next layer's experts;
    our "prefetch the previous token's 16 experts" baseline (38 %) is the naive floor such
    predictors must beat.
14. X. He, S. Zhang, K. Tang, S. Shi, Y. Wang, Z. Zeng, Z. Tang, X. Chu, H. Yin,
    I. W. Tsang, Y. S. Ong. *ExpertFlow: Efficient Mixture-of-Experts Inference via
    Predictive Expert Caching and Token Scheduling.* DAC 2026; arXiv:2410.17954.
    <https://arxiv.org/abs/2410.17954> — predicts whole routing paths and reorders tokens
    to consolidate expert usage; token reordering is the one batch-side lever Section 7
    does not evaluate, and the traces in `evidence/traces/` let anyone evaluate it.
15. Q. Zhu, X. Ye, Y. Liu, H. Ouyang, C. Song. *PROBE: Co-Balancing Computation and
    Communication in MoE Inference via Real-Time Predictive Prefetching.* arXiv:2602.00509,
    January 2026. <https://arxiv.org/abs/2602.00509> — expert hotspots migrating abruptly
    under continuous batching, consistent with the union saturation we measure.
16. V. Gupta, J. H. Ju, K. Sinha, A. Gavrilovska, A. Iyer. *Lynx: Enabling Efficient MoE
    Inference through Dynamic Batch-Aware Expert Selection.* arXiv:2411.08982, November
    2024. <https://arxiv.org/abs/2411.08982> — states the tension directly ("batching
    forces the activation of many experts, negating MoEs' sparsity"); the prior statement
    of Finding 3's qualitative half.

**Training-time offloading to CPU and NVMe**

17. S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, Y. He. *ZeRO-Infinity: Breaking the
    GPU Memory Wall for Extreme Scale Deep Learning.* SC '21; arXiv:2104.07857.
    <https://arxiv.org/abs/2104.07857> — the reference design for GPU/CPU/NVMe training
    offload; this engine occupies the same design space with one GPU-less laptop and a
    USB disk.
18. J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, Y. He.
    *ZeRO-Offload: Democratizing Billion-Scale Model Training.* USENIX ATC 2021;
    arXiv:2101.06840. <https://arxiv.org/abs/2101.06840> — CPU-side optimizer state and
    update, which is what keeps ~590 MB of LoRA parameters and Adam moments resident in
    7.6 GB of RAM.
19. Y. Kim, H. Lim, D. Han. *Scaling Beyond the GPU Memory Limit for Large
    Mixture-of-Experts Model Training.* ICML 2024, PMLR 235:24342-24353 (the system is
    named ES-MoE). <https://proceedings.mlr.press/v235/kim24w.html> — expert offloading
    with pipelined expert processing for MoE *training*; the nearest prior work to this
    engine's training step, on server GPUs.
20. D. Yu, L. Shen, H. Hao, W. Gong, H. Wu, J. Bian, L. Dai, H. Xiong. *MoESys: A
    Distributed and Efficient Mixture-of-Experts Training and Inference System for
    Internet Services.* IEEE Transactions on Services Computing 17(5):2626-2639, 2024;
    arXiv:2205.10034 (2022). <https://arxiv.org/abs/2205.10034> — hierarchical storage and
    a CPU-GPU memory ring for models larger than GPU memory.
21. C. Liao, M. Sun, Z. Yang, J. Xie, K. Chen, B. Yuan, F. Wu, Z. Wang. *LoHan: Low-Cost
    High-Performance Framework to Fine-Tune 100B Model on a Consumer GPU.* arXiv:2403.06504,
    March 2024 (v2, December 2024; v1 was titled "Adding NVMe SSDs to Enable and Accelerate
    100B Model Fine-tuning on a Single GPU" and named the system Fuyou).
    <https://arxiv.org/abs/2403.06504> — SSD-backed activation swapping for single-GPU
    fine-tuning; our NVMe ring buffer for layer-boundary activations is the same idea
    without the GPU.

**Parameter-efficient fine-tuning under a memory budget**

22. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. *LoRA:
    Low-Rank Adaptation of Large Language Models.* ICLR 2022; arXiv:2106.09685.
    <https://arxiv.org/abs/2106.09685> — the adapter the engine trains (rank 16, alpha 32).
23. T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. *QLoRA: Efficient Finetuning of
    Quantized LLMs.* NeurIPS 2023; arXiv:2305.14314. <https://arxiv.org/abs/2305.14314>
    — 65 B on one 48 GB GPU by keeping the frozen weights quantized; here they are kept
    quantized *and* out of core.
24. J. Hao, W. Sun, X. Xin, Q. Meng, Z. Chen, P. Ren, Z. Ren. *MEFT: Memory-Efficient
    Fine-Tuning through Sparse Adapter.* ACL 2024, pp. 2375-2388; arXiv:2406.04984.
    <https://arxiv.org/abs/2406.04984> — adapters held and updated in CPU memory with
    MoE-style sparsity to limit PCIe traffic.
25. A. Raje, A. Nayak, G. Joshi. *MELINOE: Fine-Tuning Enables Memory-Efficient Inference
    for Mixture-of-Experts Models.* arXiv:2602.11192, 2026.
    <https://arxiv.org/abs/2602.11192> — fine-tunes an MoE to concentrate routing so that
    a small expert cache suffices; the constructive answer to Section 7, and the reason
    that section is framed as a measurement rather than an impossibility.

**Measurements of routing behaviour**

26. B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, W. Fedus. *ST-MoE:
    Designing Stable and Transferable Sparse Expert Models.* arXiv:2202.08906, 2022.
    <https://arxiv.org/abs/2202.08906> — early evidence that experts organise around
    shallow and syntactic features rather than topics.
27. A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, et al. (Mistral AI).
    *Mixtral of Experts.* arXiv:2401.04088, January 2024. <https://arxiv.org/abs/2401.04088>
    — reports no obvious domain specialization but strong consecutive-token repetition in
    routing; the 47 B precedent for both halves of Findings 2 and 3.
28. D. Dai, C. Deng, C. Zhao, R. X. Xu, et al. *DeepSeekMoE: Towards Ultimate Expert
    Specialization in Mixture-of-Experts Language Models.* ACL 2024; arXiv:2401.06066.
    <https://arxiv.org/abs/2401.06066> — fine-grained routed experts plus isolated shared
    experts, which is the MoE design K3 uses (896 routed, top-16, 2 shared); the reason
    routed experts can be strongly concentrated (Section 5) and still not topical
    (Section 6).
29. F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, Y. You. *OpenMoE: An Early Effort
    on Open Mixture-of-Experts Language Models.* ICML 2024; arXiv:2402.01739.
    <https://arxiv.org/abs/2402.01739> — context-independent specialization: routing
    driven largely by token identity and fixed early in pretraining. This is the
    mechanism Section 6 appeals to, and it predicts Finding 2 rather than merely being
    consistent with it.
30. K. M. Lo, Z. Huang, Z. Qiu, Z. Wang, J. Fu. *A Closer Look into Mixture-of-Experts in
    Large Language Models.* NAACL 2025; arXiv:2406.18219.
    <https://arxiv.org/abs/2406.18219> — layerwise similarity of experts and of their
    outputs, including a discontinuity at the last layer; the comparison for the
    non-monotone depth profile of Section 5.
31. N. Muennighoff, L. Soldaini, D. Groeneveld, K. Lo, J. Morrison, et al. *OLMoE: Open
    Mixture-of-Experts Language Models.* arXiv:2409.02060, September 2024.
    <https://arxiv.org/abs/2409.02060> — router saturation, expert co-activation and
    domain/vocabulary specialization on a fully open model; the methodological template
    the trace analysis in `scripts/analyze_trace.py` follows.
32. L. Bandarkar, C. Yang, M. Fayyaz, J. Hu, N. Peng. *Multilingual Routing in
    Mixture-of-Experts.* ICLR 2026; arXiv:2510.04694.
    <https://arxiv.org/abs/2510.04694> — language-specific routing in *early and late*
    decoder layers, with cross-lingual alignment concentrated in the middle. Section 6
    finds no late-layer language signature in K3 and reports the disagreement rather
    than smoothing it over; the two measurements are not comparable in model, scale or
    language count.
33. X. Wang, S. Hayou, E. Nalisnick. *The Myth of Expert Specialization in MoEs: Why
    Routing Reflects Geometry, Not Necessarily Domain Expertise.* arXiv:2604.09780,
    April 2026. <https://arxiv.org/abs/2604.09780> — because the router is a linear map,
    expert-set similarity largely restates hidden-state similarity; the caution that
    applies to every Jaccard number here, stated in Sections 6 and 13.

**Massive activations and outlier features**

34. T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix
    Multiplication for Transformers at Scale.* NeurIPS 2022; arXiv:2208.07339.
    <https://arxiv.org/abs/2208.07339> — systematic large-magnitude outlier features
    emerging with scale; the quantization-side consequence of Section 9, and directly
    relevant to a model whose expert weights are MXFP4.
35. Y. Bondarenko, M. Nagel, T. Blankevoort. *Quantizable Transformers: Removing Outliers
    by Helping Attention Heads Do Nothing.* NeurIPS 2023; arXiv:2306.12929.
    <https://arxiv.org/abs/2306.12929> — outliers as the attention mechanism's way of
    emitting a no-op; the mechanistic account Section 9 points at for the single-token
    spike.
36. G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming Language Models with
    Attention Sinks.* ICLR 2024; arXiv:2309.17453. <https://arxiv.org/abs/2309.17453>
    — attention sinks on semantically thin tokens; the closest published analogue to a
    `" front"` or `):\n` token carrying a residual norm of 10⁴.
37. M. Sun, X. Chen, J. Z. Kolter, Z. Liu. *Massive Activations in Large Language Models.*
    COLM 2024; arXiv:2402.17762. <https://arxiv.org/abs/2402.17762> — the reference
    definition of the phenomenon. It locates massive activations in *early-to-middle*
    layers with values largely constant across inputs; Section 9 finds them in the last
    two MLA layers with a carrying token that changes with the text, and states both as
    differences rather than as refutations.

**Tokenization cost and the bits-per-byte metric**

38. L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, et al. *The Pile: An 800GB
    Dataset of Diverse Text for Language Modeling.* arXiv:2101.00027, December 2020.
    <https://arxiv.org/abs/2101.00027> — the bits-per-byte definition used in Section 8
    and in the pre-registered evaluation protocol; tokenizer-independent, which is why
    Turkish and English are comparable in it and not in perplexity.
39. O. Ahia, S. Kumar, H. Gonen, J. Kasai, D. Mortensen, N. A. Smith, Y. Tsvetkov. *Do All
    Languages Cost the Same? Tokenization in the Era of Commercial Language Models.*
    EMNLP 2023; arXiv:2305.13707.
    <https://aclanthology.org/2023.emnlp-main.614/> — cross-lingual token-count
    disparities up to 5×; the published range the 1.7× Turkish figure of Section 8 sits
    inside.
40. A. Petrov, E. La Malfa, P. H. S. Torr, A. Bibi. *Language Model Tokenizers Introduce
    Unfairness Between Languages.* NeurIPS 2023; arXiv:2305.15425.
    <https://arxiv.org/abs/2305.15425> — the same disparity framed as a cost and latency
    inequity; supports Section 8's reading of the Turkish tax as a tokenizer artefact
    rather than a routing one.

**Independent measurement of the same quantity**

41. *Cacheable by Design?* arXiv:2608.18261, August 2026.
    <https://arxiv.org/abs/2608.18261> — a pre-registered study of MoE cache locality at
    the edge: Qwen3-235B decoded on an 8 GB consumer GPU, adjacent-token expert reuse at
    about 2.0× chance, and an LRU over 13.4 % of experts serving 66 % of requests. It is
    the closest external measurement to Section 7's decode-time numbers (14.3 % serving
    72 %, on a model roughly twelve times larger), and it stops at inference, which is
    where the training-batch union curve begins. Cited by short title and arXiv
    identifier only: the full title, the author list and any venue were not transcribed
    into this bibliography.‡

### Verification notes

- **Corrected identifiers.** *Pre-gated MoE* [12] had no arXiv id in the draft; it is
  arXiv:2308.12066 (ISCA 2024). *MoESys* [20] was listed as "TSC 2022": the arXiv preprint
  is 2022, but the journal version is IEEE Transactions on Services Computing
  17(5):2626-2639, **2024**. *MoE-Infinity* [4] was retitled between versions — the
  current title is "Efficient MoE Inference on Personal Machines with Sparsity-Aware
  Expert Cache", not the v1 "Activation-Aware Expert Offloading for Efficient MoE
  Serving". *LoHan* [21] was retitled from "Adding NVMe SSDs to Enable and Accelerate 100B
  Model Fine-tuning on a Single GPU", which named the system Fuyou and is how it is cited
  elsewhere. *ES-MoE* [19] is the system name, not the paper title.
- **Fields that could not be verified (‡).** *Diff-MoE* [9]: the page range in the SC '25
  proceedings could not be read (the ACM Digital Library returns 403 to automated
  fetches); the DOI is correct and should be used as-is until the pages are checked by
  hand. *MoE-Infinity* [4] and *HOBBIT* [6] are cited as preprints because no
  peer-reviewed venue is listed on their arXiv records; add the venue if one is known at
  submission time. *"Cacheable by Design?"* [41] is cited by short title and arXiv id
  only — the full title and the author list were never transcribed, and this note quotes
  its two headline numbers from `README.md` rather than from the paper's own tables. That
  is the weakest citation in this list and is marked as such at its point of use.
- **Author lists** are given in full where the paper has fewer than about ten authors and
  truncated with "et al." otherwise (entries 27, 28, 31, 38), preserving the published
  order.
- **What "checked" means here.** Each entry was opened on arXiv, the publisher's page or
  the ACL/PMLR proceedings on 9 September 2026 and the title, authors, year and venue read
  off it. The annotations after each em dash are this note's own reading of the paper, not
  the paper's abstract, and are the author's responsibility.
