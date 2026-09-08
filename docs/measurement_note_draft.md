# Expert Routing in a 2.78-Trillion-Parameter MoE, Measured on a Laptop

*Draft, 8 September 2026. Numbers are from the runs recorded in `Bulgular.md` §16-17;
every table can be regenerated from the trace files in the released dataset.*

## Abstract

We instrument the forward pass of Kimi K3 (2.78T parameters, 93 layers, 896 routed
experts per layer, top-16, MXFP4 expert weights) with an out-of-core engine that streams
the checkpoint from a 2 TB USB hard disk into 7.6 GB of RAM, and record every routing
decision for five texts in three languages and one programming language across all 92
MoE layers. The engine's forward pass matches an independent C implementation layer by
layer (cosine ≥ 0.988 over 93 layers). We report four findings. (1) Routing is strongly
concentrated: a batch touches 43-56% of the experts a uniform router would, and the
concentration deepens with depth (200 unique experts at the last layer for a 111-token
text). (2) The router is domain-sensitive but language-agnostic: expert overlap between
Turkish, English and Chinese renderings of the same paragraph equals the overlap between
two unrelated passages in one language, while prose and Python code are twice as far
apart; a language signature exists only in the first ~8 layers. (3) Expert locality is
real and short-ranged: per-token LRU caches of 1-2 GB per layer hit 62-72% of requests in
autoregressive decoding, yet the union of experts read by a training batch reaches 42% of
all experts at 128 tokens and 53% at 256, so caching and prefetching cannot help
training-time offloading; a static hot-expert residency of 200 GB saves only 22% of
reads. (4) The "low-resource language tax" for Turkish sits in the tokenizer, not the
router: the same text costs 1.7x the tokens and 1.6x the bits per byte of English, while
Turkish routes slightly *more* concentrated than English at equal length. We release the
traces (layer, token, expert, weight) so cache and scheduling policies can be evaluated at
frontier scale without the 1.5 TB checkpoint.

## 1. Setting

- Model: Kimi K3, `kimi_linear` hybrid (69 KDA linear-attention layers, 24 gated MLA
  layers), latent MoE (7168 → 3584 → experts 3584→3072→3584 → 7168), 2 shared experts,
  attention-residual bank every 12 layers. Experts stored as MXFP4 (17.5 MB each).
- Machine: i7-7700HQ (4 cores, AVX2), 7.6 GB RAM, GTX 1050 2 GB, 128 GB NVMe (packed
  non-expert weights, 108.8 GB), 2 TB HDD in a USB enclosure (routed experts, ~115 MB/s).
- Engine: LazyLoRA (this work). Forward pass verified against kimi-k3-in-c on 34 tokens
  over all 93 layers (Table 1) and on op-level fixtures at 1e-5.

**Table 1.** Cosine between LazyLoRA and the C reference residual stream, 34 tokens.

| layer | 12 | 24 | 48 | 72 | 84 | 88 | 90 | 91 | 92 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine | 0.99978 | 0.99919 | 0.99797 | 0.98791 | 0.99773 | 0.99700 | 0.99889 | 0.99966 | 0.99984 |

## 2. Data

Five texts through all layers: a paragraph rendered in Turkish (264 tokens), English (159)
and Chinese (111) with the same meaning; a Turkish news paragraph (261); a Python snippet
(167). Traces record, per layer and token, the 16 selected experts and their combining
weights. Because routing is causal, one run yields the unique-expert curve for every
prefix length.

## 3. Concentration

**Table 2.** Unique experts per layer relative to the uniform expectation
896·(1−(1−16/896)^N); top-100 share of activations; usage entropy (uniform = 9.81 bits).

| text | N | unique/uniform (mean, 92 layers) | layer 1 | layer 46 | layer 92 | top-100 share | entropy (bits) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chinese | 111 | 0.43 | 555 | 223 | 200 | 0.76 | 7.27 |
| Python | 167 | 0.54 | 705 | 450 | 533 | 0.68 | 7.70 |
| Turkish news | 261 | 0.56 | 667 | 326 | 458 | 0.71 | 7.50 |
| English | 159 | 0.50 | 690 | 295 | 453 | 0.71 | 7.54 |
| Turkish | 264 | 0.53 | 664 | 306 | 453 | 0.74 | 7.37 |

Depth profile at N=110 (mean over texts): ~430 unique experts in layers 1-36, 304 in
37-48, 243 in 49-60, 338 in 61-72, ~295 in 73-92. Combining weights are flat (effective
experts per token 12-16 of 16), so pruning to the heaviest experts would not help.

## 4. Domain over language

Jaccard overlap of unique expert sets in equal 55-token windows, mean over 92 layers:

| pair | Jaccard |
|---|---:|
| Turkish vs English (same meaning) | 0.39 |
| Turkish vs Chinese | 0.35 |
| English vs Chinese | 0.38 |
| same text, first vs second half (EN / TR news / code) | 0.36 / 0.34 / 0.37 |
| prose vs Python (EN / TR) | 0.21 / 0.20 |
| Turkish paragraph vs Turkish news | 0.30 |

Cross-language overlap equals within-language different-content overlap; code is twice as
far from prose as languages are from each other. In layers 1-8 only, cross-language
overlap (0.29-0.36) drops below within-language (0.41-0.42). At equal length Turkish is
the most concentrated of the three (347 unique experts vs 367 EN, 362 ZH; entropy 7.26 vs
7.50 vs 7.44 bits).

## 5. Locality: per-token yes, per-batch no

- Consecutive-token expert-set Jaccard: 0.258 mean (random: 0.009), decaying with
  distance: 0.21 (d=2), 0.14 (d=8), 0.12 (d=32), 0.10 (d=128).
- Autoregressive regime, per-layer LRU hit rate: 27% (16 experts), 49% (32), 62% (64,
  1.1 GB/layer), 72% (128, 2.2 GB), 80% (256), 84% (448). "Prefetch the previous token's
  16 experts": 38%.
- Batch regime, experts that must be read per layer: 16 (N=1), 116 (N=16), 263 (N=64),
  379 (N=128, 42% of all), 478 (N=256, 53%), ~760 (N=1024, 85%).
- Static residency (hot set chosen on four texts, evaluated on the fifth): 15 GB saves
  2%, 50 GB 6%, 100 GB 11%, 200 GB 22% of expert reads.

The locality assumption behind inference offloading (MoE-Infinity, Pre-gated MoE, HOBBIT,
DAOP) holds for single-token decoding of this model and is worth 1-2 GB per layer of
cache. It does not survive batching: the quantity that matters for a training step is the
union of expert sets, and it saturates. The correct primitive for training-time
offloading is a bandwidth-optimal sequential sweep amortised over the batch, which is what
the engine does (measured: cost per sweep, 8x more tokens costs 2x more time).

## 6. Where the Turkish tax lives

The same paragraph: 111 Chinese, 159 English, 264 Turkish tokens. On 2048-token
Wikipedia slices of the same eleven topics, the model spends 0.194 bits per byte on
English and 0.311 on Turkish (1.6x); on post-release Turkish news, 0.455. Routing for
Turkish is not more diffuse than for English (Section 4). The cost of Turkish for this
model is therefore in tokenization and per-token modeling, not in expert allocation.

## 7. Model behaviour worth reporting

In the last two (MLA) layers a single token per text develops a residual norm of
1-2×10⁴ (median 78; the BOS token 275): " front" in the English paragraph, `):\n` in the
code. The C reference reproduces it exactly (std 22.18 vs 22.28 at layer 91). The final
per-token RMSNorm absorbs it; perplexity is unaffected (English paragraph 5.9, Turkish
2.2, code 1.9). It is a massive-activation phenomenon at the end of the network rather
than the middle.

## 8. Cost of looking

Forward pass, 93 layers, this machine: 100-140 s per layer at 128-1024 tokens, bounded by
the disk at ~115 MB/s after a fused MXFP4 kernel removed the dequantisation cost
(289 → ~25-55 ms per expert). One text costs 1.3-1.9 hours; the five traces cost eight.
Training (LoRA, 256-token step): 4.5 hours per step.

## 9. Limitations

One model; five short texts; concentration numbers depend on N (reported per N); the
Wikipedia evaluation slices are memorised (bpb 0.19-0.31) and used only as a forgetting
control. The English/Chinese renderings are translations of a Turkish original.

## Data and code

Traces: `traces/*_L93_2026-09-06/{trace.bin,trace.json}` (format: `lazy_lora/monitor/trace.py`).
Code: github.com/heyobi/LazyLora. Reference engine: kimi-k3-in-c (FareedKhan-dev).

## References (to complete)

MoE-Infinity (arXiv 2401.14361); Pre-gated MoE (ISCA 2024); HOBBIT (2411.01433); DAOP
(2501.10375); Diff-MoE (SC'25); FreeToken (2608.16157); ES-MoE (ICML 2024); MoESys (TSC
2022); LoHan (2403.06504); ZeRO-Infinity (SC'21); MELINOE (2602.11192); Sun et al.,
Massive Activations (2024); Kimi K3 technical report.
