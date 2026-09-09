> **SUPERSEDED — do not cite this file.** The current version of this note is
> [`docs/measurement_note.md`](measurement_note.md) (v1.2, 10 September 2026). This draft is
> kept only as a record of what was written on 8 September and of what changed. It is wrong
> in at least these ways, all of them corrected in the current note:
>
> - **It said the traces were in a released dataset** (its subtitle: "every table can be
>   regenerated from the trace files in the released dataset") at a time when they were on
>   the machine that recorded them and nowhere else. That claim became true on 9 September
>   2026, when the five traces were committed to `evidence/traces/` — but it was not true
>   when this draft made it, and the current note (v1.2) is the one that describes what the
>   repository actually contains.
> - **"op-level fixtures at 1e-5"** (§1). Seven of the eight reference-op fixtures match
>   kimi-k3-in-c at 1e-5 absolute / 1e-4 relative; the MoE block fixture matches at 2e-4
>   absolute with cosine 1.000000 (`lazy_lora/tests/test_reference_ops.py` hardcodes
>   `abs_tol=2e-4`; `Bulgular.md` §15.6).
> - **"128 GB NVMe"** (§1). The NVMe on this machine is 117 GB, 108.8 GB of it the packed
>   non-expert trunk; the current note and `README.md` use 117 GB throughout.
> - **"~115 MB/s"** as the measured disk bound (§8, and again in reference [11]). 115 MB/s
>   is the USB enclosure's own sequential specification. The aggregate rate measured from
>   the training process's read counters is 110 MB/s across the USB disk and the NVMe
>   trunk, and the effective rate during a layer sweep is 61 MB/s.
> - **"Training (LoRA, 256-token step): 4.5 hours per step"** (§8). That is an early
>   256-token step. The main run's measured step on a full 1024-token packed sequence is
>   6 h 59 m: forward 3 h 11 m (123 s per layer) plus backward 3 h 48 m (147 s per layer).
> - **"the concentration deepens with depth (200 unique experts at the last layer for a
>   111-token text)"** (Abstract). 200 is the layer-92 figure for the Chinese text alone;
>   the other four sit at 453-533 there, and the depth profile in the current note is not
>   monotone — it has a trough at layers 49-60 and widens again after it.
> - **Its bibliography was not verified.** The current note's reference section was checked
>   entry by entry on 9 September 2026 and lists the identifiers that were wrong here in its
>   verification notes.

# Expert Routing in a 2.78-Trillion-Parameter MoE, Measured on a Laptop

*Draft, 8 September 2026. Numbers are from the runs recorded in `Bulgular.md` §16-17;
every table can be regenerated from the trace files in the released dataset.*

## Abstract

We instrument the forward pass of Kimi K3 (2.78T parameters, 93 layers, 896 routed
experts per layer, top-16, MXFP4 expert weights) with an out-of-core engine that streams
the checkpoint from a 2 TB USB hard disk into 7.6 GB of RAM, and record every routing
decision for five texts in three languages and one programming language across all 92
MoE layers. The engine's forward pass matches an independent C implementation layer by
layer (cosine ≥ 0.9857 over 93 layers; this draft said 0.988, see docs/measurement_note.md). We report four findings. (1) Routing is strongly
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

The locality assumption behind inference offloading (MoE-Infinity [4], Pre-gated MoE [12], HOBBIT [6],
DAOP [8]) holds for single-token decoding of this model and is worth 1-2 GB per layer of
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

## References

Every entry below was checked against arXiv, the publisher's page, or the
ACL/PMLR proceedings on 9 September 2026. Identifiers that were wrong or
incomplete in the earlier draft are flagged in the verification notes at the end;
nothing here is cited from memory.

**Model and architecture**

1. Kimi Team. *Kimi K3: Open Frontier Intelligence.* arXiv:2607.24653, July 2026.
   <https://arxiv.org/abs/2607.24653> — the model measured here: 2.8T parameters,
   104B active, 896 routed experts, KDA/MLA hybrid, 1M context.
2. Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture.*
   arXiv:2510.26692, October 2025. <https://arxiv.org/abs/2510.26692> — Kimi Delta
   Attention and the 3:1 KDA/MLA interleave that K3 inherits; source for the layer
   taxonomy in Section 1.

**Expert offloading and caching for MoE inference**

3. A. Eliseev, D. Mazur. *Fast Inference of Mixture-of-Experts Language Models with
   Offloading.* arXiv:2312.17238, December 2023. <https://arxiv.org/abs/2312.17238>
   — the original LRU-plus-speculation offloading argument for single-token
   decoding; our per-token LRU curve (Section 5) reproduces its premise at 100x the
   model scale.
4. L. Xue, Y. Fu, Z. Lu, L. Mai, M. Marina. *MoE-Infinity: Efficient MoE Inference on
   Personal Machines with Sparsity-Aware Expert Cache.* arXiv:2401.14361, January
   2024 (v3, March 2025). <https://arxiv.org/abs/2401.14361> — sequence-level
   activation tracing and prefetching; assumes the temporal locality we confirm for
   decoding and quantify as insufficient for batches.
5. K. Kamahori, T. Tang, Y. Gu, K. Zhu, B. Kasikci. *Fiddler: CPU-GPU Orchestration
   for Fast Inference of Mixture-of-Experts Models.* ICLR 2025; arXiv:2402.07033.
   <https://arxiv.org/abs/2402.07033> — computes on the CPU instead of moving
   experts; the alternative to our sequential sweep when a GPU is present.
6. P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, M. Guo. *HOBBIT: A
   Mixed Precision Expert Offloading System for Fast MoE Inference.* arXiv:2411.01433,
   November 2024. <https://arxiv.org/abs/2411.01433> — replaces cache-miss experts
   with low-precision copies; orthogonal to our finding, and a plausible use of the
   released traces.
7. S. Cao, S. Liu, T. Griggs, P. Schafhalter, X. Liu, Y. Sheng, J. E. Gonzalez,
   M. Zaharia, I. Stoica. *MoE-Lightning: High-Throughput MoE Inference on
   Memory-constrained GPUs.* ASPLOS 2025, pp. 715-730; arXiv:2411.11217.
   <https://arxiv.org/abs/2411.11217> — the closest prior art to our conclusion:
   at batch scale the win comes from pipelining a bandwidth-bound sweep, not from
   caching.
8. Y. Zhang, S. Aggarwal, T. Mitra. *DAOP: Data-Aware Offloading and Predictive
   Pre-Calculation for Efficient MoE Inference.* DATE 2025; arXiv:2501.10375.
   <https://arxiv.org/abs/2501.10375> — per-sequence expert placement plus one-layer
   -ahead prediction; a per-sequence policy of exactly the kind Section 5 shows
   saturating under batching.
9. K. Li, W. Huang, Q. Wang, L. Zheng, X. Liao, H. Jin, J. Xue. *Diff-MoE: Efficient
   Batched MoE Inference with Priority-Driven Differential Expert Caching.* SC '25,
   ACM, 2025. DOI 10.1145/3712285.3759903. — states the same problem we measure
   (prefetch/cache systems are built for batch size one and break at larger batches)
   and answers it with a global/local hot-set hierarchy; our static-residency curve
   (200 GB saves 22%) is the frontier-scale version of its global cache.
10. S. Yang, X. Fan, M. Pan, H. Xi, Z. Wang, S. Sun, K. Keutzer, S. Han, M. Zaharia,
    C. Xu, I. Stoica. *FreeToken: Efficient Edge-Native MoE Serving with
    Bandwidth-Adaptive Execution.* arXiv:2608.16157, August 2026.
    <https://arxiv.org/abs/2608.16157> — current state of the art for MoE serving on
    one personal machine (753B on a single workstation GPU); the inference-side
    counterpart to this engine, and evidence that the bandwidth-adaptive framing,
    not the cache-hit framing, is where the field has moved.
11. K. Alizadeh, I. Mirzadeh, D. Belenko, K. Khatamifard, M. Cho, C. C. Del Mundo,
    M. Rastegari, M. Farajtabar. *LLM in a flash: Efficient Large Language Model
    Inference with Limited Memory.* arXiv:2312.11514, December 2023.
    <https://arxiv.org/abs/2312.11514> — reading weights from flash in large
    contiguous chunks; the same I/O economics that make our sequential sweep the
    right primitive at 115 MB/s.

**Predicting and prefetching expert selection**

12. R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, M. Yang. *Pre-gated MoE: An
    Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.*
    ISCA 2024; arXiv:2308.12066. <https://arxiv.org/abs/2308.12066> — makes routing
    predictable by changing the architecture; the strongest counter-position to
    Section 5, since it removes the need to predict rather than predicting better.
13. X. Song, Z. Zhong, R. Chen, H. Chen. *ProMoE: Fast MoE-based LLM Serving using
    Proactive Caching.* arXiv:2410.22134, October 2024 (v3, September 2025).
    <https://arxiv.org/abs/2410.22134> — learned prediction of the next layer's
    experts; our "prefetch the previous token's 16 experts" baseline (38%) is the
    naive floor such predictors must beat.
14. X. He, S. Zhang, K. Tang, S. Shi, Y. Wang, Z. Zeng, Z. Tang, X. Chu, H. Yin,
    I. W. Tsang, Y. S. Ong. *ExpertFlow: Efficient Mixture-of-Experts Inference via
    Predictive Expert Caching and Token Scheduling.* DAC 2026; arXiv:2410.17954.
    <https://arxiv.org/abs/2410.17954> — predicts whole routing paths and reorders
    tokens to consolidate expert usage; token reordering is the one batch-side lever
    Section 5 does not evaluate.
15. Q. Zhu, X. Ye, Y. Liu, H. Ouyang, C. Song. *PROBE: Co-Balancing Computation and
    Communication in MoE Inference via Real-Time Predictive Prefetching.*
    arXiv:2602.00509, January 2026. <https://arxiv.org/abs/2602.00509> — shows expert
    hotspots migrating abruptly under continuous batching, which is consistent with
    the union-of-experts saturation we measure.
16. V. Gupta, J. H. Ju, K. Sinha, A. Gavrilovska, A. Iyer. *Lynx: Enabling Efficient
    MoE Inference through Dynamic Batch-Aware Expert Selection.* arXiv:2411.08982,
    November 2024. <https://arxiv.org/abs/2411.08982> — states the tension directly
    ("batching forces the activation of many experts, negating MoEs' sparsity");
    the single best citation for Finding 3's negative half.

**Training-time offloading to CPU and NVMe**

17. S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, Y. He. *ZeRO-Infinity: Breaking
    the GPU Memory Wall for Extreme Scale Deep Learning.* SC '21; arXiv:2104.07857.
    <https://arxiv.org/abs/2104.07857> — the reference design for GPU/CPU/NVMe
    training offload; our engine occupies the same design space with one GPU-less
    laptop and a USB disk.
18. J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li,
    Y. He. *ZeRO-Offload: Democratizing Billion-Scale Model Training.* USENIX ATC
    2021; arXiv:2101.06840. <https://arxiv.org/abs/2101.06840> — CPU-side optimizer
    state and update, which is what keeps our 590 MB of LoRA parameters and Adam
    moments resident in 7.6 GB of RAM.
19. Y. Kim, H. Lim, D. Han. *Scaling Beyond the GPU Memory Limit for Large
    Mixture-of-Experts Model Training.* ICML 2024, PMLR 235:24342-24353 (the system
    is named ES-MoE). <https://proceedings.mlr.press/v235/kim24w.html> — expert
    offloading with pipelined expert processing for MoE *training*; the nearest prior
    work to our training step, at three orders of magnitude less scale.
20. D. Yu, L. Shen, H. Hao, W. Gong, H. Wu, J. Bian, L. Dai, H. Xiong. *MoESys: A
    Distributed and Efficient Mixture-of-Experts Training and Inference System for
    Internet Services.* IEEE Transactions on Services Computing 17(5):2626-2639, 2024;
    arXiv:2205.10034 (2022). <https://arxiv.org/abs/2205.10034> — hierarchical storage
    and a CPU-GPU memory ring for models larger than GPU memory.
21. C. Liao, M. Sun, Z. Yang, J. Xie, K. Chen, B. Yuan, F. Wu, Z. Wang. *LoHan:
    Low-Cost High-Performance Framework to Fine-Tune 100B Model on a Consumer GPU.*
    arXiv:2403.06504, March 2024 (v2, December 2024; v1 was titled "Adding NVMe SSDs
    to Enable and Accelerate 100B Model Fine-tuning on a Single GPU" and named the
    system Fuyou). <https://arxiv.org/abs/2403.06504> — SSD-backed activation swapping
    for single-GPU fine-tuning; our NVMe ring buffer for layer-boundary activations is
    the same idea without the GPU.

**Parameter-efficient fine-tuning under a memory budget**

22. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen.
    *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022; arXiv:2106.09685.
    <https://arxiv.org/abs/2106.09685> — the adapter we train (rank 16, alpha 32).
23. T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. *QLoRA: Efficient Finetuning
    of Quantized LLMs.* NeurIPS 2023; arXiv:2305.14314.
    <https://arxiv.org/abs/2305.14314> — 65B on one 48GB GPU by keeping the frozen
    weights quantized; we keep them quantized *and* out of core, which is the step
    this work adds.
24. J. Hao, W. Sun, X. Xin, Q. Meng, Z. Chen, P. Ren, Z. Ren. *MEFT: Memory-Efficient
    Fine-Tuning through Sparse Adapter.* ACL 2024, pp. 2375-2388; arXiv:2406.04984.
    <https://arxiv.org/abs/2406.04984> — adapters held and updated in CPU memory with
    MoE-style sparsity to limit PCIe traffic.
25. A. Raje, A. Nayak, G. Joshi. *MELINOE: Fine-Tuning Enables Memory-Efficient
    Inference for Mixture-of-Experts Models.* arXiv:2602.11192, 2026.
    <https://arxiv.org/abs/2602.11192> — fine-tunes an MoE to concentrate routing so
    that a small expert cache suffices; the direct constructive answer to Finding 3,
    and worth citing as "the locality that is missing can be trained in".

**Measurements of routing behaviour**

26. B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, W. Fedus.
    *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* arXiv:2202.08906,
    2022. <https://arxiv.org/abs/2202.08906> — early evidence that experts specialize
    by shallow/syntactic features rather than by topic.
27. A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, et al. (Mistral AI).
    *Mixtral of Experts.* arXiv:2401.04088, January 2024.
    <https://arxiv.org/abs/2401.04088> — reports no obvious domain specialization but
    strong positional/consecutive-token repetition in routing; the 47B precedent for
    both halves of our Findings 2 and 3.
28. D. Dai, C. Deng, C. Zhao, R. X. Xu, et al. *DeepSeekMoE: Towards Ultimate Expert
    Specialization in Mixture-of-Experts Language Models.* ACL 2024; arXiv:2401.06066.
    <https://arxiv.org/abs/2401.06066> — fine-grained experts plus isolated shared
    experts, the design K3 uses (2 shared experts); relevant to why routed experts can
    be simultaneously concentrated and non-topical.
29. F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, Y. You. *OpenMoE: An Early
    Effort on Open Mixture-of-Experts Language Models.* ICML 2024; arXiv:2402.01739.
    <https://arxiv.org/abs/2402.01739> — "context-independent specialization": routing
    is driven largely by token identity and is fixed early in pretraining. This is the
    most likely mechanism behind our Finding 2 and should be cited as the explanation,
    not merely as related work.
30. K. M. Lo, Z. Huang, Z. Qiu, Z. Wang, J. Fu. *A Closer Look into Mixture-of-Experts
    in Large Language Models.* NAACL 2025; arXiv:2406.18219.
    <https://arxiv.org/abs/2406.18219> — layerwise similarity of experts and of their
    outputs, including the discontinuity at the last layer; a useful comparison for our
    depth profile.
31. N. Muennighoff, L. Soldaini, D. Groeneveld, K. Lo, J. Morrison, et al. *OLMoE: Open
    Mixture-of-Experts Language Models.* arXiv:2409.02060, September 2024.
    <https://arxiv.org/abs/2409.02060> — router saturation, expert co-activation and
    domain/vocabulary specialization measured on a fully open model; the methodological
    template our trace analysis follows.
32. L. Bandarkar, C. Yang, M. Fayyaz, J. Hu, N. Peng. *Multilingual Routing in
    Mixture-of-Experts.* ICLR 2026; arXiv:2510.04694.
    <https://arxiv.org/abs/2510.04694> — finds language-specific routing in *early and
    late* layers with cross-lingual alignment in the middle. This partially contradicts
    our Finding 2, which sees a language signature only in layers 1-8; the difference
    must be reported, not smoothed over.
33. X. Wang, S. Hayou, E. Nalisnick. *The Myth of Expert Specialization in MoEs: Why
    Routing Reflects Geometry, Not Necessarily Domain Expertise.* arXiv:2604.09780,
    April 2026. <https://arxiv.org/abs/2604.09780> — because the router is a linear map,
    expert-set similarity is a restatement of hidden-state similarity; the strongest
    caution against reading our Jaccard tables as evidence of semantic specialization.

**Massive activations and outlier features**

34. T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix
    Multiplication for Transformers at Scale.* NeurIPS 2022; arXiv:2208.07339.
    <https://arxiv.org/abs/2208.07339> — systematic large-magnitude outlier features
    emerging with scale; the quantization-side consequence of what Section 7 observes.
35. Y. Bondarenko, M. Nagel, T. Blankevoort. *Quantizable Transformers: Removing
    Outliers by Helping Attention Heads Do Nothing.* NeurIPS 2023; arXiv:2306.12929.
    <https://arxiv.org/abs/2306.12929> — outliers as the attention mechanism's way of
    emitting "no update"; a mechanistic account of the single-token spikes we see.
36. G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming Language Models
    with Attention Sinks.* ICLR 2024; arXiv:2309.17453.
    <https://arxiv.org/abs/2309.17453> — attention sinks on semantically empty tokens;
    the closest published analogue to a `):\n` token carrying a 10^4 residual norm.
37. M. Sun, X. Chen, J. Z. Kolter, Z. Liu. *Massive Activations in Large Language
    Models.* COLM 2024; arXiv:2402.17762. <https://arxiv.org/abs/2402.17762> — the
    reference definition of the phenomenon; note that they locate massive activations
    in *early-to-middle* layers, whereas we find them in the last two MLA layers,
    which is the contribution of Section 7 and must be stated as a difference.

**Tokenization cost and the bits-per-byte metric**

38. L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, et al. *The Pile: An 800GB
    Dataset of Diverse Text for Language Modeling.* arXiv:2101.00027, 2020.
    <https://arxiv.org/abs/2101.00027> — the bits-per-byte definition used throughout
    Section 6 and in the evaluation protocol (tokenizer-independent, so Turkish and
    English are comparable).
39. O. Ahia, S. Kumar, H. Gonen, J. Kasai, D. Mortensen, N. A. Smith, Y. Tsvetkov.
    *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language
    Models.* EMNLP 2023; arXiv:2305.13707. <https://aclanthology.org/2023.emnlp-main.614/>
    — cross-lingual token-count disparities up to 5x; the prior result our 1.7x figure
    for Turkish sits inside.
40. A. Petrov, E. La Malfa, P. H. S. Torr, A. Bibi. *Language Model Tokenizers Introduce
    Unfairness Between Languages.* NeurIPS 2023; arXiv:2305.15425.
    <https://arxiv.org/abs/2305.15425> — the same disparity framed as a cost and
    latency inequity; supports Finding 4's claim that the tax is a tokenizer artefact.

### Verification notes

- **Corrected identifiers.** *Pre-gated MoE* had no arXiv id in the draft; it is
  arXiv:2308.12066 (ISCA 2024). *MoESys* was listed as "TSC 2022": the arXiv preprint
  is 2022, but the journal version is IEEE TSC 17(5):2626-2639, **2024**. *MoE-Infinity*
  was retitled between versions — the current title is "Efficient MoE Inference on
  Personal Machines with Sparsity-Aware Expert Cache", not the v1 "Activation-Aware
  Expert Offloading for Efficient MoE Serving"; cite the current one. *LoHan* was
  retitled from "Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a
  Single GPU" (system name Fuyou), which is how it is cited elsewhere. *ES-MoE* is the
  system name, not the paper title.
- **Not fully pinned down.** Diff-MoE's page range in the SC '25 proceedings could not
  be read (the ACM Digital Library returns 403 to automated fetches); the DOI is
  correct and should be used as-is until the page numbers are checked by hand. HOBBIT
  and MoE-Infinity are cited as arXiv preprints because no peer-reviewed venue is
  listed on their arXiv records; if a venue is known at submission time, add it.
- **Author lists** are given in full where the paper has fewer than about ten authors
  and truncated with "et al." otherwise (entries 27, 28, 31, 38), preserving the
  published order.
