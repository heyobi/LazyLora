# LazyLoRA

Out-of-core LoRA fine-tuning and routing measurement for **Kimi K3** (2.78 T parameters,
93 layers, 896 routed experts per layer, MXFP4 expert weights) on a consumer laptop:
7.6 GB RAM, 4 CPU cores, a 2 GB GPU, and the 1.56 TB checkpoint on a USB hard disk.

The model never fits in memory. Every layer streams its weights from disk, the forward
pass keeps only one layer resident, layer-boundary activations go to an NVMe ring
buffer, and the backward pass replays each layer under autograd with the routed experts
streamed a second time. Only the LoRA adapters (fp32, ~590 MB) and their Adam moments
live in RAM.

*Türkçe okuyucu için: proje günlüğü [DEVAM.md](DEVAM.md), deney kayıtları
[Bulgular.md](Bulgular.md), fikir havuzu [Fikirler.md](Fikirler.md).*

## Status (9 September 2026)

| | |
|---|---|
| Forward pass | matches the independent C implementation [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) layer by layer over **all 93 layers** (cosine ≥ 0.988, 0.99984 at the output) |
| Backward pass | LoRA gradients, input and residual-bank gradients verified by central finite differences on KDA, MLA and block-boundary layers (relative error ≤ 2e-3) |
| End-to-end | perplexity 5.9 on an English paragraph, 2.2 on Turkish, 1.9 on Python; **learning demonstrated**: on two fixed 1024-token sequences the loss fell pass after pass (0.909 → 0.500 → 0.157 and 0.521 → 0.193, Bulgular.md §18) |
| Speed | forward ~110 s per layer at 1024 tokens, disk-bound at ~115 MB/s; a 1024-token training step (forward + backward + AdamW) 5.7 h |
| Now running | Turkish instruction run started 9 September: 400 Dolly-tr examples, 100 steps of 1024 packed tokens, LoRA lr 5e-4 cosine; ends ~3 October, then the pre-registered evaluation |

## Findings so far

From routing traces of five texts (Turkish, English, Chinese, Turkish news, Python) over
all 92 MoE layers ([Bulgular.md](Bulgular.md) §16-17, draft note in
[docs/measurement_note_draft.md](docs/measurement_note_draft.md)):

- **Concentration.** A batch touches 43-56 % of the experts a uniform router would;
  deeper layers concentrate more (200 unique experts at layer 92 for a 111-token text).
- **Domain over language.** Expert overlap between Turkish, English and Chinese versions
  of the same paragraph (0.35-0.39) equals the overlap between two unrelated passages in
  one language; prose vs Python is 0.20. A language signature exists only in layers 1-8.
- **Locality is per token, not per batch.** Consecutive tokens share 26 % of their
  experts (random: 1 %) and a 1-2 GB per-layer LRU hits 62-72 % in autoregressive
  decoding, but a training batch reads the union: 42 % of all experts at 128 tokens,
  53 % at 256, ~85 % at 1024. Caches and prefetching cannot help training-time
  offloading; a 200 GB static hot set saves 22 %.
- **The Turkish tax is in the tokenizer.** The same text costs 1.7× the tokens and 1.6×
  the bits per byte of English; routing for Turkish is not more diffuse.
- **Massive activations at the end.** In the last two MLA layers one token per text
  reaches a residual norm of 10⁴ (median 78); the C reference reproduces it.

## Hardware and layout

```
/home/ibox/calisma/LazyLora        this repository
/home/ibox/calisma/kimi-k3-in-c    reference C engine (oracle for the forward pass)
/mnt/disk2tb/hamza/kimi_k3_model_weights   1.56 TB checkpoint (96 safetensors shards)
/mnt/disk2tb/hamza/LazyLora_Workspace      traces, logs, datasets, eval corpora
/mnt/nvme/lazylora/k3trunk        packed non-expert weights (108.8 GB), served via an index overlay
/mnt/nvme/lazylora/{activations,checkpoints}
```

Every path is a `LAZYLORA_*` environment variable with defaults in
`lazy_lora/core/config.py`. Two virtualenvs: `~/venvs/lazylora` (CPU torch) and
`~/venvs/lazylora-cu` (torch cu126 for the routed-expert path on the GTX 1050).

## Engine

```
lazy_lora/
  core/config.py        paths, model/LoRA/training config, the synthetic-tensor gate
  core/attention.py     KDA (chunked, checkpointed recurrence) and gated MLA, block-residual mixer
  core/linear32.py      frozen-weight matmuls in fp32 (bf16 GEMM has no fast path on this CPU)
  core/lora_layer.py    LoRA A/B (fp32) with analytic gradients
  native/mxfp4_gemm.c   fused MXFP4 decode-and-dot, transposed product, fast decoder (OpenMP/AVX2)
  streaming/            pread shard reader with NVMe trunk overlay, packed-expert streamer, activation ring buffer
  trainer/lazy_trainer.py  one differentiable layer function shared by forward and backward;
                         routed experts as a single autograd.Function; bank gradient routing;
                         optional CUDA expert path; full atomic checkpoints with --resume
  monitor/trace.py      expert access trace format (layer, token, expert, weight) + reader
scripts/
  check_shards.py       offline checkpoint integrity (catches empty / truncated shards)
  compare_with_c_dump.py   replay the C engine's per-layer dump and compare
  verify_backward.py    finite-difference check of the backward on real weights (fp32)
  measure_routing.py / analyze_trace.py   routing traces and their statistics
  build_eval_corpus.py / build_eval_news.py / eval_perplexity.py   evaluation protocol
  build_train_set.py    Dolly-15k-tr selection (400 examples), packing + prompt masking in the data path
  train_lazy_lora.sh    training entry point (refuses to start on an incomplete checkpoint)
  watchdog.py + systemd/   unattended-run watchdog: progress, health, auto-resume, push to phone
  status.sh             one-screen status
```

Design rules that came out of the handoff review: a tensor missing on disk raises instead
of being replaced by random weights; the forward is never trusted until it matches the C
oracle; the backward is never trusted until finite differences say so; every measurement
is a file that can be re-read later.

## Quick verification

```bash
export PYTHONPATH=$PWD
~/venvs/lazylora/bin/python -m unittest lazy_lora.tests.test_reference_ops    # 8 op fixtures
~/venvs/lazylora/bin/python scripts/check_shards.py                             # checkpoint integrity
~/venvs/lazylora/bin/python scripts/compare_with_c_dump.py --dump <chdump> --ids 19180,11 --layers 13
~/venvs/lazylora/bin/python scripts/verify_backward.py --layer 1 --param-probes 16
bash scripts/run_mock_tests.sh                                                   # engine on synthetic weights
```

## Evaluation protocol

Fixed before training (DEVAM.md §16): bits per byte on a 2048-token slice of Turkish news
published after the model's release (baseline 0.455), with Turkish and English Wikipedia
slices as memorisation / forgetting controls (0.311 / 0.194). Success means the news
slice improves by ≥ 3 % while English Wikipedia degrades by ≤ 2 %; a negative result is
reported as such.

## Acknowledgements

The forward pass was validated against [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
(FareedKhan-dev), whose op-level fixtures and per-layer dump hook made layer-by-layer
comparison possible. Training data: `atasoglu/databricks-dolly-15k-tr` (CC BY-SA 3.0).
Evaluation text: Wikipedia (CC BY-SA 4.0), Anadolu Agency and BBC Türkçe (evaluation only).
