# Quickstart: run the engine yourself

> **This document describes scripts that have not been run yet.** `scripts/quickstart.sh`,
> `scripts/make_tiny_model.py`, `scripts/demo_generate.py` and `scripts/export_traces.py`
> were written by reading the engine, line by line, while the machine that could have run
> them was busy with a 29-day training job. **Every runtime, every memory figure and every
> expected output in Part 1 below is derived from the code, not measured** — including the
> per-step times in the table, the peak resident set, and the byte counts in "Roughly what
> you should see". The first person to run `scripts/quickstart.sh` on a free machine is
> validating it; `.github/workflows/quickstart.yml` does that on a public runner, and its
> first green run is the first evidence any of this works. Please report whatever it gets
> wrong.
>
> Part 2 is different: those numbers come from the main run on the author's machine and are
> measured. Each one says which run it came from.

LazyLoRA fine-tunes a LoRA adapter on **Kimi K3** — 2.78 trillion parameters, 93 layers,
896 routed experts per layer — out of core, on one consumer laptop. The checkpoint is
1.56 TB and lives on a USB hard disk. Almost nobody who reads this repository has that
checkpoint, and the interesting question is not "can I download 1.56 TB" but "is this
engine really doing what it says".

So there are two paths.

**Part 1** runs the whole engine end to end in two or three minutes with no checkpoint at
all: a tiny Kimi-K3-shaped model is generated on disk and the real code streams it, routes
through it, differentiates it and trains on it. **Part 2** is for the small number of
people who do have the real weights.

Before either of them, there is a path that runs nothing at all. `evidence/` in this clone
is 6.2 MB of the actual output of the runs the README describes — the 93-row layer-by-layer
comparison against an independent C implementation, and all five expert-routing traces over
all 92 MoE layers. Reading it needs no checkpoint, no GPU and no install beyond numpy, and
it is the quickest way to check that the numbers being claimed are the numbers the machine
produced. See [the evidence bundle](#the-evidence-bundle-nothing-to-run) below.

---

## The evidence bundle, nothing to run

`evidence/` is 6.2 MB in the clone, and `evidence/README.md` describes every file in it.
`SHA256SUMS` covers all 25 of them:

```bash
cd evidence && sha256sum -c SHA256SUMS && cd ..
```

**`evidence/cmp93_en34_2026-09-06.log`** is the whole forward comparison against
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) — 98 lines, one row per
layer for all 93 layers, each with the cosine similarity, the maximum absolute difference,
both implementations' standard deviations and the number of experts that layer read, and
the run's totals (2869 s, 426.59 GB read) at the end:

```bash
grep -o 'cosine=[0-9.]*' evidence/cmp93_en34_2026-09-06.log | sort -t= -k2 -n | head -3
```

The lowest of the 93 is 0.985744 at layer 71; the dip runs from layer 68 to layer 72
(0.989709, 0.987459, 0.986975, 0.985744, 0.987909), with the three lowest rows of the whole
file at 69–71, and the last layer is 0.999840. You do not have to take any of that from the
README; they are rows in that file. (Earlier drafts called 0.988 at layer 72 the minimum.
That was the lowest of the nine layers spot-checked in `Bulgular.md` §17.1; the log is what
settles it.)

**`evidence/traces/`** holds the five expert-routing traces, one directory per text, each
covering all 92 MoE layers: which 16 of the 896 experts every token was routed to, and with
what combining weight. Every routing number in `docs/measurement_note.md` and in
`Bulgular.md` §16–17 is recomputable from them, with `scripts/analyze_trace.py` or with
numpy and a `struct.unpack_from` loop — the loop is twenty lines and `evidence/README.md`
prints it, the format is defined in `lazy_lora/monitor/trace.py`. No checkpoint, no GPU,
nothing this project's machine has that yours does not. (Run without `--prefix`,
`analyze_trace.py` rewrites `analysis.json` in the trace directory — that is how the
committed copies were made — so copy the directory first, or use `--prefix N`, which only
prints, if you want `sha256sum -c SHA256SUMS` to keep passing.)

**`evidence/forward_loss_main.jsonl`** and **`forward_loss_proof.jsonl`** are one JSON
object per completed forward pass — step, loss, perplexity, Unix timestamp. The proof run's
step times and the main run's forward are subtractions of two of those timestamps, so that
arithmetic is checkable. The main run's backward (3 h 48 m 07 s) and its 6 h 59 m 41 s total
are not in the bundle — a line is written when the forward ends, and nothing here marks the
end of a backward — and come from the trainer's printed timings:

```bash
tail -5 evidence/forward_loss_proof.jsonl    # the five proof-run steps, 0.909 -> 0.157
```

They are two snapshots of one rolling log rather than one file per run — the main file is
the proof file with the main run's step 1 appended — so earlier smoke-test rows are in them
too: a `nan` from a run that was killed, and a series at loss ≈ 12.0 from runs on random
weights. The proof run is the last five records of the proof file; the main run's step 1 is
the last record of the main one.

What is *not* in the bundle: the 1.56 TB checkpoint, the C engine's per-layer dump, the
108.8 GB NVMe trunk and the 1.8 GB training checkpoints. Nor is the finite-difference
output — that harness prints its table to the terminal rather than to a file, so those
numbers are quoted from `DEVAM.md` §11 and reproducing them means Part 2 §3 on real weights.

---

## Part 1 — the two-minute path

### What you need

* Linux or macOS. The loader's hot path is `os.pread`
  (`lazy_lora/streaming/mmap_loader.py:269`), which does not exist on Windows. WSL is fine
  — step 5's hardware-profiler check reports SKIP there unless a D: drive is mounted, which
  is informational and cannot fail the run.
* Python 3.10 or newer.
* Two packages: `numpy>=1.24` and `torch>=2.3`. That is all `requirements.txt` contains,
  so `pip install -r requirements.txt` and `pip install -e .` install the same thing.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.3'   # CPU build
pip install -e .                                                           # numpy, metadata
# or: pip install -r requirements.txt                                      # the same two
```

`torch >= 2.3` and `numpy >= 1.24` are enough, and `pip install -e .` from the clone
installs exactly those two and puts `lazy_lora` on the path, so nothing needs `PYTHONPATH`
set by hand. **2.3 is a hard floor, not a preference:** the loader reinterprets a numpy
`uint16` array as bfloat16 with `torch.from_numpy(...).view(torch.bfloat16)`
(`lazy_lora/streaming/mmap_loader.py:326`), and `torch.from_numpy` only learned `uint16` in
2.3. The engine imports nothing else at run time: it parses safetensors headers itself, so
even `safetensors` is not a dependency. `transformers` is needed only where a real
tokenizer is (Part 2) — `pip install -e '.[data]'`.

Optional: a `gcc` with OpenMP on an AVX2 host. The MXFP4 expert kernel
(`lazy_lora/native/mxfp4_gemm.c`) is built on first use if it can be; if it cannot, every
caller falls back to a slower PyTorch decoder that computes the same thing, and the run
continues. Nothing breaks without it.

* About 1.5 GB of free RAM and 1 GB of temporary disk (500 MB and almost no disk with
  `--fast`) — derived from the code, not measured. The sandbox is created under `$TMPDIR`;
  if `/tmp` is a tmpfs on your system, the ~330 MB step 5 writes is RAM rather than disk, on
  top of the ~330 MB the same test holds in a Python dict and the ~400 MB of imported
  torch. Set `TMPDIR` to a real filesystem or pass `--dir`.

### The command

```bash
git clone https://github.com/heyobi/LazyLora && cd LazyLora
bash scripts/quickstart.sh
```

That is the whole thing. Everything it writes goes into one `mktemp -d` directory that is
deleted when it exits (`--keep` keeps it, `--dir PATH` chooses it). It touches nothing
else. It sets the `LAZYLORA_*` path variables into that directory and it **unsets**
`LAZYLORA_ALLOW_SYNTHETIC` for every step that involves the tiny model — that flag is the
gate that lets the engine substitute a random tensor for one it cannot find
(`lazy_lora/core/config.py:166-182`), and keeping it closed is half of what the demo
demonstrates.

Useful flags: `--fast` skips the older synthetic-weight suite (step 5, the slowest);
`--keep` keeps the sandbox; `QS_STEPS`, `QS_SEQ_LEN`, `QS_FD_LAYERS`, `QS_FD_PROBES` and
`QS_SEED` tune the run.

### What each step proves

| # | Step | Time, derived from the code | What a pass means |
|---|---|---|---|
| 1 | Op fixtures vs the C reference | 2–5 s | SiTU-GLU, RMSNorm, the causal short convolution, the KDA decay gate, the router, the block-residual mixer, MLA and the whole latent MoE block agree with [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) — an implementation written by somebody else, in another language. Seven of the eight match at 1e-5 absolute / 1e-4 relative; the MoE block matches at 2e-4 absolute with cosine 1.000000, which is the MXFP4 decode path's own rounding (`lazy_lora/tests/test_reference_ops.py` hardcodes `abs_tol=2e-4` for it). This is the only external ground truth in the repository. |
| 2 | `make_tiny_model.py` | 3–10 s | A real checkpoint now exists: real safetensors bytes, real MXFP4 expert blocks packed two 4-bit codes to a byte with one E8M0 scale per 32 channels, every tensor the loader will ask for. The script re-reads its own output with the engine's indexer and says so. |
| 3 | `check_shards.py` | < 1 s | The integrity checker that guards the real run — the one that exists because two empty shards were once counted as healthy — accepts this checkpoint: every file present, size equal to header plus data, every tensor the index names really in its header. |
| 4 | A real forward pass | 5–20 s | The engine streamed weights off disk one layer at a time, ran KDA linear attention on three layers and gated MLA on one, kept and mixed the residual bank, routed every token through the MXFP4 experts, and produced a loss sitting on the uniform prior `ln(V)` — which is exactly where an untrained model belongs. A number far from `ln(V)` would mean the pipeline is not computing what it thinks. |
| 5 | The synthetic-weight suite | 40–70 s | Reported in two parts. `5a` is the hardware profiler, which describes the machine rather than the engine and is informational only. `5b` is the older harness: the 93-layer activation ring buffer round-trips bit-exactly, the top-k router selects and normalises correctly at full Kimi K3 dimensions, one LoRA gradient matches a central difference, checkpoints save and reload. This is the one place `LAZYLORA_ALLOW_SYNTHETIC=1` is legitimate. It writes and reads about 330 MB. |
| 6 | Ten real training steps | 20–60 s | The complete out-of-core loop: layer-boundary activations spilled to disk, every layer replayed under autograd in the backward pass, the routed experts streamed a second time, AdamW applied to the adapter, an atomic checkpoint written — and reloaded into a fresh engine so that every LoRA tensor and every Adam moment comes back identical. The loss falls, and `lora_B`, which is exactly zero at initialisation, is not zero any more. |
| 7 | Finite differences | 15–45 s | The project's central credibility claim, reproduced on your laptop: the analytic gradient of a LoRA tensor, of the layer input and of the residual bank each agree with a central finite difference of the forward pass to better than 2 %. It is the same method `scripts/verify_backward.py` runs against the real checkpoint. |
| extra | Native kernel A/B | 10–20 s | The same forward loss with the hand-written AVX2/OpenMP MXFP4 kernel and with the PyTorch dequantiser. Agreement to fp32 rounding means the fast path is not quietly a different model. Skipped, honestly, when the kernel cannot be built. |

Total: about two minutes with `--fast`, three or four with step 5 included; peak RSS around
600 MB, most of it importing PyTorch, except during step 5, which holds about 330 MB of
activations of its own. **None of these times or sizes has been measured** — they are
reasoned from the work each step does (tensor counts, byte counts, the number of layer
sweeps) on a machine roughly like the author's. Treat them as an order of magnitude, and
report what you actually see.

### The tiny model

`scripts/make_tiny_model.py` writes about 8.3 MB — 8,324,144 bytes of tensor data across
two shards, plus two small headers, a `config.json` and an index: 4 layers (layer 0 dense, layer 1 MLA,
layers 2 and 3 KDA), hidden size 256, 4 attention heads of 96, 8 experts per MoE layer with
top-2 routing, a 2048-token vocabulary, MXFP4 experts, two shards and an index. Its header
comment records, tensor by tensor and shape by shape, which line of the loader demands
each choice, because the values are not arbitrary: `v_head_dim` must equal `head_dim` or
MLA's output gate does not line up; the expert widths must be multiples of 32 or the C
kernel multiplies uninitialised stack memory; the vocabulary must exceed 355 because the
dataset iterator's byte fallback emits `byte + 100`.

Every tensor is drawn from SHAKE-256 keyed by `(seed, tensor name)`, so the same seed gives
byte-identical shards on any machine and any NumPy version. The script prints each shard's
sha256; two people running `--seed 0` must see the same digests.

It is not Kimi K3 and it knows nothing: the values are deterministic noise. The bytes, the
file format, the quantisation and the geometry are real, which is what makes the engine's
behaviour on it meaningful — the forward and the backward read the same bytes off the same
disk through the same loader, so the loss curve and the gradient check mean something even
though the weights mean nothing.

### Roughly what you should see

```
[2/7] Generating a tiny Kimi-K3-shaped checkpoint
model dir     : /tmp/lazylora-quickstart-XXXX/model
geometry      : 4 layers (MLA at 1-based [2], dense below 1), hidden 256, attn 4x96 = 384,
                experts 8 top-2 (64 wide in a 128 latent), vocab 2048
tensors       : 250
  model-00001-of-00002.safetensors   174 tensors    6.77 MB data + x.x kB header
  model-00002-of-00002.safetensors    76 tensors    1.56 MB data + x.x kB header
verified      : all 250 tensors resolve through the engine's own SafetensorsIndex ...

[4/7] A real forward pass through the real engine
      64 tokens through 4 layers in x.x s, x.xx MB streamed off disk
      experts touched per layer: [0, 8, 8, 8]  (dense layers show 0)
      loss 7.6xxx   uniform prior ln(2048) = 7.6246   perplexity 20xx.x

[6/7] Real training steps: forward, backward, AdamW, checkpoint, reload
  📉 [FORWARD LOSS] step 1: 7.62xx  (perplexity 20xx.x)
  ...
      10 steps on one fixed 64-token sequence in xx.x s: 7.62xx -> <lower>

[7/7] Finite differences: is the backward pass the gradient of the forward pass?
      direction                       analytic  central diff       eps   rel err
      param down_lora.B            -1.234567e-02 -1.234512e-02   1.6e-01  4.5e-05
      h_in                          3.210000e-02  3.209900e-02   6.2e-02  3.1e-05
      residual bank                 ...
      worst relative error 4.50e-04 (tolerance 2e-02)
```

The exact numbers will differ; the shape of the output should not. The two shard sizes are
the exception: the split is on a layer boundary, so shard 1 carries the globals and layers
0–2 (6,766,240 bytes) and shard 2 carries layer 3 alone (1,557,904 bytes). Those two counts
follow from the geometry and should come out exactly, on any machine, at any seed.

### When something fails

| Symptom | What it means, what to do |
|---|---|
| `MISSING numpy (...)` or `MISSING torch (...)` | Install them as above: `pip install -e .` from the clone, or `pip install -r requirements.txt` — that file is the same two packages (`numpy>=1.24`, `torch>=2.3`) with a header comment pointing at the CPU wheel index. |
| `MISSING torch 2.2.x is older than the required 2.3` | Upgrade torch. `torch.from_numpy` on a numpy `uint16` array is how the loader gets bfloat16 out of the file (`mmap_loader.py:326`), and that overload does not exist before 2.3. |
| Step 1 says `SKIP: fixtures absent` | Should no longer happen: the fifteen fixtures are committed at `tests/fixtures/ops/`, copied from the reference repository under Apache-2.0. If you see the skip, your checkout is missing that directory. `LAZYLORA_REF_FIXTURES` still overrides the location if you want to compare against your own clone of kimi-k3-in-c. |
| `[!] MXFP4 native kernel not available` | Informational. There is no gcc with OpenMP, or the host is not x86-64 with AVX2. The engine uses the PyTorch decoder instead: same results, slower. Step "extra" will report SKIP. |
| Step 5 says `SKIP` on `5a hardware profile` | Expected on most machines. That test asserts the machine has a second, non-system volume and that a specific directory exists — or, on WSL, that a `/mnt/d` path is recommended, which only happens when a D: drive is actually mounted, so it fails on WSL without one. It describes the author's laptop, not the engine. It is reported but deliberately cannot fail the run; the other eight suites in step 5 can. |
| Step 4 loss is far from `ln(V)` | This is a real failure and worth reporting: an untrained model with random weights must sit at the uniform prior. Something in the streaming or the layer stack is wrong. |
| Step 7 reports a MISMATCH | Also worth reporting, and the most serious of the failures: it means the backward pass is not the gradient of the forward pass. Note which direction failed. A single mismatch on a `param` direction with a large `eps` is more likely to be a routing flip inside the step (the script retries these automatically) than a broken gradient; a mismatch on `h_in` or the bank is not. |
| `MissingTensorError: ... is not on disk` | The engine refused to invent a tensor, which is the intended behaviour. In the quickstart it means the tiny checkpoint is incomplete — regenerate it. Never "fix" this by setting `LAZYLORA_ALLOW_SYNTHETIC=1`; that flag makes the numbers meaningless. |
| Everything is very slow | The KDA recurrence is a Python loop over tokens, and the engine deliberately runs frozen matmuls in fp32 because bf16 GEMM has no fast path on the author's CPU. On four tiny layers this is still seconds. If it is minutes, check that `QS_SEQ_LEN` is not large. |

---

## Part 2 — with the real 1.56 TB checkpoint

For the few who have Kimi K3 on disk. Point the engine at it:

```bash
export LAZYLORA_MODEL_DIR=/path/to/kimi_k3_model_weights   # 96 safetensors shards, 1.56 TB
export LAZYLORA_WORKSPACE_DIR=/path/with/room              # traces, logs, datasets, eval
export LAZYLORA_FAST_SCRATCH_DIR=/path/on/ssd              # activation ring buffer, checkpoints
pip install -e '.[data]'                                   # the package, plus the tokenizer
```

Optional: `LAZYLORA_TRUNK_DIR` points at the packed non-expert "trunk" (108.8 GB) if you
have built one; set it to the empty string to disable the overlay.

The timings below are measured on the author's machine: an i7-7700HQ with 4 cores, 7.6 GB
of RAM, the routed experts on a 2 TB disk in a USB enclosure, and the packed non-expert
trunk on NVMe. They are dominated by reading.

Three read numbers appear in this repository and they are not the same number:

* **110 MB/s aggregate, measured.** The trainer process had read 3,219,659,335,955 bytes
  through `read()` after 8 h 06 m 57 s of the main run. That is the whole engine, across
  the USB disk (routed experts) and the NVMe trunk (everything else) together.
* **61 MB/s effective within one layer sweep, measured** — 14.5 GB per MoE layer in 238 s
  at 1024 tokens (`Bulgular.md` §16.5). The reader thread idles during compute; pipelining
  it is open work.
* **~115 MB/s, not measured here.** That is the USB enclosure's own sequential benchmark,
  i.e. a specification the engine does not reach. It is never an achieved rate.

On faster hardware these scale roughly with sequential read speed.

### 1. Checkpoint integrity — minutes

```bash
python scripts/check_shards.py
```

Reads only the headers of the 96 shards, checks each file's size against `8 + header +
data`, and checks that every tensor named in `model.safetensors.index.json` is really in
the header of the shard that claims it. `--hash <shard>` adds sha256 for comparison with
the LFS oids on the Hub, at roughly three minutes per 17 GB shard. Exit code 0 means the
checkpoint is complete. `scripts/train_lazy_lora.sh` refuses to start a run if this fails,
because two zero-byte shards were once reported clean by a weaker scan and cost two whole
layers.

### 2. The forward pass against the C reference — hours

```bash
# in kimi-k3-in-c, with K3_DUMP_H set, produce h_layer_NNN.bin for the same token ids
python scripts/compare_with_c_dump.py --dump <dump dir> --ids 19180,11 --layers 13
```

Replays the reference engine's per-layer hidden-state dump through this forward pass and
reports cosine and max difference per layer. It needs the dump, not the C engine, and only
the shards those layers live in — so it can be run while a download is still finishing.
Cost: one layer sweep per layer compared, at a handful of tokens.

What the author measured, on 34 tokens: all 93 layers at cosine ≥ 0.9857, the lowest at
layer 71, and 0.999840 at the output. **The full comparison log is in this repository**, at
`evidence/cmp93_en34_2026-09-06.log` — all 93 rows, with the max difference, both standard
deviations and the expert count per layer, so your own run can be compared against it line
by line rather than against the nine cosines discussed in `Bulgular.md` §17.1. That run took
2869 s and read 426.59 GB.

### 3. Finite differences on real weights — minutes to an hour, per layer

```bash
python scripts/verify_backward.py --layer 1  --ids 19180,11,1632,691 --param-probes 16
python scripts/verify_backward.py --layer 12 --param-probes 8      # a block-boundary layer
python scripts/verify_backward.py --layer 13 --param-probes 8      # two bank entries
```

Forces `LAZYLORA_COMPUTE_FP32=1`, runs the forward up to layer L, then compares the
production backward's gradients — LoRA tensors, layer input, residual bank — against
central differences with an adaptive step, a float64 loss reduction and routing-flip
detection. The cost is the forward to layer L (L layer-sweeps at 4 tokens) plus about four
single-layer evaluations per probe, so layer 1 is single-digit minutes and layer 13 is a
few tens of minutes.

What has actually been checked (`DEVAM.md` §11 and lines 268–276): four layers, on 4
tokens — layer 1 (KDA + MoE, one bank entry), layer 3 (MLA), layer 12 (a block boundary)
and layer 13 (two bank entries). Layers 1 and 3 were swept over all 16 LoRA tensors plus
the input and residual-bank directions; layers 12 and 13 covered the input and bank
directions and that layer's LoRA tensors.

Published result: worst relative error 3.6e-3 on the MLA layer (3) and ≤ 2.0e-3 on layers
1, 12 and 13, **except** two directions at layer 1 whose analytic derivative is about 3e-4,
where it reaches 9.1e-3 — at that magnitude the finite difference is the noisier of the two
estimates. The bare figure "≤ 2e-3" is a cherry-pick and should not be quoted without that
qualification.

This is the same check Part 1 step 7 runs on the tiny model. If you only have time for one
thing on real weights, this is the one.

### 4. One training step — about 7 hours

```bash
bash scripts/train_lazy_lora.sh --steps 1 --seq-len 1024 --no-pack
```

Forward, backward and AdamW on a 1024-token packed sequence: **6 h 59 m 41 s**, measured on
step 1 of the main run — started 2026-09-09 12:53:32, forward loss written 16:05:06 (that
line is in `evidence/forward_loss_main.jsonl`), backward finished 19:53:13. That is a
forward of 3 h 11 m 34 s (123.6 s per layer over 93 layers) and a backward of 3 h 48 m 07 s
(147.2 s per layer). That step wrote no checkpoint, because the main run saves every fifth
step. Resident set 4.0–4.7 GB, with swap in use; the highest peak ever recorded anywhere in
this project is 6.24 GB, on an earlier 256-token step, and it is not the current peak.

The step is bandwidth-bound: see the three read numbers above. Add `--forward-only` to stop
after the loss, which costs a little under half of the total.

An earlier draft quoted 5.7 hours and 110 s per layer. Those are the **proof run's** rates,
whose two packed sequences held about 541 tokens each rather than a full 1024, and they
must not be quoted for the main run. Step 1 measured 6 h 59 m 41 s; the cadence over the
first three steps is 7.44 h (7.26 h and 7.62 h between them), so a hundred steps is about
31 days, finishing around 9-11 October 2026.

### 5. Before/after generation — an estimated five minutes per token

Write a prompts file first. One JSON object per line, in the same shape as the training
data (`instruction`, optional `input`), or a plain `.txt` with one prompt per line:

```jsonl
{"instruction": "Boğaziçi Köprüsü ne zaman açıldı?"}
{"instruction": "Aşağıdaki paragrafı iki cümlede özetle.", "input": "..."}
```

```bash
python scripts/demo_generate.py --checkpoint <lazy_lora_step_00100.pt> \
                                --prompts my_prompts.jsonl \
                                --max-new-tokens 24 --dry-run
```

Drop `--dry-run` when the estimate it prints is one you are willing to pay (estimated from
the per-sweep read volume; no generation has been run on this engine yet, so there is no
measured cost per token). There is no KV cache: every generated token is a full 93-layer
sweep, which reads the top-16 routed experts of each of the 92 MoE layers off the USB disk
— 16 × 17.5 MB × 92, about 26 GB — plus the 108.8 GB packed trunk off the NVMe. At the
110 MB/s aggregate and 61 MB/s per-sweep rates measured during training, the expert reads
alone are four to seven minutes, which is where "about five minutes per token" comes from.
On that estimate a 24-token answer is about two hours and answering three prompts with the
adapter both off and on is about twelve. `--minutes-per-token` overrides the assumption;
the first real run replaces it. Each token is flushed to the output JSON as it arrives and
`--resume` continues the same file, so an interrupted run loses at most one token. Run it
under `nohup` or systemd.

"Adapter off" is not a second model: LoRA's B matrix is zero at initialisation, so a
trainer with `B = 0` computes the base model exactly. The output file carries both answers
per prompt, side by side, with the prompt built through the model's own chat template —
the same generation prompt the training data path masks the loss up to.

### 6. The evaluation — hours per chunk

```bash
python scripts/build_eval_corpus.py && python scripts/build_eval_news.py
python scripts/eval_perplexity.py --corpus tr_news,tr_wiki,en_wiki --chunks 2 --tag baseline
```

Bits per byte on fixed 2048-token slices; each chunk is one full sweep, so budget the best
part of a day for a full set. The protocol was fixed **before** training: the primary
metric is Turkish news published after the model's release (baseline 0.455, success
threshold ≤ 0.441), with Turkish and English Wikipedia as memorisation and forgetting
controls (0.311 and 0.194; English must not degrade past 0.198).

**No evaluation result exists yet.** The 100-step run finishes around 9-11 October 2026 —
100 steps at the measured 7.44 h cadence is about 31 days — and the evaluation runs after it. Nothing
in this repository says the evaluation succeeded, or that Turkish improved, because it has
not been run. A negative result will be reported as a negative result.

---

## What none of this shows

Part 1 shows that the mechanism works and that its gradients are real. It says nothing
about Kimi K3, because the weights are eight megabytes of deterministic noise. It also has
not been run yet: until `.github/workflows/quickstart.yml` is green, or somebody runs
`scripts/quickstart.sh` on a free machine, Part 1 is a claim about code that has been read
rather than a claim about code that has run.

The evidence bundle shows a third thing and only that: what the runs actually printed. It
lets a reader check the numbers quoted here against the files they were read off, and
recompute every routing statistic from the traces. It cannot show that those traces came
from Kimi K3 rather than from something else — only re-running Part 2 against the real
checkpoint can.

Part 2, on the real checkpoint, shows that this engine computes the same forward pass as
an independent implementation, that its backward pass is the true gradient of that forward
pass, and that a LoRA adapter on a 2.78-trillion-parameter model can be moved by gradient
descent on a laptop with 7.6 GB of RAM.

Neither shows that the model became better at anything, and neither is "training Kimi K3":
the 2.78 trillion parameters are frozen on disk throughout, and what is trained is a LoRA
adapter. The proof-of-learning run (`Bulgular.md` §18) is deliberate memorisation of five
examples: the loss falls because the model is being made to memorise them, which
demonstrates the loop and nothing more — not that the model learned anything. The question
of whether the adapter helps is what the pre-registered evaluation is for, and it has not
been answered.
