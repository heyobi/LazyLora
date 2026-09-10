**Written** 10 September 2026.
**Status:** not posted.

## Title options

1. **Recommended.** `LoRA on Kimi K3 (2.78 T MoE) from a USB hard disk on a 7.6 GB laptop: 7.44 h a step, logs in repo`
   Why: model, disk, RAM and cost in one line, so nobody can read it as a speed claim, and "logs in repo" is what this audience opens first.
2. `Out-of-core LoRA on a 2.78 T MoE in 7.6 GB of RAM, all 93 layers checked against kimi-k3-in-c`
3. `I trained a LoRA adapter on Kimi K3 on a 2017 laptop with 7.6 GB RAM. Here is what you can verify.`

## Body

Kimi K3 is a 2.78 T parameter MoE; its 1.56 TB checkpoint sits on a USB hard disk plugged into a 2017 laptop with 8 GB of RAM (7.6 GB usable). I am training a LoRA adapter on it out of core: one layer resident at a time, streamed expert by expert since a layer's 896 experts are 15.7 GB, base weights frozen, only the 590 MB adapter in RAM.

The one-minute check is `evidence/cmp93_en34_2026-09-06.log`, my forward pass against kimi-k3-in-c, FareedKhan-dev's independent C implementation: all 93 layers at cosine 0.9857 or better, output 0.999840, on 34 tokens with LoRA B zeroed. `evidence/traces/` holds raw routing records for five texts over all 92 MoE layers; `scripts/analyze_trace.py` recomputes every routing number below with NumPy, no checkpoint. `scripts/quickstart.sh` builds a synthetic K3-shaped checkpoint and runs a forward pass, ten training steps and a finite-difference gradient check on a GitHub runner on every push, plus eight kimi-k3-in-c op fixtures. Its first run failed: the gradient check missed at 3.1e-2 on a 2e-2 tolerance because the step was below fp32 resolution against a tensor of norm 60.85. The gradient was right; the check, made noise-aware, agrees to 2.09e-05.

| | |
|---|---|
| 1024-token step | 7.44 h, mean of 7.26 h and 7.62 h |
| Step 1, forward / backward | 3 h 11 m 34 s / 3 h 48 m 07 s |
| Resident set | 4.0-4.7 GB, swap in use |
| Read throughput | 110 MB/s aggregate, 61 MB/s per MoE sweep |
| Cosine minimum | 0.985744, layer 71 |
| Trained / frozen | 590 MB adapter, 147 M parameters / 2.78 T base |

Turkish, English and Chinese versions of one paragraph share experts at 0.35-0.39, the same as two halves of one text; prose against Python is 0.20-0.21, so subject matters more than language. Consecutive tokens share 0.258 of their experts (0.009 at random) and a 128-expert LRU hits 72 % when decoding, but a training batch reads the union, about 85 % of experts at 1024 tokens (layers 0-12, an upper bound), so caching cannot help training-time offloading.

The proof run is memorisation of five examples, loss 0.909 to 0.157 on one fixed sequence: it proves the loop, not the model. The main run, 400 Turkish instruction examples over 100 steps, is at step 3 and ends 9-11 October. The threshold was committed before it started (commit 6605306, tag preregistration-2026-09-08): Turkish news bits per byte 0.455 to 0.441 or lower, English Wikipedia no worse than 0.198 from 0.194. I expect no large jump from 400 examples; a negative result gets published as negative. The adapter is one rank-16 LoRA per layer shared by all 896 experts, not one per expert.

Repository: https://github.com/heyobi/LazyLora. Please poke holes, especially in the verification; FAQ.md already answers the obvious objections.
