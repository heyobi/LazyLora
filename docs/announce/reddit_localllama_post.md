**Written** 10 September 2026.
**Status:** not posted.

## Title options

1. **Recommended.** `LoRA on Kimi K3 (2.78 T MoE) from a USB hard disk on a 7.6 GB laptop: 7.44 h a step, logs in repo`
   Why: model, disk, RAM and cost in one line, so nobody can read it as a speed claim, and "logs in repo" is what this audience opens first.
2. `Out-of-core LoRA on a 2.78 T MoE in 7.6 GB of RAM, all 93 layers checked against kimi-k3-in-c`
3. `I trained a LoRA adapter on Kimi K3 on a 2017 laptop with 7.6 GB RAM. Here is what you can verify.`

## Body

Kimi K3 is a 2.78 T parameter MoE, and its 1.56 TB checkpoint sits on a hard disk in a USB enclosure plugged into a 2017 laptop with 8 GB of RAM (7.6 GB usable). I am training a LoRA adapter on it out of core: one layer resident at a time, the 2.78 T base weights frozen on disk, only the 590 MB adapter in RAM. One layer's 896 routed experts are 15.7 GB, twice the RAM, so the streaming unit is the 17.5 MB expert.

The one-minute check is `evidence/cmp93_en34_2026-09-06.log`: my forward pass against FareedKhan-dev's independent C implementation, kimi-k3-in-c. All 93 layers agree at cosine 0.9857 or better; the lowest row is 0.985744 at layer 71, the output 0.999840. Scope: 34 English tokens, LoRA B zeroed. `evidence/traces/` holds raw routing records for five texts over all 92 MoE layers, and `scripts/analyze_trace.py` recomputes every routing number below with NumPy, no checkpoint. `scripts/quickstart.sh` builds a synthetic K3-shaped checkpoint and runs a forward pass, ten training steps and a finite-difference gradient check on a GitHub runner on every push, plus eight op fixtures vendored from kimi-k3-in-c under Apache-2.0. Its first run failed: the gradient check missed at 3.1e-2 against a 2e-2 tolerance because the step was below fp32 resolution against a tensor of norm 60.85. The gradient was right, the check was not; it is now noise-aware and that direction agrees to 2.09e-05 (Bulgular.md section 20).

| | |
|---|---|
| 1024-token step | 7.44 h (7.26 h and 7.62 h between the first three steps) |
| Step 1, forward / backward | 3 h 11 m 34 s / 3 h 48 m 07 s |
| Resident set | 4.0-4.7 GB, swap in use |
| Read throughput | 110 MB/s aggregate; 61 MB/s inside one MoE sweep |
| Cosine vs kimi-k3-in-c | minimum 0.985744 at layer 71, output 0.999840 |
| Trained / frozen | 590 MB adapter, 147 M parameters / 2.78 T base |

Expert overlap between Turkish, English and Chinese versions of one paragraph is 0.35-0.39, the same as between two halves of one text, while prose against Python is 0.20-0.21: expert choice tracks subject matter more than language. Consecutive tokens share 0.258 of their experts against 0.009 at random, and a 128-expert LRU hits 72 % when decoding, but a training batch reads the union, 42 % of experts at 128 tokens and about 85 % at 1024 (layers 0-12, an upper bound), so caching cannot help training-time offloading.

The proof run is memorisation of five examples, loss 0.909 to 0.157 on one fixed sequence; it proves the loop, not the model. The main run, 400 Turkish instruction examples for 100 steps, is at step 3 of 100 and ends around 9-11 October. The threshold went into git before it started (commit 6605306, tag preregistration-2026-09-08): Turkish news bits per byte from 0.455 to 0.441 or lower, English Wikipedia not past 0.198 from 0.194. I do not expect a large jump from 400 examples, and a negative result gets published as negative. The routed-expert adapter is one rank-16 adapter per layer shared by all 896 experts of that layer, not one per expert.

Repository: https://github.com/heyobi/LazyLora. Please poke holes, especially in the verification. FAQ.md already answers the obvious objections, each with the file to open; start with what it misses.
