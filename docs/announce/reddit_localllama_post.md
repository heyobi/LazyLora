**Written** 10 September 2026.
**Status:** not posted. Replace STEP_N with the step count on the day. The posting account
needs 10 sitewide karma to comment and at least 5 comment karma earned inside r/LocalLLaMA
to submit (rule of 24 April 2026); without that AutoModerator removes the post at once.
Text post, link in the body, flair Discussion, Monday to Wednesday 16:00-17:30 Turkey time,
then four hours at the keyboard. Re-snapshot evidence/forward_loss_main.jsonl first so the
about 7.4 h cadence is derivable from the bundle.

## Title options

1. **Recommended.** `LoRA on Kimi K3 (2.78 T MoE) from a USB hard disk on a 7.6 GB laptop: about 7.4 h a step, logs in repo`
   Why: model, disk, RAM and cost in one line, so nobody can read it as a speed claim, and "logs in repo" is what this audience opens first.
2. `Out-of-core LoRA on a 2.78 T MoE in 7.6 GB of RAM, all 93 layers checked against kimi-k3-in-c`

## Body

Kimi K3 is a 2.78 T MoE; its 1.56 TB checkpoint sits on a USB hard disk plugged into a 2017 laptop with 7.6 GB of RAM. I am training a LoRA adapter on it out of core: the non-expert weights of one layer at a time, its 896 experts streamed one by one since together they are 15.7 GB, base weights frozen, and the 590 MB adapter the only thing trained.

The one-minute check is `evidence/cmp93_en34_2026-09-06.log`, my forward pass against kimi-k3-in-c, FareedKhan-dev's independent C implementation: all 93 layers at cosine 0.9857 or better, output 0.999840, on 34 tokens with LoRA B zeroed. `evidence/traces/` holds raw routing records for five texts over all 92 MoE layers; `scripts/analyze_trace.py` recomputes every routing number below with NumPy alone. `scripts/quickstart.sh` builds a synthetic K3-shaped checkpoint and runs a forward pass, ten training steps and a finite-difference gradient check on a GitHub runner on every push, plus eight op-level checks against kimi-k3-in-c fixtures. Its first run failed: the gradient check missed at 3.1e-2 on a 2e-2 tolerance because the step was below fp32 resolution against a tensor of norm 60.85. The gradient was right; the check, made noise-aware, agrees to 2.09e-05 (Bulgular.md §20, in Turkish). It failed once more on a second runner for the same reason, a step too small against that runner's fp32 rounding, and now also has to move the loss by at least 2000 ulps.

| | |
|---|---|
| 1024-token step | 7.42 h, mean of 7.26 h, 7.62 h, 7.37 h |
| Step 1, forward / backward | 3 h 11 m 34 s / 3 h 48 m 07 s |
| Resident set | 4.0-4.7 GB, swap in use |
| Read throughput | 110 MB/s aggregate, 61 MB/s per MoE sweep |
| Cosine minimum | 0.985744, layer 71 |
| Trained / frozen | 590 MB adapter, 147 M parameters / 2.78 T base |

Turkish, English and Chinese versions of one paragraph share experts at Jaccard 0.35-0.39, about the same as two halves of one text (0.34-0.37); prose against Python is 0.20-0.21, so subject matters more than language. Consecutive tokens' expert sets have Jaccard 0.258 (0.009 for random pairs) and a 128-expert LRU hits 72 % when decoding, but a training batch reads the union, about 85 % of experts at 1024 tokens (layers 0-12, an upper bound), so an expert cache buys little for a training batch.

The proof run is memorisation of five examples, loss 0.909 to 0.157 on one fixed sequence: it proves the loop, not the model. The main run, 400 Turkish instruction examples over 100 steps, is at step STEP_N and ends 9-11 October. The threshold was committed before it started (commit 6605306, tag preregistration-2026-09-08): Turkish news bits per byte 0.455 to 0.441 or lower, English Wikipedia no worse than 0.198 from 0.194. I expect no large jump from 400 examples; a negative result gets published as negative. The adapter is one rank-16 LoRA per layer shared by all 896 experts, not one per expert.

Seven hours a step is useless for production fine-tuning; the point is that the cost is now a measured number, with the logs. None of the components are new: the idea is layer-streamed LoRA taken down to the expert level, and the related-work table in the README says what AirLLM, KTransformers, ZeRO-Infinity and the NVMe expert-streaming engines do that this does not.

Disclosure, since the rules ask for it: English is not my first language and I used Claude to tidy the wording of this post. The code was also written with heavy Claude Code assistance and the Co-Authored-By trailers are in the git log; the README says so on its first screen. The hardware, the runs, every number and every check against somebody else's implementation are mine, and the point of the evidence directory is that you do not have to take my word for any of it.

Repository: https://github.com/heyobi/LazyLora. Please poke holes, especially in the verification.

---

## App-safe body (no table; the Reddit mobile editor shows raw pipes)

Posted to r/LocalLLM (not r/LocalLLaMA, whose in-sub karma gate the account has not passed)
on 10 September 2026 with the author's own title. Same numbers, the table turned into lines.

Kimi K3 is a 2.78 T MoE; its 1.56 TB checkpoint sits on a USB hard disk plugged into a 2017 laptop (i7-7700HQ, 7.6 GB of RAM, a 2 GB GTX 1050 that only does the routed-expert matmuls). I am training a LoRA adapter on it out of core: the non-expert weights of one layer at a time, its 896 experts streamed one by one since together they are 15.7 GB, base weights frozen, and the 590 MB adapter the only thing trained.

The one-minute check is evidence/cmp93_en34_2026-09-06.log: my forward pass against kimi-k3-in-c, FareedKhan-dev's independent C implementation, all 93 layers at cosine 0.9857 or better, output 0.999840, on 34 tokens with LoRA B zeroed. evidence/traces/ holds raw routing records for five texts over all 92 MoE layers; scripts/analyze_trace.py recomputes every routing number below with NumPy alone. scripts/quickstart.sh builds a synthetic K3-shaped checkpoint and runs a forward pass, ten training steps and a finite-difference gradient check on a GitHub runner on every push, plus eight op-level checks against kimi-k3-in-c fixtures. Its first run failed: the gradient check missed at 3.1e-2 on a 2e-2 tolerance because the step was below fp32 resolution against a tensor of norm 60.85. The gradient was right; the check, made noise-aware, agrees to 2.09e-05.

The numbers, all from the logs in the repo:

- 1024-token step: about 7.4 h, the mean of the 7.26 h, 7.62 h, 7.37 h intervals between the first 4 steps
- step 1, forward / backward: 3 h 11 m 34 s / 3 h 48 m 07 s
- resident set: 4.0-4.7 GB, swap in use
- read throughput: 110 MB/s aggregate, 61 MB/s within one MoE sweep
- cosine minimum against the C engine: 0.985744, layer 71
- trained / frozen: 590 MB adapter (147 M parameters) / 2.78 T base

Turkish, English and Chinese versions of one paragraph share experts at Jaccard 0.35-0.39, about the same as two halves of one text (0.34-0.37); prose against Python is 0.20-0.21, so subject matters more than language. Consecutive tokens' expert sets have Jaccard 0.258 (0.009 for random pairs) and a 128-expert LRU hits 72 % when decoding, but a training batch reads the union, about 85 % of experts at 1024 tokens (layers 0-12, an upper bound), so an expert cache buys little for a training batch.

The proof run is memorisation of five examples, loss 0.909 to 0.157 on one fixed sequence: it proves the loop, not the model. The main run, 400 Turkish instruction examples over 100 steps, is at step STEP_N and ends 9-11 October. The threshold was committed before it started (commit 6605306, tag preregistration-2026-09-08): Turkish news bits per byte 0.455 to 0.441 or lower, English Wikipedia no worse than 0.198 from 0.194. I expect no large jump from 400 examples; a negative result gets published as negative. The adapter is one rank-16 LoRA per layer shared by all 896 experts, not one per expert.

Seven hours a step is useless for production fine-tuning; the point is that the cost is now a measured number, with the logs. None of the components are new: the idea is layer-streamed LoRA taken down to the expert level, and the related-work table in the README says what AirLLM, KTransformers, ZeRO-Infinity, Colibri, WARP and BigMoeOnEdge do that this does not.

Disclosure: English is not my first language and I used Claude to tidy the wording of this post. The code was also written with heavy Claude Code assistance and the Co-Authored-By trailers are in the git log; the README says so on its first screen. The hardware, the runs, every number and every check against somebody else's implementation are mine, and the point of the evidence directory is that you do not have to take my word for any of it.

Repository: https://github.com/heyobi/LazyLora. Please poke holes, especially in the verification.

