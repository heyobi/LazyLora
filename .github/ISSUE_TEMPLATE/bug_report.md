---
name: Bug report
about: Something in the engine, the scripts or the quickstart does not work
title: ''
labels: bug
assignees: ''
---

<!-- If instead you re-ran a check and got a different NUMBER from the published one,
     please use the "Verification report" template — that one is more useful to us. -->

## What you ran

The exact command, copied, including any `LAZYLORA_*` variables you set:

```sh

```

## What happened

Paste the output. If it is long, the last 50 lines and the traceback are enough.

```

```

## What you expected instead

## Machine

| | |
|---|---|
| OS / kernel | |
| CPU (and AVX2? `grep -o avx2 /proc/cpuinfo \| head -1`) | |
| RAM, and swap | |
| Where the paths point (`LAZYLORA_WORKSPACE_DIR`, `LAZYLORA_FAST_SCRATCH_DIR`, `LAZYLORA_MODEL_DIR`) | |
| Filesystem and device type of the scratch path (tmpfs? NVMe? USB?) | |
| Python (`python3 -V`) | |
| torch (`python3 -c 'import torch; print(torch.__version__)'`) — 2.3 is the floor | |
| numpy | |
| This repository's commit (`git rev-parse --short HEAD`) | |

## Does it reproduce with the quickstart?

- [ ] `bash scripts/quickstart.sh --fast` — result:
- [ ] `bash scripts/run_mock_tests.sh` — result:
- [ ] Not applicable / did not get that far

The quickstart has never been executed on the author's machine (it was written while that
machine was busy with the training run), so "the quickstart itself is broken" is an entirely
expected report and a useful one.

## Anything else

Was `LAZYLORA_ALLOW_SYNTHETIC` set in your shell? It must be unset for everything except
`scripts/run_mock_tests.sh`; if it was set, please say so — it changes what the engine does
when a tensor is missing.
