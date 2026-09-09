---
name: Verification report
about: You re-ran a check and got a different number from the published one
title: 'Verification: '
labels: verification
assignees: ''
---

<!-- This is the issue this project most wants to receive. Every published number comes
     from one laptop and one operator. The 93-layer cosine comparison and the five routing
     traces are in `evidence/`, so you can check those without asking us; the
     finite-difference numbers and anything needing the 1.56 TB checkpoint you cannot.
     A number of yours that disagrees with one of ours is not a complaint — it is the
     check working. -->

## Which check did you run?

- [ ] `bash scripts/quickstart.sh` (or `--fast`)
- [ ] `bash scripts/run_mock_tests.sh`
- [ ] `python3 -m unittest lazy_lora.tests.test_reference_ops` (op fixtures vs kimi-k3-in-c)
- [ ] `python3 scripts/verify_backward.py --layer N` (finite differences)
- [ ] `python3 scripts/check_shards.py` / `scripts/compare_with_c_dump.py` (needs the real checkpoint)
- [ ] Something else:

## The exact command

Including every `LAZYLORA_*` variable you set, so it can be re-run verbatim:

```sh

```

## The number you got

Paste the raw output that carries the number, not just the number.

```

```

## The number you expected

Where the published figure comes from — README table, `Bulgular.md` section, `DEVAM.md`
section, a docstring, or the script's own printed expectation — and what it says:

## Machine

| | |
|---|---|
| OS / kernel | |
| CPU model, cores, AVX2 present? | |
| RAM, and whether swap was in use | |
| Storage the check read from (NVMe / SATA SSD / USB HDD / tmpfs) | |
| Python version | |
| **torch version** (2.3 is the floor; the loader needs `uint16` `from_numpy`) | |
| numpy version | |
| This repository's commit (`git rev-parse --short HEAD`) | |
| For the op fixtures: the `kimi-k3-in-c` commit, and `LAZYLORA_REF_FIXTURES` | |

## How big is the disagreement?

- Relative difference between your number and the published one:
- Did it reproduce on a second run? Same seed?
- Anything that would explain it (different BLAS, a GPU torch build, a different fixture
  commit, a different sequence length or token count)?

## Notes on tolerances, so you can tell a real disagreement from a known one

- Op fixtures: seven of eight match at 1e-5 absolute / 1e-4 relative; the composite **MoE
  block fixture matches at 2e-4 absolute**, cosine 1.000000 (`abs_tol=2e-4` is hardcoded in
  `lazy_lora/tests/test_reference_ops.py`). A MoE mismatch inside 2e-4 is expected; anything
  outside it is not.
- Finite differences: four layers were checked — 1 (KDA + MoE, one bank entry), 3 (MLA), 12
  (block boundary), 13 (two bank entries) — on 4 tokens, fp32 engine, float64 loss reduction,
  adaptive epsilon, routing-flip detection. Worst relative error 9.1e-3, at layer 1, in two
  directions whose analytic derivative is about 3e-4, where the finite difference is the
  noisier of the two estimates; 3.6e-3 on the MLA layer; the rest at or under 2.0e-3.
- Timings and memory in the quickstart's documentation were **derived from reading the code,
  never measured** — a disagreement there is expected and still worth reporting, so the
  figures can be replaced with measurements.
- Step time and throughput figures for the main run were measured on one machine, on a USB
  enclosure: 6 h 59 m for a 1024-token packed sequence, and 110 MB/s aggregate read across
  the USB disk and the NVMe trunk (the 115 MB/s figure quoted elsewhere is the enclosure's
  own sequential benchmark, not an achieved aggregate). Yours will differ; that is the point.
