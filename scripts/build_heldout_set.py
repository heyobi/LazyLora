#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ibrahim Polat
"""
Select a held-out Dolly-tr set: examples that pass the same filter as the 400 training
examples (scripts/build_train_set.py) but are not among them.

    python scripts/build_heldout_set.py [--n 100] [--seed 1]

Writes <workspace>/datasets/dolly_tr_heldout_<n>.jsonl and a manifest with the selection
rules and the sha256 of the training file it was disjoint from. Declared on 10 September
2026, one day into the main run and a month before its evaluation, as a SECONDARY metric:
the mean masked answer loss (the same quantity the trainer logs in forward_loss.jsonl) on
these examples, base model against adapter. It measures what the run actually optimises,
instruction-following loss on unseen examples from the same distribution, and is added
because the pre-registered primary (news bits per byte) is expected to move little on 51k
trained tokens. The primary stays primary; this cannot replace it, only sit beside it.
"""
import argparse, collections, hashlib, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402
from build_train_set import KEEP, TR, load  # noqa: E402  (same filter, same loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train", default="dolly_tr_400.jsonl")
    args = ap.parse_args()
    D = get_default_config().paths.dataset_dir
    raw = [p for p in os.listdir(os.path.join(D, "raw")) if p.startswith("dolly") and p.endswith((".json", ".jsonl"))][0]
    rows = load(os.path.join(D, "raw", raw))
    rows = [x for r in rows for x in (r if isinstance(r, list) else [r])]   # a JSON array or concatenated objects
    train_path = os.path.join(D, args.train)
    train_ins = {json.loads(l)["instruction"] for l in open(train_path, encoding="utf-8") if l.strip()}
    good, seen = [], set()
    for r in rows:
        ins, ctx, res, cat = (r.get("instruction") or "").strip(), (r.get("context") or "").strip(), \
            (r.get("response") or "").strip(), r.get("category")
        if cat not in KEEP or ins in seen or ins in train_ins:
            continue
        if not (150 <= len(res) <= 1200 and 15 <= len(ins) <= 300 and len(ctx) <= 1500):
            continue
        if not TR.search(res) or re.search(r"\[\d+\]", res + ctx):
            continue
        seen.add(ins)
        good.append({"instruction": ins, "input": ctx, "output": res, "category": cat})
    rng = random.Random(args.seed)
    rng.shuffle(good)
    picked = good[: args.n]
    out = os.path.join(D, f"dolly_tr_heldout_{args.n}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    man = {"source": raw, "disjoint_from": args.train,
           "train_sha256": hashlib.sha256(open(train_path, "rb").read()).hexdigest(),
           "seed": args.seed, "candidates": len(good), "selected": len(picked),
           "by_category": dict(collections.Counter(p["category"] for p in picked)),
           "declared": "2026-09-10, secondary metric, see README 'Evaluation protocol'",
           "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
    json.dump(man, open(out.replace(".jsonl", "_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(good)} candidates disjoint from {args.train} -> {len(picked)} selected -> {out}")
    print(json.dumps(man["by_category"], ensure_ascii=False), "sha256", man["sha256"][:16])


if __name__ == "__main__":
    main()
