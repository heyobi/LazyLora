#!/usr/bin/env python3
"""
Select the LIMA-style Turkish instruction set from the Turkish translation of Dolly-15k
(atasoglu/databricks-dolly-15k-tr, CC BY-SA 3.0; originals human-written by Databricks).

    python scripts/build_train_set.py [--n 400] [--seed 0]

Writes <workspace>/datasets/dolly_tr_<n>.jsonl (instruction / input / output records, the
format the trainer's dataset iterator expects), dolly_tr_proof.jsonl (the first 5, for
the proof-of-learning run) and a manifest with the selection rules and category counts.

Selection rules (quality over quantity):
  - categories: open_qa, general_qa, brainstorming, creative_writing, summarization,
    closed_qa (classification and information_extraction are mostly one-word answers)
  - response 150-1200 characters, instruction 15-300, context at most 1500
  - Turkish letters present in the response, no citation brackets like [3]
  - deduplicated on instruction; stratified across categories, then shuffled with --seed
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402

KEEP = ["open_qa", "general_qa", "brainstorming", "creative_writing", "summarization", "closed_qa"]
TR = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def load(path):
    raw = open(path, encoding="utf-8").read()
    dec = json.JSONDecoder(); rows = []; i = 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \r\n\t":
            i += 1
        if i >= len(raw):
            break
        obj, i = dec.raw_decode(raw, i)
        rows.append(obj)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = get_default_config()
    D = cfg.paths.dataset_dir
    rows = load(os.path.join(D, "raw", "databricks-dolly-15k-tr.jsonl"))

    good, seen = [], set()
    for r in rows:
        ins, ctx, res, cat = (r.get("instruction") or "").strip(), (r.get("context") or "").strip(), \
            (r.get("response") or "").strip(), r.get("category")
        if cat not in KEEP or ins in seen:
            continue
        if not (150 <= len(res) <= 1200 and 15 <= len(ins) <= 300 and len(ctx) <= 1500):
            continue
        if not TR.search(res) or re.search(r"\[\d+\]", res + ctx):
            continue
        seen.add(ins)
        good.append({"instruction": ins, "input": ctx, "output": res, "category": cat})
    by_cat = collections.defaultdict(list)
    for g in good:
        by_cat[g["category"]].append(g)
    rng = random.Random(args.seed)
    for v in by_cat.values():
        rng.shuffle(v)
    # stratified: proportional to availability, at least 20 per category where possible
    total = sum(len(v) for v in by_cat.values())
    quota = {c: max(20, round(args.n * len(v) / total)) for c, v in by_cat.items()}
    picked = []
    for c, q in quota.items():
        picked.extend(by_cat[c][:q])
    rng.shuffle(picked)
    picked = picked[: args.n]
    out = os.path.join(D, f"dolly_tr_{args.n}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(D, "dolly_tr_proof.jsonl"), "w", encoding="utf-8") as f:
        for p in picked[:5]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump({"source": "atasoglu/databricks-dolly-15k-tr", "licence": "CC BY-SA 3.0", "seed": args.seed,
               "candidates_after_filter": len(good), "selected": len(picked),
               "by_category": dict(collections.Counter(p["category"] for p in picked)),
               "rules": __doc__.split("Selection rules")[1].strip()},
              open(os.path.join(D, f"dolly_tr_{args.n}_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(good)} candidates after filtering -> {len(picked)} selected -> {out}")
    print("by category:", dict(collections.Counter(p["category"] for p in picked)))
    lens = sorted(len(p["output"]) for p in picked)
    print(f"response chars: median {lens[len(lens)//2]}, max {lens[-1]}")
    for p in picked[:3]:
        print(" -", p["category"], "|", p["instruction"][:70], "->", p["output"][:80].replace("\n", " "))


if __name__ == "__main__":
    main()
