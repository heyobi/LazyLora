#!/usr/bin/env python3
"""
Routing statistics from expert traces (report §3.5 and §6.2).

    python scripts/analyze_trace.py <trace_dir>                 # one run
    python scripts/analyze_trace.py <trace_dir_A> <trace_dir_B> # plus a per-layer comparison
                                                                # (e.g. Turkish vs English)

Per layer:
  unique      unique experts touched by the whole batch, and the uniform-routing expectation
              896 * (1 - (1 - 16/896)^N)
  top100      share of all (token, expert) activations captured by the 100 most used experts
  H_use       entropy (bits) of the expert usage distribution; 896 experts uniform = 9.81 bits
  eff_k       mean effective experts per token, exp(entropy of the 16 combining weights)
  jac_t       mean Jaccard overlap of consecutive tokens' expert sets (temporal locality)
  jac_L       mean Jaccard overlap of a token's expert sets at this layer and the next
              (cross-layer predictability; the prefetch signal that needs no model change)
Plus the unique-experts-vs-N curve over token prefixes, which one run yields directly
because routing is causal.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.monitor.trace import read_trace  # noqa: E402

E = 896
K = 16
PREFIXES = [1, 8, 16, 24, 32, 64, 96, 127, 128, 192, 256, 384, 512]


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def layer_stats(idx, w, idx_next=None):
    n, k = idx.shape
    counts = np.bincount(idx.reshape(-1).astype(np.int64), minlength=E)
    uniq = int((counts > 0).sum())
    expected = E * (1 - (1 - K / E) ** n)
    top100 = float(np.sort(counts)[::-1][:100].sum() / counts.sum())
    h_use = entropy_bits(counts / counts.sum())
    wf = w.astype(np.float64)
    wf = wf / np.clip(wf.sum(-1, keepdims=True), 1e-9, None)
    eff_k = float(np.mean([2 ** entropy_bits(row) for row in wf]))
    sets = [set(r.tolist()) for r in idx]
    jac_t = float(np.mean([len(sets[i] & sets[i + 1]) / len(sets[i] | sets[i + 1]) for i in range(n - 1)])) if n > 1 else float("nan")
    jac_l = float("nan")
    if idx_next is not None and idx_next.shape[0] == n:
        nxt = [set(r.tolist()) for r in idx_next]
        jac_l = float(np.mean([len(sets[i] & nxt[i]) / len(sets[i] | nxt[i]) for i in range(n)]))
    return {"n": n, "unique": uniq, "expected_uniform": round(expected, 1), "top100_share": round(top100, 3),
            "H_use_bits": round(h_use, 2), "eff_k": round(eff_k, 2), "jac_t": round(jac_t, 3), "jac_L": round(jac_l, 3)}


def prefix_curve(idx):
    n = idx.shape[0]
    out = {}
    for p in PREFIXES:
        if p <= n:
            out[p] = int(len(np.unique(idx[:p])))
    if n not in out:
        out[n] = int(len(np.unique(idx)))
    return out


def analyze(trace_dir):
    manifest, layers = read_trace(trace_dir)
    keys = sorted(layers)
    rows = {}
    for i, L in enumerate(keys):
        idx, w = layers[L]
        nxt = layers[keys[i + 1]][0] if i + 1 < len(keys) and keys[i + 1] == L + 1 else None
        rows[L] = layer_stats(idx, w, nxt)
    curves = {L: prefix_curve(layers[L][0]) for L in keys}
    return manifest, rows, curves, layers


def print_report(name, manifest, rows, curves):
    n = manifest.get("n_tokens")
    print(f"\n## {name}: {n} tokens, {len(rows)} MoE layers, tag={manifest.get('tag')}")
    if manifest.get("total_seconds"):
        print(f"total {manifest['total_seconds']}s, read {manifest.get('bytes_read', 0) / 1e9:.1f} GB, peak RSS {manifest.get('peak_rss_gb')} GB")
    print(f"\n| layer | unique | uniform exp. | top100 share | H_use bits | eff_k | jac_t | jac_L | s | GB |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    timing = {e["layer"]: e for e in manifest.get("layers", [])}
    for L, r in rows.items():
        t = timing.get(L, {})
        print(f"| {L} | {r['unique']} | {r['expected_uniform']} | {r['top100_share']} | {r['H_use_bits']} | "
              f"{r['eff_k']} | {r['jac_t']} | {r['jac_L']} | {t.get('seconds', '')} | "
              f"{round(t.get('bytes_read', 0) / 1e9, 2) if t else ''} |")
    us = np.array([r["unique"] for r in rows.values()])
    ex = np.array([r["expected_uniform"] for r in rows.values()])
    print(f"\nmean unique/expected ratio over layers: {float(np.mean(us / ex)):.3f}   "
          f"(1.0 = uniform routing; lower = concentrated)")
    # unique-vs-N curve, averaged over layers
    Ns = sorted({p for c in curves.values() for p in c})
    print("\nunique experts vs prefix length N (mean over layers, and uniform expectation):")
    print("| N | measured | uniform |")
    print("|---:|---:|---:|")
    for p in Ns:
        vals = [c[p] for c in curves.values() if p in c]
        print(f"| {p} | {np.mean(vals):.0f} | {E * (1 - (1 - K / E) ** p):.0f} |")


def compare(name_a, la, name_b, lb):
    print(f"\n## comparison {name_a} vs {name_b} (per layer)")
    print("| layer | jaccard(unique sets) | top100 overlap | H_use A | H_use B |")
    print("|---:|---:|---:|---:|---:|")
    for L in sorted(set(la) & set(lb)):
        ia, wa = la[L]
        ib, wb = lb[L]
        sa, sb = set(np.unique(ia).tolist()), set(np.unique(ib).tolist())
        ca = np.bincount(ia.reshape(-1).astype(np.int64), minlength=E)
        cb = np.bincount(ib.reshape(-1).astype(np.int64), minlength=E)
        ta = set(np.argsort(ca)[::-1][:100].tolist())
        tb = set(np.argsort(cb)[::-1][:100].tolist())
        print(f"| {L} | {len(sa & sb) / len(sa | sb):.3f} | {len(ta & tb) / 100:.2f} | "
              f"{entropy_bits(ca / ca.sum()):.2f} | {entropy_bits(cb / cb.sum()):.2f} |")


def analyze_prefix(trace_dir, prefix):
    """Like analyze() but on the first `prefix` tokens only (fair cross-trace comparison)."""
    manifest, layers = read_trace(trace_dir)
    layers = {L: (i[:prefix], w[:prefix]) for L, (i, w) in layers.items()}
    keys = sorted(layers)
    rows = {}
    for i, L in enumerate(keys):
        idx, w = layers[L]
        nxt = layers[keys[i + 1]][0] if i + 1 < len(keys) and keys[i + 1] == L + 1 else None
        rows[L] = layer_stats(idx, w, nxt)
    manifest = dict(manifest, n_tokens=min(prefix, manifest.get("n_tokens", prefix)))
    return manifest, rows, {L: prefix_curve(layers[L][0]) for L in keys}, layers


def main():
    args = sys.argv[1:]
    prefix = None
    if "--prefix" in args:
        i = args.index("--prefix")
        prefix = int(args[i + 1])
        del args[i:i + 2]
    dirs = args
    if not dirs:
        print(__doc__)
        return 1
    results = []
    for d in dirs:
        manifest, rows, curves, layers = analyze(d) if prefix is None else analyze_prefix(d, prefix)
        print_report(os.path.basename(d.rstrip("/")), manifest, rows, curves)
        results.append((os.path.basename(d.rstrip("/")), layers))
        if prefix is None:
            with open(os.path.join(d, "analysis.json"), "w") as f:
                json.dump({"rows": rows, "prefix_curves": curves}, f, indent=1)
    if len(results) == 2:
        compare(results[0][0], results[0][1], results[1][0], results[1][1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
