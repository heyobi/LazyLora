#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ibrahim Polat
"""
Package expert-routing trace directories into a release folder.

    python scripts/export_traces.py <source_dir> <release_dir>
    python scripts/export_traces.py ~/work/traces ./release --glob '*_L93_*'
    python scripts/export_traces.py ~/work/traces ./release --keep-text-for tr_paragraph,code_python

---------------------------------------------------------------------------------------
!! WRITTEN WITHOUT BEING EXECUTED !!
Written while the machine was busy with the 100-step training run, so it has never been
run and no trace has been exported with it. The five traces in evidence/traces/ were not
exported with this script: all five of their texts were written for this project, so
nothing had to be withheld and the directories were copied in whole, with only the local
model path replaced. Use this script when you trace a text you do not own - and then run
it on a copy first and read the withheld-text fields of every output trace.json by hand,
because it is the only thing standing between someone else's paragraph and a public
dataset.
---------------------------------------------------------------------------------------

What it does, in order, for every trace directory found under <source_dir>:

  1. validates the trace by parsing every record with the reader this repository ships
     (lazy_lora.monitor.trace.read_trace) and then re-walking the file byte by byte to
     catch what the reader silently tolerates: trailing bytes, truncated tails, duplicate
     layer records, expert ids outside [0, num_experts), non-finite weights, and
     disagreement between trace.bin and the per-layer entries in trace.json;
  2. copies trace.bin verbatim, computing its sha256 as it goes;
  3. copies trace.json with the input text and token ids removed (they reconstruct the
     prompt losslessly, and not every traced text is redistributable), replacing them with
     a sha256 and a byte count -- see --keep-text / --keep-text-for;
  4. copies the derived analysis files if they exist;
  5. writes MANIFEST.json (per-file sha256, sizes, record counts, per-layer summary),
     CHECKSUMS.sha256, a sources.json stub and a README.md stub if either is missing;
  6. prints a summary table and exits non-zero if any trace failed validation.

It never writes to <source_dir>: every source file is opened read-only, and the tool
refuses to run if the release directory lies inside the source directory.

Trace format: lazy_lora/monitor/trace.py. Dataset card: docs/traces/README.md.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from lazy_lora.monitor.trace import read_trace  # noqa: E402

EXPORTER_VERSION = "1.0"
HEADER = struct.Struct("<iii")
HEADER_SIZE = HEADER.size          # 12 bytes: int32 layer, int32 N, int32 K
BYTES_PER_VALUE = 2                # int16 expert id, float16 weight
DERIVED_FILES = ("analysis.json", "analysis.md")
TEXT_KEYS = ("text", "ids")
CHUNK = 1 << 20

README_STUB = """# Kimi K3 expert-routing traces

Placeholder. The full dataset card lives in the LazyLoRA repository at
`docs/traces/README.md`; copy it here before publishing this release.

It must describe, at minimum:

- what a trace record is and why it is useful (evaluating cache and scheduling policies at
  frontier scale without the 1.56 TB checkpoint);
- the binary format: `int32 layer, int32 N, int32 K`, then `int16[N*K]` expert ids and
  `float16[N*K]` combining weights, little-endian, no header and no footer;
- the JSON manifest fields, including `layers`, `token_norms` and the withheld-text keys;
- the traced texts with language, token count, provenance and licence (`sources.json`);
- how the traces were produced: `scripts/measure_routing.py --layers 93`, untrained
  adapter, and the machine;
- file sizes and sha256 (see `MANIFEST.json` and `CHECKSUMS.sha256`);
- a worked example that recomputes one published number;
- licence and attribution: routing arrays CC0-1.0, manifests CC BY 4.0, each traced text
  under its own licence; not affiliated with Moonshot AI;
- known limitations and the citation block.
"""

SOURCES_STUB_NOTE = (
    "Provenance of each traced text. Fill in source, licence and retrieval date before "
    "publishing. Where a text is not redistributable, leave 'text_released': false and "
    "keep only the hash the exporter recorded in trace.json."
)


# --------------------------------------------------------------------------- helpers

def sha256_file(path):
    """sha256 of a file, read-only, in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def copy_file_hashed(src, dst):
    """Copy src -> dst without touching src; returns (sha256, bytes)."""
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for block in iter(lambda: fi.read(CHUNK), b""):
            h.update(block)
            fo.write(block)
            n += len(block)
    return h.hexdigest(), n


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=False)
        f.write("\n")


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GiB"


def _children_with_traces(parent, patterns):
    out = []
    for name in sorted(os.listdir(parent)):
        d = os.path.join(parent, name)
        if not os.path.isdir(d):
            continue
        if not (os.path.isfile(os.path.join(d, "trace.bin"))
                and os.path.isfile(os.path.join(d, "trace.json"))):
            continue
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        out.append(d)
    return out


def find_traces(src, patterns):
    """
    Directories directly under src that contain both trace.bin and trace.json.

    Falls back to src/traces/ if src itself holds none, because that is where
    measure_routing.py writes them (<workspace>/traces/<tag>_<timestamp>/). Returns
    (directories, searched_root).
    """
    found = _children_with_traces(src, patterns)
    if found:
        return found, src
    nested = os.path.join(src, "traces")
    if os.path.isdir(nested):
        return _children_with_traces(nested, patterns), nested
    return [], src


# --------------------------------------------------------------------------- validation

def walk_records(bin_path, num_experts):
    """
    Byte-exact walk of trace.bin. Returns (records, errors).

    records: list of dicts with layer, n, k, offset, nbytes, unique, min_id, max_id
    errors:  list of strings; empty means the file is exactly a sequence of well-formed
             records with nothing left over.
    """
    records, errors = [], []
    size = os.path.getsize(bin_path)
    seen_layers = set()
    with open(bin_path, "rb") as f:
        offset = 0
        while True:
            head = f.read(HEADER_SIZE)
            if not head:
                break
            if len(head) < HEADER_SIZE:
                errors.append(f"truncated record header at byte {offset} "
                              f"({len(head)} of {HEADER_SIZE} bytes)")
                break
            layer, n, k = HEADER.unpack(head)
            if n <= 0 or k <= 0:
                errors.append(f"record at byte {offset}: implausible shape N={n} K={k}")
                break
            payload = n * k * BYTES_PER_VALUE
            need = 2 * payload
            if offset + HEADER_SIZE + need > size:
                errors.append(f"record for layer {layer} at byte {offset} needs "
                              f"{need} payload bytes, file has "
                              f"{size - offset - HEADER_SIZE}")
                break
            ids = np.frombuffer(f.read(payload), dtype="<i2").reshape(n, k)
            w = np.frombuffer(f.read(payload), dtype="<f2").reshape(n, k)
            if layer in seen_layers:
                errors.append(f"duplicate record for layer {layer} at byte {offset}")
            seen_layers.add(layer)
            lo, hi = int(ids.min()), int(ids.max())
            if lo < 0 or hi >= num_experts:
                errors.append(f"layer {layer}: expert id out of range "
                              f"[{lo}, {hi}] not within [0, {num_experts})")
            if not np.isfinite(w.astype(np.float32)).all():
                errors.append(f"layer {layer}: non-finite combining weight")
            records.append({
                "layer": layer, "n": n, "k": k, "offset": offset,
                "nbytes": HEADER_SIZE + need,
                "unique_experts": int(np.unique(ids).size),
                "min_expert_id": lo, "max_expert_id": hi,
            })
            offset += HEADER_SIZE + need
        if offset != size and not errors:
            errors.append(f"{size - offset} trailing bytes after the last record")
    return records, errors


def validate(trace_dir, num_experts):
    """
    Returns (manifest, records, errors, warnings).

    The trace is first parsed with the repository's own reader, so that anything this
    release contains is guaranteed to load with read_trace(); the byte walk then checks
    what that reader tolerates in silence.
    """
    errors, warnings = [], []

    try:
        manifest, layers = read_trace(trace_dir)
    except Exception as exc:                       # noqa: BLE001 - report, do not raise
        return None, [], [f"read_trace() failed: {exc.__class__.__name__}: {exc}"], []

    records, walk_errors = walk_records(os.path.join(trace_dir, "trace.bin"), num_experts)
    errors.extend(walk_errors)

    if len(layers) != len(records):
        errors.append(f"read_trace() returned {len(layers)} layers, the byte walk found "
                      f"{len(records)} records")

    if not records:
        errors.append("no records in trace.bin")
        return manifest, records, errors, warnings

    ks = {r["k"] for r in records}
    ns = {r["n"] for r in records}
    if len(ks) != 1:
        errors.append(f"inconsistent K across records: {sorted(ks)}")
    if len(ns) != 1:
        errors.append(f"inconsistent N across records: {sorted(ns)}")

    n_tokens = manifest.get("n_tokens")
    if n_tokens is not None and len(ns) == 1 and int(n_tokens) != next(iter(ns)):
        errors.append(f"manifest n_tokens={n_tokens} but records carry N={next(iter(ns))}")

    ids = manifest.get("ids")
    if isinstance(ids, list) and n_tokens is not None and len(ids) != int(n_tokens):
        errors.append(f"manifest has {len(ids)} token ids but n_tokens={n_tokens}")

    # per-layer entries in the manifest must agree with the file
    entries = {int(e["layer"]): e for e in manifest.get("layers", []) if "layer" in e}
    by_layer = {r["layer"]: r for r in records}
    missing = sorted(set(by_layer) - set(entries))
    extra = sorted(set(entries) - set(by_layer))
    if missing:
        warnings.append(f"{len(missing)} layers in trace.bin are absent from "
                        f"trace.json['layers'] (first: {missing[:3]})")
    if extra:
        errors.append(f"trace.json lists layers with no record in trace.bin: {extra[:5]}")
    for layer, rec in sorted(by_layer.items()):
        e = entries.get(layer)
        if not e:
            continue
        for key in ("n", "k", "unique_experts"):
            if key in e and int(e[key]) != rec[key]:
                errors.append(f"layer {layer}: trace.json {key}={e[key]} but trace.bin "
                              f"gives {rec[key]}")

    expected = sum(HEADER_SIZE + 4 * r["n"] * r["k"] for r in records)
    actual = os.path.getsize(os.path.join(trace_dir, "trace.bin"))
    if expected != actual:
        errors.append(f"trace.bin is {actual} bytes, records account for {expected}")

    if manifest.get("error"):
        warnings.append(f"the run recorded an error: {manifest['error']}")
    if manifest.get("total_seconds") is None:
        warnings.append("no total_seconds in the manifest: the run may not have finished")

    return manifest, records, errors, warnings


# --------------------------------------------------------------------------- export

def sanitise_manifest(manifest, keep_text):
    """
    Copy of the manifest with the prompt removed unless keep_text.

    trace.json stores the input text and its token ids; the ids reconstruct the text
    losslessly, so a trace of a copyrighted paragraph contains that paragraph. Replace
    both with a hash and a length, which is enough to verify that a measurement used the
    text it claims to have used.
    """
    out = dict(manifest)
    text = manifest.get("text")
    ids = manifest.get("ids")
    if isinstance(text, str):
        out["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out["text_bytes"] = len(text.encode("utf-8"))
        out["text_chars"] = len(text)
    if isinstance(ids, list):
        out["n_ids"] = len(ids)
        blob = ",".join(str(int(i)) for i in ids).encode("utf-8")
        out["ids_sha256"] = hashlib.sha256(blob).hexdigest()
    if keep_text:
        out["text_released"] = True
    else:
        for key in TEXT_KEYS:
            out.pop(key, None)
        out["text_released"] = False
        out["text_withheld_note"] = (
            "The input text and its token ids were removed for redistribution; the token "
            "ids reconstruct the text exactly. See sources.json for the source and "
            "licence of this text, and text_sha256 / ids_sha256 above to verify it."
        )
    return out


def export_one(trace_dir, dest_dir, manifest, records, keep_text, force):
    """Copy one validated trace into dest_dir. Returns the manifest entry."""
    name = os.path.basename(trace_dir.rstrip(os.sep))
    out_dir = os.path.join(dest_dir, name)

    # Decide everything that will be written before writing anything, so a collision
    # cannot leave a half-exported trace behind.
    planned = ["trace.bin", "trace.json"]
    planned += [e for e in DERIVED_FILES if os.path.isfile(os.path.join(trace_dir, e))]
    if not force:
        clash = [p for p in planned if os.path.exists(os.path.join(out_dir, p))]
        if clash:
            raise FileExistsError(f"{out_dir} already has {', '.join(clash)}; "
                                  f"pass --force to overwrite")
    os.makedirs(out_dir, exist_ok=True)

    files = {}

    src_bin = os.path.join(trace_dir, "trace.bin")
    dst_bin = os.path.join(out_dir, "trace.bin")
    digest, nbytes = copy_file_hashed(src_bin, dst_bin)
    files["trace.bin"] = {"sha256": digest, "bytes": nbytes}

    clean = sanitise_manifest(manifest, keep_text)
    clean["exported_by"] = f"scripts/export_traces.py {EXPORTER_VERSION}"
    clean["source_trace_json_sha256"] = sha256_file(os.path.join(trace_dir, "trace.json"))
    dst_json = os.path.join(out_dir, "trace.json")
    write_json(dst_json, clean)
    files["trace.json"] = {"sha256": sha256_file(dst_json),
                           "bytes": os.path.getsize(dst_json)}

    for extra in DERIVED_FILES:
        src = os.path.join(trace_dir, extra)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out_dir, extra)
        digest, nbytes = copy_file_hashed(src, dst)
        files[extra] = {"sha256": digest, "bytes": nbytes}

    layer_ids = sorted(r["layer"] for r in records)
    uniq = [r["unique_experts"] for r in records]
    return {
        "dir": name,
        "tag": manifest.get("tag"),
        "created": manifest.get("created"),
        "n_tokens": manifest.get("n_tokens"),
        "records": len(records),
        "k": records[0]["k"],
        "layers": {"count": len(layer_ids),
                   "first": layer_ids[0],
                   "last": layer_ids[-1],
                   "contiguous": layer_ids == list(range(layer_ids[0], layer_ids[-1] + 1))},
        "unique_experts": {"min": int(min(uniq)),
                           "max": int(max(uniq)),
                           "mean": round(float(sum(uniq)) / len(uniq), 1)},
        "max_expert_id": max(r["max_expert_id"] for r in records),
        "text_released": bool(keep_text),
        "total_seconds": manifest.get("total_seconds"),
        "bytes_read": manifest.get("bytes_read"),
        "peak_rss_gb": manifest.get("peak_rss_gb"),
        "compute_dtype": manifest.get("compute_dtype"),
        "files": files,
    }


def write_stubs(dest_dir, entries):
    """
    README.md and sources.json stubs, written only when absent.

    Never overwritten, not even with --force: by the time a release has a real dataset
    card and a filled-in sources.json, clobbering them with placeholders would be the
    most expensive thing this tool could do.
    """
    written = []
    readme = os.path.join(dest_dir, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(README_STUB)
        written.append("README.md")
    sources = os.path.join(dest_dir, "sources.json")
    if not os.path.exists(sources):
        stub = {"_note": SOURCES_STUB_NOTE, "texts": {}}
        for e in entries:
            stub["texts"][e["tag"] or e["dir"]] = {
                "trace_dir": e["dir"],
                "n_tokens": e["n_tokens"],
                "source": "FILL: url, dataset id, or 'written by the author'",
                "licence": "FILL",
                "retrieved": "FILL: YYYY-MM-DD",
                "text_released": e["text_released"],
            }
        write_json(sources, stub)
        written.append("sources.json")
    return written


def write_checksums(dest_dir, entries):
    """sha256sum -c compatible, paths relative to the release root."""
    lines = []
    for e in entries:
        for fname, info in e["files"].items():
            lines.append(f"{info['sha256']}  {e['dir']}/{fname}")
    path = os.path.join(dest_dir, "CHECKSUMS.sha256")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- reporting

def print_table(rows, title):
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in ([title] + rows)) for i in range(len(title))]
    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
    print(line(title))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Package expert-routing traces into a release folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The source directory is never modified.")
    ap.add_argument("source", help="directory containing trace directories "
                                   "(each with trace.bin and trace.json)")
    ap.add_argument("dest", help="release directory to create or add to")
    ap.add_argument("--glob", action="append", default=None, metavar="PATTERN",
                    help="only export trace directories whose name matches (repeatable; "
                         "default: every trace directory found)")
    ap.add_argument("--num-experts", type=int, default=896,
                    help="routed experts per layer, for the id range check (default 896)")
    ap.add_argument("--keep-text", action="store_true",
                    help="keep the input text and token ids in the exported trace.json. "
                         "Off by default: token ids reconstruct the prompt exactly, and "
                         "not every traced text is redistributable")
    ap.add_argument("--keep-text-for", default="", metavar="TAGS",
                    help="comma-separated tags or directory names whose text may be kept")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that already exist in the release directory")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="validate and report, write nothing")
    args = ap.parse_args(argv)

    src = os.path.abspath(os.path.expanduser(args.source))
    dst = os.path.abspath(os.path.expanduser(args.dest))

    if not os.path.isdir(src):
        print(f"error: source directory not found: {src}", file=sys.stderr)
        return 2
    if src == dst:
        print("error: source and destination are the same directory", file=sys.stderr)
        return 2
    if dst == os.path.commonpath([src, dst]):
        print(f"error: the source directory {src} lies inside the release directory "
              f"{dst}; refusing to risk writing into it", file=sys.stderr)
        return 2
    if src == os.path.commonpath([src, dst]):
        print(f"error: the release directory {dst} lies inside the source directory "
              f"{src}; choose a destination outside the source", file=sys.stderr)
        return 2

    patterns = args.glob or []
    trace_dirs, searched = find_traces(src, patterns)
    if not trace_dirs:
        print(f"error: no trace directories (trace.bin + trace.json) under {searched}"
              + (f" matching {patterns}" if patterns else ""), file=sys.stderr)
        return 2
    if searched != src:
        print(f"note: reading traces from {searched}")

    keep_for = {t.strip() for t in args.keep_text_for.split(",") if t.strip()}

    if not args.dry_run:
        os.makedirs(dst, exist_ok=True)

    entries, rows, failed = [], [], []
    for trace_dir in trace_dirs:
        name = os.path.basename(trace_dir)
        manifest, records, errors, warnings = validate(trace_dir, args.num_experts)
        tag = (manifest or {}).get("tag") or name
        keep = args.keep_text or tag in keep_for or name in keep_for

        for w in warnings:
            print(f"warning  {name}: {w}", file=sys.stderr)
        if errors:
            failed.append(name)
            for e in errors:
                print(f"INVALID  {name}: {e}", file=sys.stderr)
            rows.append([name, str((manifest or {}).get("n_tokens", "?")),
                         str(len(records)), "-", "-", "-", "FAILED"])
            continue

        nbytes = os.path.getsize(os.path.join(trace_dir, "trace.bin"))
        if args.dry_run:
            entry = {"dir": name, "tag": tag, "n_tokens": manifest.get("n_tokens"),
                     "records": len(records), "text_released": keep, "files": {}}
        else:
            try:
                entry = export_one(trace_dir, dst, manifest, records, keep, args.force)
            except FileExistsError as exc:
                failed.append(name)
                print(f"INVALID  {name}: {exc}", file=sys.stderr)
                continue
        entries.append(entry)

        uniq = [r["unique_experts"] for r in records]
        rows.append([
            name,
            str(manifest.get("n_tokens", "?")),
            str(len(records)),
            f"{min(r['layer'] for r in records)}-{max(r['layer'] for r in records)}",
            f"{sum(uniq) / len(uniq):.0f}",
            human(nbytes),
            "text kept" if keep else "text withheld",
        ])

    print()
    print_table(rows, ["trace", "tokens", "records", "layers", "uniq/layer",
                       "trace.bin", "manifest"])
    print()

    if args.dry_run:
        print(f"dry run: {len(entries)} trace(s) valid, {len(failed)} failed; "
              f"nothing written")
        return 1 if failed else 0

    if entries:
        release = {
            "dataset": "Kimi K3 expert-routing traces",
            "exporter": f"scripts/export_traces.py {EXPORTER_VERSION}",
            "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_directory_name": os.path.basename(src),
            "num_experts": args.num_experts,
            "record_format": ("little-endian: int32 layer, int32 N, int32 K, "
                              "int16[N*K] expert ids, float16[N*K] combining weights; "
                              "see lazy_lora/monitor/trace.py"),
            "licence": {"trace.bin": "CC0-1.0",
                        "manifests_and_derived": "CC BY 4.0",
                        "traced_texts": "see sources.json"},
            "traces": entries,
            "failed": failed,
        }
        write_json(os.path.join(dst, "MANIFEST.json"), release)
        write_checksums(dst, entries)
        stubs = write_stubs(dst, entries)
        total = sum(i["bytes"] for e in entries for i in e["files"].values())
        print(f"wrote {len(entries)} trace(s), {human(total)} to {dst}")
        print("      MANIFEST.json, CHECKSUMS.sha256"
              + (", " + ", ".join(stubs) if stubs else ""))
        if stubs:
            print("      the stubs above are placeholders: fill sources.json and replace "
                  "README.md with docs/traces/README.md before publishing")
        print("      the dataset card also expects the 93-row cosine table and the "
              "comparison log in the release root; this tool does not copy them")
    if failed:
        print(f"\n{len(failed)} trace(s) failed validation and were not exported: "
              f"{', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
