#!/usr/bin/env python3
"""
Verify the Kimi K3 checkpoint on disk without any network access.

For every shard named in model.safetensors.index.json the script checks that the file
exists, is not empty, has a parseable safetensors header, that its size equals
8 + header length + the last data offset (so it is neither truncated nor padded), and that
every tensor the index assigns to it is really in its header. Optionally (--hash) it also
computes sha256 for the shards you name, to compare with the LFS oids on the Hub.

Exit code 0 means the checkpoint is complete; 1 means at least one problem was found.
The earlier corruption scan reported a zero-byte shard as "0 damaged"; this one does not.
"""
import argparse, hashlib, json, os, struct, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import default_model_dir  # noqa: E402


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            return None, "file shorter than 8 bytes"
        n = struct.unpack("<Q", head)[0]
        if n > 100 * 1024 * 1024:
            return None, f"implausible header length {n}"
        try:
            hdr = json.loads(f.read(n))
        except Exception as exc:
            return None, f"header is not JSON: {exc}"
    return (n, hdr), None


def sha256_of(path, bufsize=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=default_model_dir())
    ap.add_argument("--hash", nargs="*", default=None,
                    help="shard file names to sha256 (slow: ~3 min per 17 GB shard on the HDD)")
    args = ap.parse_args()

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"FAIL: no index at {index_path}")
        return 1
    index = json.load(open(index_path))
    weight_map = index["weight_map"]
    expected_total = int(index.get("metadata", {}).get("total_size", 0))

    by_shard = {}
    for tensor, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(tensor)

    problems = []
    on_disk_total = 0
    t0 = time.time()
    for shard in sorted(by_shard):
        path = os.path.join(args.model_dir, shard)
        if not os.path.exists(path):
            problems.append(f"{shard}: MISSING ({len(by_shard[shard])} tensors)")
            continue
        size = os.path.getsize(path)
        on_disk_total += size
        if size == 0:
            problems.append(f"{shard}: EMPTY (0 bytes, {len(by_shard[shard])} tensors lost)")
            continue
        parsed, err = read_header(path)
        if err:
            problems.append(f"{shard}: {err}")
            continue
        n, hdr = parsed
        data_end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        want_size = 8 + n + data_end
        if size != want_size:
            problems.append(f"{shard}: size {size} != header+data {want_size} ({'truncated' if size < want_size else 'oversized'})")
        present = set(k for k in hdr if k != "__metadata__")
        missing = by_shard[shard] - present
        if missing:
            problems.append(f"{shard}: {len(missing)} tensors named in the index are not in its header, e.g. {sorted(missing)[:3]}")

    print(f"shards in index : {len(by_shard)}")
    print(f"bytes on disk   : {on_disk_total / 1e9:.2f} GB   (index says {expected_total / 1e9:.2f} GB)")
    print(f"checked in      : {time.time() - t0:.1f} s")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("\nOK: every shard is present, complete and carries the tensors the index expects.")

    if args.hash:
        print("\nsha256 (compare with the LFS oid on huggingface.co):")
        for shard in args.hash:
            path = os.path.join(args.model_dir, shard)
            if not os.path.exists(path):
                print(f"  {shard}: missing")
                continue
            t1 = time.time()
            print(f"  {shard}: {sha256_of(path)}  ({time.time() - t1:.0f} s)")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
