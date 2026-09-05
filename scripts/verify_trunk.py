#!/usr/bin/env python3
"""
Check that the packed trunk on the NVMe serves exactly the bytes the HDD shards hold.

Reads every remapped tensor of the given layers (default: 0, 1, 3, 12, 92) from both
locations and compares them byte for byte. Layer 0 alone is 2.3 GB, so this takes a
few minutes; run it once after copying trunk.bin, and again if the pack is rebuilt.

    python scripts/verify_trunk.py [--layers 0,1,3,12,92] [--sample N]
"""
import argparse, hashlib, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config, default_trunk_dir  # noqa: E402
from lazy_lora.streaming.mmap_loader import MmapTensorStreamer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,3,12,92")
    ap.add_argument("--sample", type=int, default=0, help="additionally check N random remapped tensors")
    args = ap.parse_args()
    cfg = get_default_config()
    os.environ["LAZYLORA_TRUNK_DIR"] = ""            # plain shard index
    shards = MmapTensorStreamer(cfg.paths.base_model_dir)
    del os.environ["LAZYLORA_TRUNK_DIR"]
    trunk = MmapTensorStreamer(cfg.paths.base_model_dir)
    if not trunk.index.trunk_remapped:
        print(f"FAIL: no trunk overlay active ({default_trunk_dir()})")
        return 1
    layers = {int(x) for x in args.layers.split(",")}
    names = [n for n, loc in trunk.index.tensor_locations.items()
             if loc[0] == trunk.index.trunk_path and (m := re.search(r"layers\.(\d+)\.", n)) and int(m.group(1)) in layers]
    if args.sample:
        pool = [n for n, loc in trunk.index.tensor_locations.items() if loc[0] == trunk.index.trunk_path and n not in names]
        names += random.sample(pool, min(args.sample, len(pool)))
    print(f"trunk overlay: {trunk.index.trunk_remapped} tensors remapped; checking {len(names)}")
    bad = 0
    total = 0
    t0 = time.time()
    for n in names:
        a = shards.load_tensor(n, as_torch=False)
        b = trunk.load_tensor(n, as_torch=False)
        total += a.nbytes
        if a.tobytes() != b.tobytes():
            bad += 1
            print(f"  MISMATCH {n}")
    print(f"{len(names) - bad}/{len(names)} identical, {total / 1e9:.2f} GB compared in {time.time() - t0:.0f}s")
    print("OK" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
