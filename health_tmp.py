"""Is expert 629's scale block corrupt on disk, and is the damage localised?"""
import os

import numpy as np

from lazy_lora.streaming.mmap_loader import MmapTensorStreamer

s = MmapTensorStreamer("/mnt/d/hamza/kimi_k3_model_weights")
idx = s.index
LAYER = 23


def scale_of(expert, w="w1"):
    name = f"model.layers.{LAYER}.block_sparse_moe.experts.{expert}.{w}.weight_scale"
    r = idx._resolve_name(name)
    return r, np.asarray(s.load_tensor(name, as_torch=False))


print("neighbouring experts, w1 scale statistics:")
for e in range(625, 635):
    r, v = scale_of(e)
    flat = v.reshape(-1)
    print(f"  expert {e:3d}: min={flat.min():3d} max={flat.max():3d} "
          f"mean={flat.mean():6.1f} std={flat.std():5.1f}"
          f"{'   <-- suspect' if flat.std() > 40 else ''}")

r, v = scale_of(629)
flat = v.reshape(-1)
print(f"\nexpert 629 w1 scale, shape {v.shape}")
print(f"  first 32 bytes: {flat[:32].tolist()}")
print(f"  rows whose std is small (healthy): {(v.std(axis=1) < 5).sum()} of {v.shape[0]}")
print(f"  rows whose std is large (garbage): {(v.std(axis=1) >= 5).sum()} of {v.shape[0]}")

# Read the same range twice: a flaky read differs, a corrupt file repeats
shard, start, end, shape, dtype = idx.tensor_locations[r]
a = s._read_range(shard, start, end)
b = s._read_range(shard, start, end)
print(f"\ntwo reads of the same range identical: {a == b}")
print(f"  offset {start}, length {end - start}, sector {start // 4096}")
