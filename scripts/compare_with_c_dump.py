#!/usr/bin/env python3
"""
Replay the C reference's per-layer hidden-state dump through the LazyLoRA forward pass.

kimi-k3-in-c writes the residual stream after every layer as float32 [T, hidden] when
K3_DUMP_H is set (h_layer_NNN.bin). This script embeds the same token ids, runs
forward_layer for the same layers, and reports cosine / max-diff per layer. Untrained
LoRA has B = 0, so the two engines must agree up to bf16-vs-fp32 precision.

It needs only the shards those layers live in, so it runs while the checkpoint is
incomplete, and it needs no C engine at run time, only its dump.

    python scripts/compare_with_c_dump.py --dump <dir with h_layer_*.bin> --ids 19180,11 --layers 13
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from lazy_lora.core.config import get_default_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="directory holding h_layer_NNN.bin from the C engine")
    ap.add_argument("--ids", default="19180,11", help="comma-separated token ids the dump was made with")
    ap.add_argument("--layers", type=int, default=13)
    ap.add_argument("--out", default=None, help="write the per-layer table here as well")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",")]
    cfg = get_default_config()
    hidden = cfg.model.hidden_size
    trainer = LazyLoRATrainer(cfg)
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit(f"model dir : {cfg.paths.base_model_dir}")
    emit(f"dump dir  : {args.dump}")
    emit(f"ids       : {ids}   layers: {args.layers}")

    t_all = time.time()
    with torch.no_grad():
        h = trainer._embed_tokens(torch.tensor([ids], dtype=torch.long))
        trainer._reset_block_residual()
        emit(f"embed     : std={h.float().std():.4f}")
        for l in range(args.layers):
            t0 = time.time()
            h, experts = trainer.forward_layer(l, h)
            dt = time.time() - t0
            ref_path = os.path.join(args.dump, f"h_layer_{l:03d}.bin")
            ours = h[0].float().numpy().reshape(-1)
            if not os.path.exists(ref_path):
                emit(f"after layer {l:2d}   {dt:7.1f}s  experts={len(experts):3d}  (no reference file)")
                continue
            ref = np.fromfile(ref_path, dtype=np.float32)
            if ref.size != ours.size:
                emit(f"after layer {l:2d}   reference has {ref.size // hidden} tokens, we have {ours.size // hidden}; ids differ?")
                return 2
            cos = float(ours @ ref / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-20))
            emit(f"after layer {l:2d}   {dt:7.1f}s  experts={len(experts):3d}  cosine={cos:.6f}  "
                 f"maxdiff={np.abs(ours - ref).max():8.4f}  ours std={ours.std():8.4f}  C std={ref.std():8.4f}")
    emit(f"total {time.time() - t_all:.0f}s, bytes read {trainer.mmap_streamer.bytes_read / 1e9:.2f} GB")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
