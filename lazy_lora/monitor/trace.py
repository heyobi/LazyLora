"""
Expert access trace: what the router chose, layer by layer, token by token.

This is the measurement instrument the research programme rests on (report §6.2, §6.3):
a few hundred MB of (layer, token, chosen experts, weights) records lets anyone study
expert concentration, cache policies and language-dependent routing without the 1.56 TB
checkpoint or this hardware. The format is deliberately trivial so it can be read with
NumPy alone.

trace.bin  : a sequence of records
             int32 layer, int32 N, int32 K,
             int16[N*K] expert ids (row-major), float16[N*K] combining weights
trace.json : manifest - token ids, text, timestamps, per-layer timings and byte counts,
             peak RSS, the engine version, and the list of layers recorded.

Both files are appended as the forward pass proceeds, so a crash keeps what was measured.
"""
import json
import os
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np

_REC = struct.Struct("<iii")


class ExpertTraceWriter:
    """Append-only writer for one forward pass."""

    def __init__(self, out_dir: str, tag: str = "", meta: Optional[Dict[str, Any]] = None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.bin_path = os.path.join(out_dir, "trace.bin")
        self.json_path = os.path.join(out_dir, "trace.json")
        self.manifest: Dict[str, Any] = {
            "tag": tag,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layers": [],
            **(meta or {}),
        }
        self._f = open(self.bin_path, "wb")
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.json_path)

    def record_layer(self, layer_idx: int, topk_indices, topk_weights, extra: Optional[Dict[str, Any]] = None) -> None:
        idx = np.asarray(topk_indices.detach().cpu().numpy() if hasattr(topk_indices, "detach") else topk_indices)
        w = np.asarray(topk_weights.detach().float().cpu().numpy() if hasattr(topk_weights, "detach") else topk_weights)
        idx = idx.reshape(-1, idx.shape[-1]).astype(np.int16)
        w = w.reshape(-1, w.shape[-1]).astype(np.float16)
        n, k = idx.shape
        self._f.write(_REC.pack(int(layer_idx), int(n), int(k)))
        self._f.write(idx.tobytes())
        self._f.write(w.tobytes())
        self._f.flush()
        entry = {"layer": int(layer_idx), "n": int(n), "k": int(k),
                 "unique_experts": int(len(np.unique(idx)))}
        if extra:
            entry.update(extra)
        self.manifest["layers"].append(entry)
        self._flush_manifest()

    def note(self, **kw) -> None:
        self.manifest.update(kw)
        self._flush_manifest()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
        self._flush_manifest()


def read_trace(trace_dir: str):
    """Returns (manifest, {layer: (indices int16 [N,K], weights float16 [N,K])})."""
    with open(os.path.join(trace_dir, "trace.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    layers = {}
    with open(os.path.join(trace_dir, "trace.bin"), "rb") as f:
        while True:
            head = f.read(_REC.size)
            if len(head) < _REC.size:
                break
            layer, n, k = _REC.unpack(head)
            idx = np.frombuffer(f.read(n * k * 2), dtype=np.int16).reshape(n, k)
            w = np.frombuffer(f.read(n * k * 2), dtype=np.float16).reshape(n, k)
            layers[layer] = (idx, w)
    return manifest, layers
