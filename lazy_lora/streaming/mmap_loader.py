"""
Memory-Mapped (mmap) Safetensors Shard Reader and Zero-Copy Tensor Streamer.
Enables streaming individual layer/expert weight matrices from 16 GB shards
without loading whole files into RAM or VRAM.
"""

import os
import mmap
import json
import struct
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


DTYPE_MAP_NUMPY = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # NumPy does not natively have bfloat16 in older versions; stored as uint16 raw bits
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafetensorsIndex:
    """Parses and caches the tensor index across all safetensors shards in a directory."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tensor_locations: Dict[str, Tuple[str, int, int, List[int], str]] = {}
        self._index_built = False
        if os.path.isdir(model_dir):
            self.build_index()

    def _cache_file(self) -> str:
        """Index cache path, keyed by model directory.

        A single shared cache file would hand the real Kimi K3 index to every streamer
        (mock tests included) no matter which directory it was pointed at.
        """
        import hashlib
        key = hashlib.sha1(os.path.abspath(self.model_dir).encode("utf-8")).hexdigest()[:16]
        return f"/mnt/d/hamza/LazyLora_Workspace/cache/safetensors_index_{key}.json"

    def build_index(self) -> None:
        """Scan directory and parse JSON headers of all .safetensors files."""
        cache_file = self._cache_file()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self.tensor_locations = {k: tuple(v) for k, v in cached_data.items()}
                self._index_built = True
                return
            except Exception:
                pass

        shard_files = sorted([
            f for f in os.listdir(self.model_dir)
            if f.endswith(".safetensors")
        ])

        for shard in shard_files:
            shard_path = os.path.join(self.model_dir, shard)
            try:
                with open(shard_path, "rb") as f:
                    # Read first 8 bytes (little-endian uint64 header size)
                    header_len_bytes = f.read(8)
                    if len(header_len_bytes) < 8:
                        continue
                    header_len = struct.unpack("<Q", header_len_bytes)[0]
                    if header_len > 100 * 1024 * 1024:  # Sanity check < 100MB
                        continue
                    header_json_bytes = f.read(header_len)
                    header = json.loads(header_json_bytes.decode("utf-8"))

                    data_offset_base = 8 + header_len
                    for tensor_name, info in header.items():
                        if tensor_name == "__metadata__":
                            continue
                        offsets = info.get("data_offsets", [0, 0])
                        shape = info.get("shape", [])
                        dtype_str = info.get("dtype", "F32")
                        start = data_offset_base + offsets[0]
                        end = data_offset_base + offsets[1]
                        self.tensor_locations[tensor_name] = (
                            shard_path,
                            start,
                            end,
                            shape,
                            dtype_str,
                        )
            except Exception:
                continue

        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.tensor_locations, f)
        except Exception:
            pass

        self._index_built = True

    def _resolve_name(self, tensor_name: str) -> Optional[str]:
        if tensor_name in self.tensor_locations:
            return tensor_name
        for prefix in ["language_model.", "model.", "language_model.model."]:
            cand = prefix + tensor_name
            if cand in self.tensor_locations:
                return cand
            if tensor_name.startswith("model."):
                cand2 = "language_model." + tensor_name
                if cand2 in self.tensor_locations:
                    return cand2
        return None

    def has_tensor(self, tensor_name: str) -> bool:
        return self._resolve_name(tensor_name) is not None

    def list_tensors_for_layer(self, layer_idx: int) -> List[str]:
        """Find all tensor names belonging to layer_idx."""
        prefix1 = f"model.layers.{layer_idx}."
        prefix2 = f"language_model.model.layers.{layer_idx}."
        return [t for t in self.tensor_locations if t.startswith(prefix1) or t.startswith(prefix2)]

    def list_expert_tensors(self, layer_idx: int, expert_idx: int) -> List[str]:
        """Find tensors for specific expert in layer_idx."""
        prefix1 = f"model.layers.{layer_idx}.moe.experts.{expert_idx}."
        prefix2 = f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}."
        return [t for t in self.tensor_locations if t.startswith(prefix1) or t.startswith(prefix2)]


class MmapTensorStreamer:
    """
    High-performance zero-copy tensor loader using memory mapping.
    Maintains open mmap descriptors and extracts slice views instantly.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.index = SafetensorsIndex(model_dir)
        self._mmap_handles: Dict[str, Tuple[mmap.mmap, int]] = {}  # shard_path -> (mmap_obj, fd)

    def _get_mmap(self, shard_path: str) -> mmap.mmap:
        """Get or create mmap handle for shard."""
        if shard_path not in self._mmap_handles:
            fd = os.open(shard_path, os.O_RDONLY)
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._mmap_handles[shard_path] = (mm, fd)
        return self._mmap_handles[shard_path][0]

    def load_tensor(
        self,
        tensor_name: str,
        target_device: str = "cpu",
        as_torch: bool = True,
    ) -> Optional[Union["torch.Tensor", np.ndarray]]:
        """
        Extracts a single tensor directly from disk via mmap.
        """
        resolved_name = self.index._resolve_name(tensor_name)
        if resolved_name is None:
            return None

        shard_path, start, end, shape, dtype_str = self.index.tensor_locations[resolved_name]
        mm = self._get_mmap(shard_path)

        # Slice raw bytes view without copying entire file
        raw_bytes = mm[start:end]

        np_dtype = DTYPE_MAP_NUMPY.get(dtype_str, np.float32)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

        if as_torch and HAS_TORCH:
            if dtype_str == "BF16":
                # Convert uint16 view to torch.bfloat16
                t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            elif dtype_str == "F16":
                t = torch.from_numpy(arr.copy())
            else:
                t = torch.from_numpy(arr.copy())

            if target_device != "cpu" and torch.cuda.is_available():
                t = t.to(target_device, non_blocking=True)
            return t
        else:
            return arr

    def load_tensor_rows(
        self,
        tensor_name: str,
        row_indices: Union[List[int], np.ndarray],
        target_device: str = "cpu",
        as_torch: bool = True,
    ) -> Optional[Union["torch.Tensor", np.ndarray]]:
        """
        Extract only selected rows of a 2-D tensor.

        The embedding table and LM head are 163840 x 7168 (2.35 GB each); a training step
        touches at most a few hundred of those rows, so copying the whole matrix into RAM
        is pure waste and the main source of memory pressure in the engine.
        """
        resolved_name = self.index._resolve_name(tensor_name)
        if resolved_name is None:
            return None

        shard_path, start, end, shape, dtype_str = self.index.tensor_locations[resolved_name]
        if len(shape) != 2:
            return None

        np_dtype = np.dtype(DTYPE_MAP_NUMPY.get(dtype_str, np.float32))
        row_bytes = shape[1] * np_dtype.itemsize
        mm = self._get_mmap(shard_path)

        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        out = np.empty((rows.shape[0], shape[1]), dtype=np_dtype)
        for i, r in enumerate(rows):
            if r < 0 or r >= shape[0]:
                out[i] = 0
                continue
            off = start + int(r) * row_bytes
            out[i] = np.frombuffer(mm[off:off + row_bytes], dtype=np_dtype)

        if as_torch and HAS_TORCH:
            t = torch.from_numpy(out)
            if dtype_str == "BF16":
                t = t.view(torch.bfloat16)
            if target_device != "cpu" and torch.cuda.is_available():
                t = t.to(target_device, non_blocking=True)
            return t
        return out

    def tensor_shape(self, tensor_name: str) -> Optional[List[int]]:
        """Shape of a tensor without reading any of its data."""
        resolved_name = self.index._resolve_name(tensor_name)
        if resolved_name is None:
            return None
        return self.index.tensor_locations[resolved_name][3]

    def load_tensor_row_slice(
        self,
        tensor_name: str,
        row_start: int,
        row_end: int,
        target_device: str = "cpu",
        as_torch: bool = True,
    ) -> Optional[Union["torch.Tensor", np.ndarray]]:
        """Extract a contiguous row band [row_start, row_end) of a 2-D tensor."""
        resolved_name = self.index._resolve_name(tensor_name)
        if resolved_name is None:
            return None

        shard_path, start, end, shape, dtype_str = self.index.tensor_locations[resolved_name]
        if len(shape) != 2:
            return None

        row_start = max(0, row_start)
        row_end = min(shape[0], row_end)
        if row_end <= row_start:
            return None

        np_dtype = np.dtype(DTYPE_MAP_NUMPY.get(dtype_str, np.float32))
        row_bytes = shape[1] * np_dtype.itemsize
        mm = self._get_mmap(shard_path)
        off = start + row_start * row_bytes
        raw = mm[off:off + (row_end - row_start) * row_bytes]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(row_end - row_start, shape[1])

        if as_torch and HAS_TORCH:
            t = torch.from_numpy(arr.copy())
            if dtype_str == "BF16":
                t = t.view(torch.bfloat16)
            if target_device != "cpu" and torch.cuda.is_available():
                t = t.to(target_device, non_blocking=True)
            return t
        return arr

    def close(self) -> None:
        """Close all open mmap descriptors."""
        for shard_path, (mm, fd) in self._mmap_handles.items():
            try:
                mm.close()
                os.close(fd)
            except Exception:
                pass
        self._mmap_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
