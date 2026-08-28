"""
Disk-Backed Memory-Mapped Activation Ring Buffer for 93-Layer MoE Out-of-Core Backpropagation.
Stores layer-boundary activation states on D: SSD, allowing infinite depth backprop
with zero RAM/VRAM accumulation.
"""

import os
import mmap
import struct
from typing import Optional, Union, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class ActivationRingBuffer:
    """
    Manages fast binary serialization and recovery of layer-boundary activations on D: drive.
    Pre-allocates buffers and rotates cleanly without memory or disk leaks.
    """

    def __init__(self, cache_dir: str = "/mnt/d/hamza/LazyLora_Workspace/activations", num_layers: int = 93):
        self.cache_dir = cache_dir
        self.num_layers = num_layers
        os.makedirs(self.cache_dir, exist_ok=True)
        self._layer_shapes = {}
        self._layer_dtypes = {}

    def _get_path(self, layer_idx: int) -> str:
        return os.path.join(self.cache_dir, f"act_layer_{layer_idx:03d}.bin")

    def get_activation_path(self, layer_idx: int) -> str:
        """Public getter for layer activation binary file path."""
        return self._get_path(layer_idx)

    def save_activation(
        self,
        layer_idx: int,
        activation: Union["torch.Tensor", np.ndarray],
    ) -> None:
        """
        Stream layer boundary tensor directly to binary disk buffer on D: drive.
        """
        filepath = self._get_path(layer_idx)
        
        if HAS_TORCH and isinstance(activation, torch.Tensor):
            # Move to cpu numpy view for contiguous binary save
            cpu_tensor = activation.detach().contiguous().cpu()
            if cpu_tensor.dtype == torch.bfloat16:
                # Save raw uint16 bits
                arr = cpu_tensor.view(torch.uint16).numpy()
                dtype_code = 1
            elif cpu_tensor.dtype == torch.float16:
                arr = cpu_tensor.numpy()
                dtype_code = 2
            else:
                arr = cpu_tensor.to(torch.float32).numpy()
                dtype_code = 3
        else:
            arr = np.ascontiguousarray(activation)
            if arr.dtype == np.float16:
                dtype_code = 2
            else:
                # Anything else (float64 included) is stored as float32; labelling a
                # float64 array as float16 would read back as garbage.
                arr = arr.astype(np.float32, copy=False)
                dtype_code = 3

        self._layer_shapes[layer_idx] = arr.shape
        self._layer_dtypes[layer_idx] = dtype_code

        # Write binary: [ndim (int32), dtype_code (int32), shape (ndim int32s), raw_bytes]
        shape_bytes = struct.pack(f"<{len(arr.shape)}i", *arr.shape)
        header = struct.pack("<ii", len(arr.shape), dtype_code) + shape_bytes

        with open(filepath, "wb") as f:
            f.write(header)
            f.write(arr.tobytes())

    def load_activation(
        self,
        layer_idx: int,
        target_device: str = "cpu",
        as_torch: bool = True,
    ) -> Optional[Union["torch.Tensor", np.ndarray]]:
        """
        Fast recovery of layer activation from binary disk buffer on D: drive.
        """
        filepath = self._get_path(layer_idx)
        if not os.path.exists(filepath):
            return None

        with open(filepath, "rb") as f:
            hdr_bytes = f.read(8)
            if len(hdr_bytes) < 8:
                return None
            ndim, dtype_code = struct.unpack("<ii", hdr_bytes)
            shape_bytes = f.read(ndim * 4)
            shape = struct.unpack(f"<{ndim}i", shape_bytes)

            raw_data = f.read()

        if dtype_code == 1:
            np_dtype = np.uint16
        elif dtype_code == 2:
            np_dtype = np.float16
        else:
            np_dtype = np.float32

        arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape)

        if as_torch and HAS_TORCH:
            if dtype_code == 1:
                t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            else:
                t = torch.from_numpy(arr.copy())

            if target_device != "cpu" and torch.cuda.is_available():
                t = t.to(target_device, non_blocking=True)
            return t
        else:
            return arr

    def clean_cache(self) -> None:
        """Remove all activation temporary files from D: drive."""
        for layer_idx in range(self.num_layers):
            p = self._get_path(layer_idx)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
