"""
Native MXFP4 kernels (see mxfp4_gemm.c). `kernel()` returns a wrapper or None when the
shared library is missing and cannot be built; callers fall back to the dequantising path.
Set LAZYLORA_NO_NATIVE=1 to force the fallback (useful for A/B checks).
"""
import ctypes
import os
import subprocess
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "libmxfp4.so")
_lib = None
_tried = False


class MXFP4Kernel:
    def __init__(self, lib):
        self.lib = lib
        f32 = ctypes.POINTER(ctypes.c_float)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        lib.mxfp4_gemm.argtypes = [f32, f32, u8, u8, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.mxfp4_gemm.restype = None
        lib.mxfp4_gemm_t.argtypes = [f32, f32, u8, u8, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.mxfp4_gemm_t.restype = None
        lib.mxfp4_threads.restype = ctypes.c_int
        lib.mxfp4_dequant.argtypes = [f32, u8, u8, ctypes.c_int, ctypes.c_int]
        lib.mxfp4_dequant.restype = None

    @staticmethod
    def _ptr(t, ctype):
        return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctype))

    def gemm(self, x, packed, scales):
        """y[M, R] = x[M, K] @ W^T for packed W[R, K]. x float32 contiguous."""
        import torch
        x = x.contiguous().float()
        M, K = x.shape
        R = packed.shape[0]
        assert packed.shape[1] * 2 == K and scales.shape == (R, K // 32), (packed.shape, scales.shape, K)
        packed = packed.contiguous()
        scales = scales.contiguous()
        y = torch.empty((M, R), dtype=torch.float32)
        self.lib.mxfp4_gemm(self._ptr(y, ctypes.c_float), self._ptr(x, ctypes.c_float),
                            self._ptr(packed, ctypes.c_uint8), self._ptr(scales, ctypes.c_uint8), M, K, R)
        return y

    def dequant(self, packed, scales):
        """W[R, K] float32 from the packed bytes (fast C decode, ~15 ms per expert matrix)."""
        import torch
        R = packed.shape[0]
        K = packed.shape[1] * 2
        packed = packed.contiguous()
        scales = scales.contiguous()
        w = torch.empty((R, K), dtype=torch.float32)
        self.lib.mxfp4_dequant(self._ptr(w, ctypes.c_float), self._ptr(packed, ctypes.c_uint8),
                               self._ptr(scales, ctypes.c_uint8), K, R)
        return w

    def gemm_t(self, dy, packed, scales):
        """y[M, K] = dy[M, R] @ W for packed W[R, K]. dy float32 contiguous."""
        import torch
        dy = dy.contiguous().float()
        M, R = dy.shape
        K = packed.shape[1] * 2
        assert packed.shape[0] == R and scales.shape == (R, K // 32)
        packed = packed.contiguous()
        scales = scales.contiguous()
        y = torch.zeros((M, K), dtype=torch.float32)
        self.lib.mxfp4_gemm_t(self._ptr(y, ctypes.c_float), self._ptr(dy, ctypes.c_float),
                              self._ptr(packed, ctypes.c_uint8), self._ptr(scales, ctypes.c_uint8), M, K, R)
        return y


def kernel() -> Optional[MXFP4Kernel]:
    global _lib, _tried
    if os.environ.get("LAZYLORA_NO_NATIVE", "") == "1":
        return None
    if _lib is not None:
        return _lib
    if _tried:
        return None
    _tried = True
    if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(os.path.join(_HERE, "mxfp4_gemm.c")):
        try:
            subprocess.run([os.path.join(_HERE, "build.sh")], check=True, capture_output=True)
        except Exception as exc:
            print(f"[!] MXFP4 native kernel not available ({exc}); using the dequantising path", flush=True)
            return None
    try:
        _lib = MXFP4Kernel(ctypes.CDLL(_SO))
    except OSError as exc:
        print(f"[!] MXFP4 native kernel failed to load ({exc}); using the dequantising path", flush=True)
        return None
    return _lib
