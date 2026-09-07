"""
Frozen-weight linear layers computed in float32.

On this CPU a bf16 matmul has no fast path: PyTorch falls back to a reference GEMM
(`cpublas_gemm_impl`), and a layer's backward through the attention projections took
~13 minutes. The weights stay bf16 in memory (they are streamed from disk and never
change); only the product runs in fp32, with the weight widened on the fly and dropped
afterwards. The output keeps the input's dtype so the forward numerics are unchanged
(same fp32 accumulation, same final rounding).

Only the input gets a gradient: base weights are frozen by LoRA's definition.
"""
import torch
import torch.nn.functional as F


class _Linear32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        ctx.x_dtype = x.dtype
        return F.linear(x.float(), weight.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (weight,) = ctx.saved_tensors
        grad_x = torch.matmul(grad_out.float(), weight.float())
        return grad_x.to(ctx.x_dtype), None


def linear32(x, weight):
    """x @ weight.T with the product in float32; weight is frozen (no gradient)."""
    if weight.is_cuda or x.is_cuda:
        return F.linear(x, weight)
    return _Linear32.apply(x, weight)


def conv1d32(x, weight, groups):
    """Depthwise causal conv in float32 (bf16 conv hits the same slow path)."""
    y = F.conv1d(x.float(), weight.float(), groups=groups)
    return y.to(x.dtype)
