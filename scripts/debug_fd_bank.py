#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ibrahim Polat
"""
Why does the residual-bank direction fail its finite-difference check on the tiny model?

    LAZYLORA_COMPUTE_FP32=1 python scripts/debug_fd_bank.py

Diagnostic, not a test. The quickstart's step 7 reported, on a 4-layer synthetic model,
layer 3 residual bank: analytic 1.086830 against a central difference of 1.121618 at
eps 1.8e-3, a relative error of 3.1e-2 where the tolerance is 2e-2. Every other direction
agreed. This script decides between the two explanations:

  truncation  the analytic gradient is right and eps is too large for this direction, in
              which case the error falls as eps^2 and Richardson extrapolation lands on
              the analytic value;
  a real bug  the difference quotient converges to something else as eps shrinks, in which
              case the bank gradient is wrong and the per-entry probe says which entry.

It runs the same setup as the quickstart (same seed, same tiny model, same layer) and then,
for the bank direction only: sweeps eps over five decades; repeats over several random
directions; probes each bank entry separately; and probes single coordinates, where the
analytic value is one element of the gradient and nothing can hide in a dot product.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from make_tiny_model import load_tiny_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402

N = int(os.environ.get("QS_SEQ_LEN", "64"))
L = int(os.environ.get("QS_FD_LAYER", "3"))
SEED = int(os.environ.get("QS_SEED", "0"))


def main():
    cfg, _ = load_tiny_config(os.environ["LAZYLORA_MODEL_DIR"])
    cfg.training.max_seq_len = N
    cfg.lora.r = 8
    cfg.lora.lora_alpha = 16
    trainer = LazyLoRATrainer(cfg)
    assert trainer.compute_dtype == torch.float32, "LAZYLORA_COMPUTE_FP32=1 did not take effect"

    V = cfg.model.vocab_size
    ids = ((torch.arange(min(N, 8), dtype=torch.long) * 37 + 11) % (V - 8) + 3).unsqueeze(0)

    torch.manual_seed(SEED)
    with torch.no_grad():
        h = trainer._embed_tokens(ids)
        trainer._reset_block_residual()
        for l in range(L):
            h, _ = trainer.forward_layer(l, h)
    h_in = h.detach().clone()
    bank = None if trainer._block_residual is None else trainer._block_residual.detach().clone()
    if bank is None:
        print("layer has no bank; nothing to diagnose")
        return 0

    bundle = trainer.lora_layers[L]
    with torch.no_grad():
        for _name, mod in bundle.all_modules():
            mod.lora_B.normal_(std=0.02)
    mods_params = trainer._lora_params_of(bundle)
    params = [p for _m, p in mods_params]

    def run(h_in_, bank_):
        with torch.no_grad():
            out, _ = trainer._run_layer(L, h_in_, bank_)
            trainer.expert_streamer.evict_layer_experts(L)
        return out

    out0 = run(h_in, bank)
    R64 = torch.randn_like(out0).double()

    def loss_at(bank_):
        return float((run(h_in, bank_).double() * R64).sum())

    for p in params:
        p.grad = None
    with torch.no_grad():
        trainer.act_buffer.save_activation(L, h_in)
    trainer.backward_layer(L, R64.float())
    entries = trainer._bank_entries_before(L)
    grad_bank = torch.stack(
        [trainer._grad_bank[l].reshape(-1, h_in.shape[-1]) for l in entries], dim=1
    ).double()

    print(f"layer {L}: {len(entries)} bank entries {entries}, bank shape {tuple(bank.shape)}, "
          f"|bank| {float(bank.norm()):.4f}, |grad_bank| {float(grad_bank.norm()):.4e}")
    print(f"is_boundary({L}) = {trainer._is_boundary(L)}   dtype {bank.dtype}")

    def fd(d, eps):
        return (loss_at(bank + eps * d) - loss_at(bank - eps * d)) / (2 * eps)

    # ---- 1. eps sweep on the direction the quickstart would have drawn
    gen = torch.Generator().manual_seed(SEED + 1)
    d = torch.randn(bank.shape, generator=gen)
    d = d / d.norm()
    analytic = float((grad_bank * d.double()).sum())
    print(f"\n1. eps sweep, one random direction. analytic = {analytic:.9e}")
    print(f"   {'eps':>10s} {'central diff':>16s} {'rel err':>10s} {'richardson':>16s} {'rel err':>10s}")
    prev = None
    for e in (1e-1, 3e-2, 1e-2, 3e-3, 1.8e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        f1 = fd(d, e)
        f2 = fd(d, e / 2)
        rich = (4 * f2 - f1) / 3          # Richardson: cancels the eps^2 term
        r1 = abs(analytic - f1) / max(abs(analytic), 1e-30)
        r2 = abs(analytic - rich) / max(abs(analytic), 1e-30)
        print(f"   {e:10.1e} {f1:16.9e} {r1:10.2e} {rich:16.9e} {r2:10.2e}")
        prev = f1
    del prev

    # ---- 2. several directions at one small eps
    print("\n2. five random directions at eps 1e-4 (Richardson)")
    print(f"   {'dir':>4s} {'analytic':>16s} {'richardson':>16s} {'rel err':>10s}")
    for k in range(5):
        dk = torch.randn(bank.shape, generator=gen)
        dk = dk / dk.norm()
        a = float((grad_bank * dk.double()).sum())
        f1, f2 = fd(dk, 1e-4), fd(dk, 5e-5)
        rich = (4 * f2 - f1) / 3
        print(f"   {k:4d} {a:16.9e} {rich:16.9e} {abs(a - rich) / max(abs(a), 1e-30):10.2e}")

    # ---- 3. one bank entry at a time
    print("\n3. one bank entry at a time, eps 1e-4 (Richardson)")
    print(f"   {'entry':>6s} {'layer':>6s} {'analytic':>16s} {'richardson':>16s} {'rel err':>10s}")
    for j, l_entry in enumerate(entries):
        dj = torch.zeros_like(bank)
        block = torch.randn(bank[:, j].shape, generator=gen)
        dj[:, j] = block / block.norm()
        a = float((grad_bank * dj.double()).sum())
        f1, f2 = fd(dj, 1e-4), fd(dj, 5e-5)
        rich = (4 * f2 - f1) / 3
        print(f"   {j:6d} {l_entry:6d} {a:16.9e} {rich:16.9e} {abs(a - rich) / max(abs(a), 1e-30):10.2e}")

    # ---- 4. single coordinates: no dot product to hide in
    print("\n4. single coordinates, eps 1e-4 (Richardson)")
    print(f"   {'index':>18s} {'analytic':>16s} {'richardson':>16s} {'rel err':>10s}")
    flat = grad_bank.reshape(-1)
    order = torch.argsort(flat.abs(), descending=True)[:6]
    for idx in order.tolist():
        dj = torch.zeros(bank.numel())
        dj[idx] = 1.0
        dj = dj.view_as(bank)
        a = float(flat[idx])
        f1, f2 = fd(dj, 1e-4), fd(dj, 5e-5)
        rich = (4 * f2 - f1) / 3
        pos = tuple(int(x) for x in torch.unravel_index(torch.tensor(idx), bank.shape))
        print(f"   {str(pos):>18s} {a:16.9e} {rich:16.9e} {abs(a - rich) / max(abs(a), 1e-30):10.2e}")

    # ---- 5. is the loss even smooth here? second differences of the raw loss
    print("\n5. loss along the first direction, to see the curvature the quotient fights")
    base = loss_at(bank)
    print(f"   f(0) = {base:.12e}")
    for e in (1e-2, 1e-3, 1e-4):
        fp, fm = loss_at(bank + e * d), loss_at(bank - e * d)
        print(f"   eps {e:8.1e}  f(+)-f(0) {fp - base:14.6e}  f(0)-f(-) {base - fm:14.6e}  "
              f"second difference {fp - 2 * base + fm:14.6e}")

    trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
