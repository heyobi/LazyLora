#!/usr/bin/env python3
"""
K8: check the analytical backward of one layer against central finite differences.

Runs entirely in float32 (LAZYLORA_COMPUTE_FP32=1 is forced here) on the real
checkpoint with a handful of tokens. For layer L it

  1. runs the forward through layers 0..L-1 to obtain h_in and the residual bank,
  2. gives the layer's LoRA B matrices a small random value (with B = 0 the gradient of A
     is identically zero and the check would be vacuous),
  3. defines the scalar loss  f = <run_layer(h_in, bank, theta), R>  for a fixed random R,
  4. takes the analytical gradients with `backward_layer` (the production path, including
     the streamed routed experts and the bank bookkeeping),
  5. compares directional derivatives  <grad, d>  against  (f(+eps d) - f(-eps d)) / 2 eps
     for random directions d over: every LoRA tensor of the layer, h_in, and the bank.

Layer 1 has one bank entry (the embedding is pushed at layer 0), layer 13 has two,
layer 12 is a boundary layer. Run at least layers 1, 12 and 13.

    python scripts/verify_backward.py --layer 1 --ids 19180,11,1632,691
"""
import argparse, os, sys, time
os.environ["LAZYLORA_COMPUTE_FP32"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lazy_lora.core.config import get_default_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--ids", default="19180,11,1632,691")
    ap.add_argument("--target-delta", type=float, default=2e-3,
                    help="eps is chosen per direction so the predicted |f(+eps)-f(-eps)| is about this")
    ap.add_argument("--param-probes", type=int, default=3)
    ap.add_argument("--b-scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2, help="relative tolerance on each directional derivative")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ids = [int(x) for x in args.ids.split(",")]
    L = args.layer
    cfg = get_default_config()
    trainer = LazyLoRATrainer(cfg)
    assert trainer.compute_dtype == torch.float32
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit(f"layer {L}  ids {ids}  target delta {args.target_delta}  fp32 engine, float64 loss")

    # 1. forward to the layer
    t0 = time.time()
    with torch.no_grad():
        h = trainer._embed_tokens(torch.tensor([ids], dtype=torch.long))
        trainer._reset_block_residual()
        for l in range(L):
            h, _ = trainer.forward_layer(l, h)
    h_in = h.detach().clone()
    bank = None if trainer._block_residual is None else trainer._block_residual.detach().clone()
    emit(f"forward to layer {L}: {time.time() - t0:.0f}s, bank entries: {0 if bank is None else bank.shape[1]}, "
         f"h_in std {h_in.std():.4f}")

    # 2. non-zero B so that A has a gradient
    bundle = trainer.lora_layers[L]
    mods_params = trainer._lora_params_of(bundle)
    with torch.no_grad():
        for _, mod in bundle.all_modules():
            mod.lora_B.normal_(std=args.b_scale)
    params = [p for _, p in mods_params]
    names = [f"{name}.{ab}" for name, mod in bundle.all_modules() for ab in ("A", "B") if mod.lora_A is not None]

    # 3. the loss and one evaluation
    def run(h_in_, bank_):
        with torch.no_grad():
            out, _ = trainer._run_layer(L, h_in_, None if bank_ is None else bank_)
            trainer.expert_streamer.evict_layer_experts(L)
        return out

    out0 = run(h_in, bank)
    R = torch.randn_like(out0)
    R64 = R.double()

    # The reduction runs in float64: a float32 sum of 4 x 7168 terms of size ~1 has an
    # absolute error around 1e-6, which is the size of the differences being measured.
    def loss_at(h_in_, bank_):
        return float((run(h_in_, bank_).double() * R64).sum())

    f0 = loss_at(h_in, bank)
    emit(f"f0 = {f0:.6f}   out std {out0.std():.4f}")

    # 4. analytical gradient through the production backward
    for p in params:
        p.grad = None
    with torch.no_grad():
        trainer.act_buffer.save_activation(L, h_in)   # forward_layer(L) would have done this
    t0 = time.time()
    grad_h_in = trainer.backward_layer(L, R)
    emit(f"backward_layer: {time.time() - t0:.0f}s")
    grad_params = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    entries = trainer._bank_entries_before(L)
    grad_bank = None
    if entries:
        grad_bank = torch.stack([trainer._grad_bank[l].reshape(-1, h_in.shape[-1]) for l in entries], dim=1)
    for p in params:
        p.grad = None

    emit("")
    emit(f"{'direction':30s} {'analytic':>13s} {'fd(eps)':>13s} {'fd(eps/2)':>13s} {'eps':>8s} {'rel err':>9s}")
    worst = 0.0
    results = []

    def pick_eps(analytic):
        # step so that the predicted change in f is ~target_delta, well above float noise
        return float(min(0.5, max(1e-4, args.target_delta / max(abs(analytic), 1e-6))))

    def central(fn, eps):
        return (fn(eps) - fn(-eps)) / (2 * eps)

    def check(label, analytic, fn):
        """fn(eps) evaluates the loss at +eps along the direction; two step sizes are reported."""
        nonlocal worst
        eps = pick_eps(analytic)
        fd1 = central(fn, eps)
        fd2 = central(fn, eps / 2)
        fd = fd2
        denom = max(abs(analytic), abs(fd), 1e-12)
        rel = abs(analytic - fd) / denom
        worst = max(worst, rel)
        results.append(rel)
        emit(f"{label:30s} {analytic:13.6e} {fd1:13.6e} {fd2:13.6e} {eps:8.1e} {rel:9.2e}{'   <-- MISMATCH' if rel > args.tol else ''}")

    # per-parameter directions: each probe perturbs one LoRA tensor along a random direction
    gen = torch.Generator().manual_seed(args.seed + 1)
    order = list(range(len(params)))
    probes = order if args.param_probes >= len(params) else \
        [order[i] for i in torch.randperm(len(order), generator=gen)[:args.param_probes].tolist()]
    for i in probes:
        p = params[i]
        d = torch.randn(p.shape, generator=gen)
        d = d / d.norm()
        analytic = float((grad_params[i] * d).sum())

        def fn(eps, p=p, d=d):
            with torch.no_grad():
                p.add_(d, alpha=eps)
                try:
                    return loss_at(h_in, bank)
                finally:
                    p.add_(d, alpha=-eps)
        check(f"param {names[i]}", analytic, fn)

    # h_in direction
    d = torch.randn_like(h_in)
    d = d / d.norm()
    analytic = float((grad_h_in.to(d.dtype) * d).sum())
    check("h_in", analytic, lambda eps, d=d: loss_at(h_in + eps * d, bank))

    # bank direction
    if bank is not None:
        d = torch.randn_like(bank)
        d = d / d.norm()
        analytic = float((grad_bank.to(d.dtype) * d).sum())
        check("bank (all entries)", analytic, lambda eps, d=d: loss_at(h_in, bank + eps * d))

    emit("")
    verdict = "PASS" if worst <= args.tol else "FAIL"
    emit(f"{verdict}: worst relative error {worst:.2e} over {len(results)} directions (tol {args.tol:.0e})")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
    trainer.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
