#!/usr/bin/env python3
"""
Run a forward pass over a prompt and record every routing decision as an expert trace.

    python scripts/measure_routing.py --text "Merhaba dünya ..." --layers 67 --tag tr_sample1
    python scripts/measure_routing.py --text-file prompt.txt --layers 13
    python scripts/measure_routing.py --ids 19180,11 --layers 13

Writes <workspace>/traces/<tag>_<timestamp>/{trace.bin,trace.json} (see
lazy_lora.monitor.trace). The manifest also carries per-layer seconds and bytes read
and the peak RSS, so one run doubles as the machine's speed / memory profile.

The model's own tokenizer is required; there is deliberately no byte fallback here.
--layers defaults to all 93 (the checkpoint is complete since 5 September 2026).
"""
import argparse, os, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from lazy_lora.core.config import get_default_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402
from lazy_lora.monitor.trace import ExpertTraceWriter  # noqa: E402


def load_tokenizer(model_dir):
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    g.add_argument("--ids")
    ap.add_argument("--layers", type=int, default=93)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-bos", action="store_true")
    ap.add_argument("--loss", action="store_true",
                    help="after the last layer, apply the final norm and LM head and report the "
                         "next-token loss / perplexity of the prompt (needs --layers 93)")
    args = ap.parse_args()

    cfg = get_default_config()
    text = None
    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        text = args.text if args.text is not None else open(args.text_file, encoding="utf-8").read()
        tok = load_tokenizer(cfg.paths.base_model_dir)
        ids = list(tok.encode(text))[: args.max_tokens - 1]
        if not args.no_bos:
            ids = [cfg.model.bos_token_id] + ids
    out_dir = args.out_dir or os.path.join(cfg.paths.workspace_dir, "traces",
                                           f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}")

    trainer = LazyLoRATrainer(cfg)
    trainer.trace = ExpertTraceWriter(out_dir, tag=args.tag, meta={
        "ids": ids, "text": text, "n_tokens": len(ids), "layers_requested": args.layers,
        "compute_dtype": str(trainer.compute_dtype), "model_dir": cfg.paths.base_model_dir,
        "cpu": os.uname().machine, "threads": torch.get_num_threads(),
    })
    print(f"tokens: {len(ids)}   layers: {args.layers}   trace: {out_dir}", flush=True)

    t_all = time.time()
    with torch.no_grad():
        h = trainer._embed_tokens(torch.tensor([ids], dtype=torch.long))
        trainer._reset_block_residual()
        for l in range(args.layers):
            t0 = time.time()
            h, experts = trainer.forward_layer(l, h)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            # per-token L2 norms of the residual stream: where does activation growth sit?
            norms = h[0].float().norm(dim=-1)
            trainer.trace.manifest.setdefault("token_norms", {})[str(l)] = [round(float(v), 3) for v in norms]
            print(f"layer {l:2d}  {time.time() - t0:6.1f}s  experts={len(experts):3d}  "
                  f"h.std={h.float().std():.4f}  max_tok_norm={float(norms.max()):.1f}@{int(norms.argmax())}  "
                  f"peak_rss={rss:.2f} GB  read={trainer.mmap_streamer.bytes_read / 1e9:.1f} GB", flush=True)
            if not torch.isfinite(h.float()).all():
                trainer.trace.note(error=f"non-finite hidden state after layer {l}")
                print("non-finite hidden state; stopping", flush=True)
                break
        if args.loss and args.layers >= cfg.model.num_hidden_layers:
            from lazy_lora.trainer.loss import compute_cross_entropy_loss
            t0 = time.time()
            h_final = trainer._finalize_hidden(h, trainer._block_residual)
            logits = trainer._project_lm_head(h_final)
            tgt = torch.tensor([ids[1:] + [cfg.model.pad_token_id]], dtype=torch.long)
            loss, _ = compute_cross_entropy_loss(logits[:, :-1], tgt[:, :-1], ignore_index=cfg.model.pad_token_id)
            # top-1 next-token agreement over the prompt
            pred = logits[0, :-1].float().argmax(-1)
            acc = float((pred == torch.tensor(ids[1:])).float().mean())
            lv = float(loss)
            trainer.trace.note(loss=round(lv, 4), perplexity=round(float(torch.exp(torch.tensor(lv))), 2),
                               top1_acc=round(acc, 4), loss_seconds=round(time.time() - t0, 1))
            print(f"loss {lv:.4f}  perplexity {torch.exp(torch.tensor(lv)):.2f}  top-1 next-token acc {acc:.3f}  "
                  f"(final norm + LM head {time.time() - t0:.0f}s)", flush=True)
    trainer.trace.note(total_seconds=round(time.time() - t_all, 1),
                       bytes_read=int(trainer.mmap_streamer.bytes_read),
                       peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    trainer.trace.close()
    trainer.close()
    print(f"done in {time.time() - t_all:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
