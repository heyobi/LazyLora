#!/usr/bin/env python3
"""
Evaluation protocol (Bulgular.md / DEVAM.md, "Değerlendirme protokolü"): next-token loss and
perplexity on the held-out Wikipedia corpora, Turkish (primary) and English (control),
as fixed 2048-token chunks run through all 93 layers.

    python scripts/eval_perplexity.py [--lang tr,en] [--chunks 2] [--checkpoint <lora.pt>] [--tag baseline]

Each chunk is one forward sweep (the engine's cost is per sweep), so a full evaluation is
4 sweeps at N=2048. Results are appended to <workspace>/eval/results.jsonl with the git
commit, the adapter checkpoint (if any), per-chunk loss, top-1 accuracy, time and peak RSS.
"""
import argparse, json, os, resource, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from lazy_lora.core.config import get_default_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402
from lazy_lora.trainer.loss import compute_cross_entropy_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr,en", help="(legacy) languages; each maps to <lang>_wiki")
    ap.add_argument("--corpus", default=None,
                    help="comma-separated corpus names under eval/ (e.g. tr_wiki,en_wiki,tr_news); overrides --lang")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=2, help="chunks per language (corpus has 4096 tokens)")
    ap.add_argument("--checkpoint", default=None, help="LoRA checkpoint to load; none = base model")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    cfg = get_default_config()
    ev = os.path.join(cfg.paths.workspace_dir, "eval")
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, cfg.paths.base_model_dir)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.paths.base_model_dir, trust_remote_code=True)
    trainer = LazyLoRATrainer(cfg)
    if args.checkpoint:
        meta = trainer.load_checkpoint(args.checkpoint)
        print(f"loaded {args.checkpoint} (step {meta['step']})", flush=True)

    out_path = os.path.join(ev, "results.jsonl")
    corpora = args.corpus.split(",") if args.corpus else [f"{l}_wiki" for l in args.lang.split(",")]
    for lang in corpora:
        ids_all = json.load(open(os.path.join(ev, f"{lang}_ids.json")))
        for c in range(args.chunks):
            chunk = ids_all[c * args.chunk:(c + 1) * args.chunk]
            if len(chunk) < args.chunk:
                break
            ids = [cfg.model.bos_token_id] + chunk[:-1]          # inputs; targets are the chunk itself
            t0 = time.time()
            with torch.no_grad():
                h = trainer._embed_tokens(torch.tensor([ids], dtype=torch.long))
                trainer._reset_block_residual()
                for l in range(cfg.model.num_hidden_layers):
                    h, _ = trainer.forward_layer(l, h)
                    if l % 10 == 0:
                        print(f"  {lang} chunk {c}: layer {l}  {time.time() - t0:.0f}s", flush=True)
                h = trainer._finalize_hidden(h, trainer._block_residual)
                logits = trainer._project_lm_head(h)
                tgt = torch.tensor([chunk], dtype=torch.long)
                loss, _ = compute_cross_entropy_loss(logits, tgt, ignore_index=cfg.model.pad_token_id)
                acc = float((logits[0].float().argmax(-1) == tgt[0]).float().mean())
            # per-token perplexity is not comparable across tokenizations (Turkish needs ~1.7x
            # the tokens of English for the same text), bits per byte of the decoded text is
            n_bytes = len(tok.decode(chunk).encode("utf-8"))
            bpb = float(loss) * args.chunk / (n_bytes * 0.6931471805599453)
            rec = {"tag": args.tag, "commit": commit, "checkpoint": args.checkpoint, "lang": lang, "chunk": c,
                   "tokens": args.chunk, "bytes": n_bytes, "loss": round(float(loss), 4),
                   "perplexity": round(float(torch.exp(loss)), 3), "bits_per_byte": round(bpb, 4),
                   "top1_acc": round(acc, 4),
                   "seconds": round(time.time() - t0, 1),
                   "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                   "date": time.strftime("%Y-%m-%d %H:%M")}
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{lang} chunk {c}: loss {rec['loss']}  ppl {rec['perplexity']}  bpb {rec['bits_per_byte']}  top1 {rec['top1_acc']}  "
                  f"{rec['seconds']:.0f}s  RSS {rec['peak_rss_gb']} GB", flush=True)
            trainer.act_buffer.clean_cache()
    trainer.close()


if __name__ == "__main__":
    main()
