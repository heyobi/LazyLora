#!/usr/bin/env python3
"""
Evaluation protocol (Bulgular.md / DEVAM.md, "Değerlendirme protokolü"): next-token loss and
perplexity on the held-out Wikipedia corpora, Turkish (primary) and English (control),
as fixed 2048-token chunks run through all 93 layers.

    python scripts/eval_perplexity.py [--lang tr,en] [--chunks 2] [--checkpoint <lora.pt>] [--tag baseline]

Each chunk is one forward sweep (the engine's cost is per sweep), so a full evaluation is
4 sweeps at N=2048. Results are appended to <workspace>/eval/results.jsonl with the git
commit, the adapter checkpoint (if any), per-chunk loss, top-1 accuracy, time and peak RSS.

This is the harness that decides whether the pre-registered threshold (commit 6605306, tag
preregistration-2026-09-08) is met, so what it reads and what it writes matter more than
usual:

  * the corpus is a pre-tokenised <workspace>/eval/<corpus>_ids.json, written by
    scripts/build_eval_corpus.py or scripts/build_eval_news.py, with no BOS - this script
    prepends cfg.model.bos_token_id itself and uses the chunk as its own targets. --text
    tokenises a plain text file into that same file instead, which is how the continuous
    -integration run feeds it a corpus it wrote itself;
  * bits per byte, not perplexity, is the number the threshold is stated in. Per-token
    perplexity is not comparable across tokenizations - Turkish needs about 1.7x the
    tokens of English for the same text - so the record carries all three and the protocol
    reads bits_per_byte;
  * every record says which tokenizer produced it and whether it means anything. With
    --tokenizer bytes it does not: see below.

---------------------------------------------------------------------------------------
NEVER RUN AGAINST THE REAL CHECKPOINT; RUN ON EVERY PUSH AGAINST A SYNTHETIC ONE
No evaluation result exists yet: the 100-step run finishes around 9-11 October 2026 and
this harness runs after it. Until then it is exercised, on every push, by
.github/workflows/tools.yml - against the tiny synthetic model of
scripts/make_tiny_model.py, on a text file the workflow writes itself, with --tokenizer
bytes. That proves the chunking, the sweep, the loss, the bits-per-byte arithmetic and the
results file all work. It proves nothing whatsoever about Kimi K3, and it must not: the
weights are eight megabytes of noise and the ids are UTF-8 bytes.

    python scripts/make_tiny_model.py /tmp/tiny
    LAZYLORA_MODEL_DIR=/tmp/tiny LAZYLORA_WORKSPACE_DIR=/tmp/tinyws \
    LAZYLORA_FAST_SCRATCH_DIR=/tmp/tinyscratch LAZYLORA_TRUNK_DIR= \
      python scripts/eval_perplexity.py --arch-json /tmp/tiny/config.json \
        --tokenizer bytes --text /tmp/corpus.txt --chunk 64 --chunks 2 --tag smoke

--tokenizer bytes maps UTF-8 byte b to id b + 100 (dataset/stream_dataset.py:136). Those
ids have no relation to the vocabulary any trained checkpoint uses, so the loss computed
from them is a number about nothing. Every record written that way carries
"tokenizer": "byte-fallback" and "meaningful": false, and the script prints a banner, so a
line from a smoke test can never be read as a measurement. The default is --tokenizer
model, and with it a tokenizer that will not load is a hard error rather than a fallback.
---------------------------------------------------------------------------------------
"""
import argparse, json, os, resource, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch  # noqa: E402
# One definition of "byte id" for the whole repository: demo_generate.py owns
# ByteFallbackTokenizer and load_tokenizer, and the evaluation must tokenise a fallback
# corpus exactly the way generation decodes one, or the two smoke tests would disagree
# about what a token is.
from demo_generate import ByteFallbackTokenizer, load_tokenizer  # noqa: E402,F401
from lazy_lora.core.config import get_default_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402
from lazy_lora.trainer.loss import compute_cross_entropy_loss  # noqa: E402


def tokenise_text_file(path, tok, ids_path):
    """A plain UTF-8 text file -> the <corpus>_ids.json this harness reads.

    A flat list of ids from the same `tok.encode(text)` call scripts/build_eval_corpus.py
    makes (build_eval_corpus.py:118), so the two agree by construction; the chunk loop
    below prepends cfg.model.bos_token_id itself and uses the chunk as its own targets.
    Going through the file rather than straight to a list is deliberate - it means --text
    exercises the same reading path a real corpus takes.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = [int(i) for i in tok.encode(text)]
    if not ids:
        raise SystemExit(f"{path} tokenised to nothing")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"tokenised {path} ({len(text.encode('utf-8'))} bytes) -> {ids_path} "
          f"({len(ids)} ids)", flush=True)
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr,en", help="(legacy) languages; each maps to <lang>_wiki")
    ap.add_argument("--corpus", default=None,
                    help="comma-separated corpus names under eval/ (e.g. tr_wiki,en_wiki,tr_news); overrides --lang")
    ap.add_argument("--force", action="store_true",
                    help="let --text overwrite an existing <name>_ids.json")
    ap.add_argument("--text", default=None, metavar="FILE",
                    help="tokenise this plain UTF-8 text file into <workspace>/eval/<name>_ids.json "
                         "and evaluate that one corpus, where <name> is the file's stem unless "
                         "--corpus names exactly one. For a corpus this run creates itself; the "
                         "pre-registered evaluation uses the committed *_ids.json instead.")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=2, help="chunks per language (corpus has 4096 tokens)")
    ap.add_argument("--checkpoint", default=None, help="LoRA checkpoint to load; none = base model")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--tokenizer", default="model", choices=("model", "bytes"),
                    help="model (default): Kimi K3's own tokenizer from the checkpoint "
                         "directory, the only setting whose numbers mean anything. bytes: "
                         "the dataset iterator's byte fallback, for a model that ships no "
                         "tokenizer (the tiny demo model). Records written with 'bytes' are "
                         "stamped meaningful=false.")
    ap.add_argument("--arch-json", default=None,
                    help="apply the architecture in this config.json (for the tiny demo model). "
                         "The engine hardcodes Kimi K3's 93 layers, so without this it would "
                         "sweep 93 layers of a 4-layer checkpoint.")
    ap.add_argument("--model-dir", default=None, help="checkpoint directory (default: config)")
    args = ap.parse_args()

    cfg = get_default_config()
    if args.arch_json:
        from make_tiny_model import load_tiny_config
        cfg, applied = load_tiny_config(os.path.dirname(os.path.abspath(args.arch_json)), cfg)
        print(f"[arch] {len(applied)} fields from {args.arch_json}: "
              f"{cfg.model.num_hidden_layers} layers, vocab {cfg.model.vocab_size}", flush=True)
    if args.model_dir:
        cfg.paths.base_model_dir = os.path.abspath(os.path.expanduser(args.model_dir))

    ev = os.path.join(cfg.paths.workspace_dir, "eval")
    os.makedirs(ev, exist_ok=True)     # PathConfig.ensure_directories() does not create it
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    tok, tok_info = load_tokenizer(args.tokenizer, cfg.paths.base_model_dir, cfg)
    meaningful = bool(tok_info["meaningful_text"])

    corpora = args.corpus.split(",") if args.corpus else [f"{l}_wiki" for l in args.lang.split(",")]
    if args.text:
        if args.corpus and len(corpora) != 1:
            raise SystemExit("--text evaluates one corpus; --corpus may name at most one with it")
        name = corpora[0] if args.corpus else os.path.splitext(os.path.basename(args.text))[0]
        corpora = [name]
        ids_path = os.path.join(ev, f"{name}_ids.json")
        if os.path.exists(ids_path) and not args.force:
            raise SystemExit(f"{ids_path} already exists; --text would overwrite the corpus the "
                             f"pre-registered evaluation reads. Pick another --corpus name or "
                             f"pass --force if you really mean to replace it.")
        n_ids = tokenise_text_file(args.text, tok, ids_path)
        if n_ids < args.chunk:
            raise SystemExit(f"{args.text} gives {n_ids} ids, fewer than one --chunk of "
                             f"{args.chunk}; nothing would be evaluated")

    trainer = LazyLoRATrainer(cfg)
    if args.checkpoint:
        meta = trainer.load_checkpoint(args.checkpoint)
        print(f"loaded {args.checkpoint} (step {meta['step']})", flush=True)

    out_path = os.path.join(ev, "results.jsonl")
    written = 0
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
                   "layers": cfg.model.num_hidden_layers,
                   "vocab_size": cfg.model.vocab_size,
                   # Which vocabulary produced the number is part of the number. A record
                   # with meaningful=false came from ids that are UTF-8 bytes rather than a
                   # trained vocabulary, and says nothing about any model; the pre-registered
                   # protocol reads only records with meaningful=true.
                   "tokenizer": tok_info["name"],
                   "meaningful": meaningful,
                   "seconds": round(time.time() - t0, 1),
                   "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                   "date": time.strftime("%Y-%m-%d %H:%M")}
            if not meaningful:
                rec["note"] = ("byte-fallback ids (--tokenizer bytes): this loss measures "
                               "nothing about any model, it records that the harness ran")
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            print(f"{'' if meaningful else '[NOT A MEASUREMENT] '}"
                  f"{lang} chunk {c}: loss {rec['loss']}  ppl {rec['perplexity']}  bpb {rec['bits_per_byte']}  top1 {rec['top1_acc']}  "
                  f"{rec['seconds']:.0f}s  RSS {rec['peak_rss_gb']} GB", flush=True)
            trainer.act_buffer.clean_cache()
    trainer.close()
    if not written:
        raise SystemExit(f"no chunk was evaluated: every corpus in {corpora} was shorter "
                         f"than --chunk {args.chunk}")
    print(f"appended {written} record(s) to {out_path}", flush=True)
    if not meaningful:
        print("\nthose records are NOT measurements: --tokenizer bytes was used, so the ids "
              "are UTF-8\nbytes rather than a vocabulary. Each one carries "
              '"meaningful": false for that reason.')


if __name__ == "__main__":
    main()
