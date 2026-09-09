#!/usr/bin/env python3
"""
Before/after evidence: answer the same prompts with the LoRA adapter on and off, and
write both answers side by side to one JSON file.

    python scripts/demo_generate.py --checkpoint <lora_step_00100.pt> \
                                    --prompts my_prompts.jsonl \
                                    --out <workspace>/demo_generate.json

The prompts file is one JSON object per line in the same shape as the training data -
{"instruction": "...", "input": "..."} , with "input" optional - or a .json list of those,
or a .txt with one prompt per line. The output JSON carries, per prompt, the answer with
the adapter off and the answer with it on, side by side.

=======================================================================================
READ THIS BEFORE STARTING IT: WHAT ONE TOKEN COSTS
=======================================================================================
This engine has no KV cache. Every generated token is a complete forward sweep of all 93
layers, which means streaming the whole non-expert trunk plus every routed expert the
batch touches off the disk again. On the author's machine that should be roughly FIVE
MINUTES per token - an estimate, not a measurement. Where it comes from: one sweep at a
single token reads the top-16 routed experts of each of the 92 MoE layers off the USB
disk, 16 x 17.5 MB x 92 = about 26 GB, plus the 108.8 GB packed non-expert trunk off the
NVMe. The expert reads alone are four to seven minutes at the read rates measured during
training - 110 MB/s aggregate over the USB disk and the NVMe trunk together (3.22 TB
through read() in 8 h 06 m 57 s of the main run) and 61 MB/s effective within a single
layer sweep (14.5 GB per MoE layer in 238 s, Bulgular.md 16.5). No generation has ever
been run on this engine, so the first run replaces this number; --minutes-per-token
overrides it in the meantime.

On that estimate a fifty-token answer is about a four-hour job, and answering five prompts
twice (adapter off, adapter on) at fifty tokens each is about forty hours.

Plan accordingly:
  * start with --max-new-tokens 24 and two or three prompts;
  * run it under nohup / systemd, not in a terminal you will close;
  * every token is flushed to the output file as soon as it exists, so an interrupted run
    loses at most the token in flight, and --resume picks the same file up again;
  * --dry-run prints the plan and the time estimate and does nothing else.

The cost is per sweep, not per token of context, so a long prompt is nearly free and a
long answer is not. Each sweep goes through `forward_layer`, the production forward, so
it also writes the layer-boundary activations to the fast scratch; they are deleted after
every token, and the peak is one token's worth.
=======================================================================================

---------------------------------------------------------------------------------------
!! WRITTEN WITHOUT BEING EXECUTED !!
Written by reading the engine while the machine was busy with the 100-step training run,
so it has never been run. Validate it once the run frees the machine - the cheap way is
against the tiny model from scripts/make_tiny_model.py:

    python scripts/make_tiny_model.py /tmp/tiny
    LAZYLORA_MODEL_DIR=/tmp/tiny LAZYLORA_WORKSPACE_DIR=/tmp/tinyws \
    LAZYLORA_FAST_SCRATCH_DIR=/tmp/tinyscratch LAZYLORA_TRUNK_DIR= \
      python scripts/demo_generate.py --arch-json /tmp/tiny/config.json --adapter off \
        --prompts /tmp/prompts.txt --max-new-tokens 4 --out /tmp/out.json

which exercises every line of this file in seconds (the tiny model has no tokenizer, so
it falls back to byte ids and the "answer" is noise - that is fine, it is a smoke test).
---------------------------------------------------------------------------------------

HOW THE PROMPT IS BUILT
-----------------------
Exactly as the training data path builds it, so that what is generated continues from the
same point the loss was computed at. `StreamingDatasetIterator.tokenize_with_prefix`
(lazy_lora/dataset/stream_dataset.py:76-114) renders a record as

    apply_chat_template([{"role": "user", ...}], tokenize=True, add_generation_prompt=True)

and calls the result the *prompt*: system + user turn + the assistant header + the opening
of the empty think block (`<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`,
stream_dataset.py:118-126). Training masks the loss over exactly those tokens
(stream_dataset.py:158) and trains on everything after them. This script starts
generation at the same token, using the model's own template from the checkpoint
directory - no hand-written prompt format anywhere.

WHAT "ADAPTER OFF" MEANS
------------------------
Not a second model and not an approximation: LoRA's B matrix is initialised to exactly
zero (lazy_lora/core/lora_layer.py:83-87) and the adapter's contribution is
`(alpha/r) * (x @ A.T) @ B.T` (lora_layer.py:104-123), so a freshly constructed trainer
with B = 0 computes the base model exactly. "Off" therefore zeroes B; "on" loads the
checkpoint; a numeric --adapter scales `mod.scaling`, which multiplies the delta and
nothing else.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from lazy_lora.core.config import get_default_config  # noqa: E402
from lazy_lora.trainer.lazy_trainer import LazyLoRATrainer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------ inputs

def load_prompts(path):
    """A list of {"prompt": str, "source": dict} from .jsonl, .json or .txt."""
    ext = os.path.splitext(path)[1].lower()
    out = []
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append({"prompt": line, "source": {"instruction": line}})
        return out

    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise SystemExit(f"{path} must hold a list of prompts, not a {type(records).__name__}")
    else:                                        # .jsonl and anything else
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    for rec in records:
        if isinstance(rec, str):
            out.append({"prompt": rec, "source": {"instruction": rec}})
            continue
        instruction = (rec.get("instruction") or rec.get("prompt") or "").strip()
        user_input = (rec.get("input") or rec.get("context") or "").strip()
        if not instruction:
            raise SystemExit(f"a prompt record has neither 'instruction' nor 'prompt': {rec}")
        # The same join the training data path uses (stream_dataset.py:94).
        prompt = f"{instruction}\n\n{user_input}".strip() if user_input else instruction
        out.append({"prompt": prompt, "source": {"instruction": instruction, "input": user_input}})
    return out


def load_tokenizer(model_dir):
    """Kimi K3's own tokenizer, from the checkpoint directory, with its chat template."""
    sys.path.insert(0, model_dir)                # the tokenizer ships its own module
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def build_prompt_ids(tokenizer, prompt, max_len):
    """The generation prompt, byte for byte as the trainer's prompt mask defines it."""
    messages = [{"role": "user", "content": prompt}]
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if max_len and len(ids) > max_len:
        raise SystemExit(f"prompt is {len(ids)} tokens, over --max-prompt-tokens {max_len}")
    return ids


def stop_token_ids(tokenizer, cfg):
    """Ids that end an assistant turn: the config's EOS plus whatever the template uses."""
    ids = {int(cfg.model.eos_token_id)}
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    for marker in ("<|end_of_msg|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            continue
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
            ids.add(int(tid))
    return ids


# ------------------------------------------------------------------------- the adapter

def adapter_plan(spec):
    """--adapter -> [(name, scale or None)], in the order they are generated."""
    if spec == "both":
        return [("base", None), ("lora", 1.0)]
    if spec == "off":
        return [("base", None)]
    if spec == "on":
        return [("lora", 1.0)]
    try:
        scale = float(spec)
    except ValueError:
        raise SystemExit("--adapter takes on, off, both, or a number to scale the delta by")
    return [(f"lora@{scale:g}", scale)]


def base_scalings(trainer):
    return {id(mod): mod.scaling
            for bundle in trainer.lora_layers for _name, mod in bundle.all_modules()}


def set_adapter(trainer, checkpoint, scale, base, quiet=False):
    """Put the adapter in the requested state. `scale=None` means off (B set to zero)."""
    modules = [mod for bundle in trainer.lora_layers for _name, mod in bundle.all_modules()]
    if scale is None:
        with torch.no_grad():
            for mod in modules:
                mod.lora_B.zero_()
                mod.scaling = base[id(mod)]
        if not quiet:
            print("[adapter] OFF - every lora_B is zero, so the adapter contributes exactly "
                  "nothing and this is the base model", flush=True)
        return {"state": "off", "scale": 0.0, "checkpoint": None}

    meta = trainer.load_checkpoint(checkpoint)
    for mod in modules:
        mod.scaling = base[id(mod)] * scale
    if not quiet:
        print(f"[adapter] ON  - {os.path.basename(checkpoint)} (step {meta['step']}), "
              f"delta scaled by {scale:g}", flush=True)
    return {"state": "on", "scale": scale, "checkpoint": checkpoint,
            "checkpoint_step": meta["step"]}


# ------------------------------------------------------------------------- the sampler

def pick_token(logits, temperature, top_p, generator):
    """Greedy when temperature <= 0, else temperature + nucleus sampling."""
    if temperature <= 0.0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_before = sorted_probs.cumsum(-1) - sorted_probs
        keep = cumulative_before < top_p          # always keeps at least the top token
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = int(torch.multinomial(sorted_probs, 1, generator=generator))
        return int(sorted_idx[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


def forward_last_logits(trainer, cfg, ids):
    """One full sweep over `ids`; returns the logits of the last position only.

    The LM head is 163840 x 7168 and `_project_lm_head` streams it band by band
    (lazy_trainer.py:358-386, :405-426). Projecting only the last row keeps that to a
    single [1, vocab] result instead of [tokens, vocab], which at 163840 columns is the
    difference between 0.7 MB and a few hundred.
    """
    with torch.no_grad():
        h = trainer._embed_tokens(torch.tensor([ids], dtype=torch.long))
        trainer._reset_block_residual()
        for layer in range(cfg.model.num_hidden_layers):
            h, _experts = trainer.forward_layer(layer, h)
        h = trainer._finalize_hidden(h, trainer._block_residual)
        logits = trainer._project_lm_head(h[:, -1:, :])
    return logits[0, -1].float()


# ------------------------------------------------------------------------ output state

def write_state(path, state):
    """Atomic rewrite: a run killed mid-write still leaves the previous complete file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def tokens_remaining(state, plan, budget):
    """Tokens still owed across every prompt and variant, for the ETA."""
    left = 0
    for record in state["results"]:
        for name, _scale in plan:
            answer = record["answers"].get(name)
            if answer and answer.get("done"):
                continue
            left += budget - (answer.get("tokens", 0) if answer else 0)
    return max(0, left)


def hms(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def git_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


# ------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Generate answers with the LoRA adapter on and off, side by side.")
    ap.add_argument("--prompts", required=True, help="prompts file (.jsonl / .json / .txt)")
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA checkpoint (.pt); required unless --adapter off")
    ap.add_argument("--model-dir", default=None, help="checkpoint directory (default: config)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <workspace>/demo_generate.json)")
    ap.add_argument("--adapter", default="both",
                    help="both (default) | on | off | a number scaling the LoRA delta")
    ap.add_argument("--max-new-tokens", type=int, default=32,
                    help="tokens per answer. Each one is a full 93-layer sweep (~5 min).")
    ap.add_argument("--max-prompt-tokens", type=int, default=512,
                    help="refuse prompts longer than this")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling, with --temperature")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (greedy ignores it)")
    ap.add_argument("--minutes-per-token", type=float, default=5.0,
                    help="only for the up-front estimate, until real timings exist")
    ap.add_argument("--arch-json", default=None,
                    help="apply the architecture in this config.json (for the tiny demo model)")
    ap.add_argument("--resume", action="store_true", help="continue an existing --out file")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = ap.parse_args()

    # -------------------------------------------------------------- refuse to start
    if os.environ.get("LAZYLORA_ALLOW_SYNTHETIC") == "1":
        raise SystemExit(
            "LAZYLORA_ALLOW_SYNTHETIC=1 is set. That lets the engine invent a tensor it "
            "cannot find, and an answer produced that way would be meaningless while "
            "looking perfectly normal. Unset it.")

    cfg = get_default_config()
    if args.arch_json:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from make_tiny_model import load_tiny_config
        cfg, applied = load_tiny_config(os.path.dirname(os.path.abspath(args.arch_json)), cfg)
        print(f"[arch] {len(applied)} fields from {args.arch_json}", flush=True)
    if args.model_dir:
        cfg.paths.base_model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    model_dir = cfg.paths.base_model_dir

    if not os.path.isdir(model_dir):
        raise SystemExit(f"model directory not found: {model_dir}\n"
                         f"Set LAZYLORA_MODEL_DIR or pass --model-dir.")
    shards = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not shards:
        raise SystemExit(f"no *.safetensors shards in {model_dir}. This script generates from "
                         f"the real model; there is nothing to stream.")
    if not os.path.isfile(args.prompts):
        raise SystemExit(f"prompts file not found: {args.prompts}")

    plan = adapter_plan(args.adapter)
    needs_ckpt = any(scale is not None for _name, scale in plan)
    if needs_ckpt:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --adapter off")
        if not os.path.isfile(args.checkpoint):
            raise SystemExit(f"LoRA checkpoint not found: {args.checkpoint}")

    prompts = load_prompts(args.prompts)
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")

    out_path = args.out or os.path.join(cfg.paths.workspace_dir, "demo_generate.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    total_tokens = len(prompts) * len(plan) * args.max_new_tokens
    print(f"model      : {model_dir} ({len(shards)} shards)")
    print(f"checkpoint : {args.checkpoint or '(none: adapter off only)'}")
    print(f"prompts    : {len(prompts)} from {args.prompts}")
    print(f"variants   : {', '.join(name for name, _ in plan)}")
    print(f"decoding   : {'greedy' if args.temperature <= 0 else f'T={args.temperature} top_p={args.top_p} seed={args.seed}'}"
          f", up to {args.max_new_tokens} new tokens")
    print(f"output     : {out_path}")
    print(f"cost       : {total_tokens} sweeps of {cfg.model.num_hidden_layers} layers; at "
          f"{args.minutes_per_token:g} min per token that is about "
          f"{hms(total_tokens * args.minutes_per_token * 60)} (h:mm:ss)")
    if args.dry_run:
        print("\n--dry-run: nothing was computed.")
        return 0

    # ------------------------------------------------------------------- state file
    state = None
    if args.resume and os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            state = json.load(f)
        if [r["prompt"] for r in state["results"]] != [p["prompt"] for p in prompts]:
            raise SystemExit(f"{out_path} was made with different prompts; use a new --out")
        print(f"[resume] continuing {out_path}", flush=True)
    if state is None:
        state = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commit": git_commit(),
            "model_dir": model_dir,
            "checkpoint": args.checkpoint,
            "layers": cfg.model.num_hidden_layers,
            "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                         "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                         "greedy": args.temperature <= 0.0},
            "note": ("Each answer is one forward sweep of every layer per token, from the "
                     "same streamed checkpoint. 'base' has every LoRA B matrix at zero, so "
                     "it is the unmodified model; the other variants load the adapter."),
            "results": [{"index": i, "prompt": p["prompt"], "source": p["source"],
                         "answers": {}} for i, p in enumerate(prompts)],
        }
        write_state(out_path, state)

    # ------------------------------------------------------------------- the engine
    print("\nloading the tokenizer and indexing the shards...", flush=True)
    tokenizer = load_tokenizer(model_dir)
    stops = stop_token_ids(tokenizer, cfg)
    trainer = LazyLoRATrainer(cfg)
    # The routed-expert sums are cached only so the backward replay can skip a sweep
    # (lazy_trainer.py:735-737); generation has no backward, so this is pure disk traffic.
    trainer.cache_routed = False
    base = base_scalings(trainer)
    # A dedicated generator, because load_checkpoint restores the global torch and numpy
    # RNG states from the checkpoint (lazy_trainer.py:1188-1191); sampling must not depend
    # on which adapter was loaded last.
    generator = torch.Generator().manual_seed(args.seed)

    prompt_ids = [build_prompt_ids(tokenizer, p["prompt"], args.max_prompt_tokens) for p in prompts]
    print(f"prompt lengths: {[len(x) for x in prompt_ids]} tokens", flush=True)

    done_tokens = 0
    spent = 0.0
    t_start = time.time()
    current_variant = None
    info = None

    for i, _prompt in enumerate(prompts):
        record = state["results"][i]
        for name, scale in plan:
            answer = record["answers"].get(name)
            if answer and answer.get("done"):
                print(f"[skip] prompt {i + 1}/{len(prompts)} {name}: already complete", flush=True)
                continue
            if current_variant != name:
                info = set_adapter(trainer, args.checkpoint, scale, base)
                current_variant = name
            if answer is None:
                answer = {"adapter": info, "token_ids": [], "text": "", "done": False,
                          "stopped_on": None, "tokens": 0, "seconds": 0.0}
                record["answers"][name] = answer

            ids = list(prompt_ids[i]) + list(answer["token_ids"])
            produced = list(answer["token_ids"])
            while len(produced) < args.max_new_tokens:
                t0 = time.time()
                logits = forward_last_logits(trainer, cfg, ids)
                token = pick_token(logits, args.temperature, args.top_p, generator)
                trainer.act_buffer.clean_cache()
                dt = time.time() - t0

                ids.append(token)
                produced.append(token)
                done_tokens += 1
                spent += dt
                previous = answer["text"]
                # A stop token is recorded in token_ids but kept out of the text, so the
                # answer reads as the model's answer and the evidence stays complete.
                visible = produced[:-1] if token in stops else produced
                answer["token_ids"] = produced
                answer["text"] = tokenizer.decode(visible)
                answer["text_clean"] = tokenizer.decode(visible, skip_special_tokens=True)
                answer["tokens"] = len(produced)
                answer["seconds"] += dt
                answer["seconds_per_token"] = round(answer["seconds"] / len(produced), 1)
                answer["peak_rss_gb"] = round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
                if token in stops:
                    answer["done"] = True
                    answer["stopped_on"] = token
                write_state(out_path, state)      # every token, before anything else

                mean = spent / done_tokens
                remaining = tokens_remaining(state, plan, args.max_new_tokens)
                delta = answer["text"][len(previous):].replace("\n", "\\n")
                print(f"[{i + 1}/{len(prompts)} {name}] token {len(produced)}/{args.max_new_tokens} "
                      f"id {token:6d}  {dt:.1f}s (mean {mean:.1f}s)  elapsed {hms(time.time() - t_start)}  "
                      f"eta {hms(max(0, remaining) * mean)}  +{delta!r}", flush=True)
                if answer["done"]:
                    break
            else:
                answer["done"] = True             # hit the token budget, not a stop token
                write_state(out_path, state)
            print(f"--- prompt {i + 1} [{name}] ---\n{answer['text']}\n", flush=True)

    trainer.close()
    print(f"\nwrote {out_path}")
    print(f"{done_tokens} tokens in {hms(time.time() - t_start)}"
          + (f" ({spent / done_tokens:.1f} s per token)" if done_tokens else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
