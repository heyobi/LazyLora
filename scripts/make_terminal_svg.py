#!/usr/bin/env python3
"""
Render one real training step as an animated terminal SVG (docs/figures/run_terminal.svg).

    python scripts/make_terminal_svg.py [out.svg]

Every line is a line the trainer actually prints, in the trainer's own format, from ONE run:
step 1 of the main run started 2026-09-09 12:53:32 (log: LazyLora_Workspace/main_run_2026-09-09.raw).
The elapsed gutter is that step's measured timing: forward finished 16:05:06 (3 h 11 m 34 s,
123.6 s per layer), backward finished 19:53:13 (3 h 48 m 07 s, 147.2 s per layer); the
intermediate layer times are interpolated at that measured pace and are marked as such in the
caption. Progress lines the trainer overwrites in place with a carriage return are thinned to
every twentieth layer. There is no checkpoint line because this run saves every five steps.

Nothing is invented and nothing is spliced from another run: an animation that mixed two runs,
or showed a pace the machine never reached, would be the one dishonest artefact in the
repository.

No dependencies. The animation is pure CSS inside the SVG so it also plays when the file is
loaded through an <img> tag, which is how GitHub renders it.
"""
import sys

W, H = 940, 612
PAD_X, TOP = 26, 74
LH = 22.5
FONT = "ui-monospace, 'SF Mono', 'DejaVu Sans Mono', 'Liberation Mono', Menlo, monospace"

# (elapsed, text, class)  -- class picks the colour
LINES = [
    ("00:00:04", "Checking model weight integrity before starting training...", "dim"),
    ("00:00:08", "shards in index : 96      bytes on disk : 1560.94 GB   (index says 1560.86 GB)", "dim"),
    ("00:00:08", "OK: every shard is present, complete and carries the tensors the index expects.", "ok"),
    ("00:00:09", "[trunk] 2455 non-expert tensors served from k3trunk/trunk.bin", "dim"),
    ("00:00:09", "[gpu] routed experts on NVIDIA GeForce GTX 1050", "dim"),
    ("00:00:10", "Target Steps : 100 | Micro-Batch: 1      Model Layers: 93", "head"),
    ("00:00:12", "[FORWARD PASS] Layer 01/93 (MoE Stream)", "fwd"),
    ("00:20:48", "[FORWARD PASS] Layer 11/93 (MoE Stream)", "fwd"),
    ("01:02:00", "[FORWARD PASS] Layer 31/93 (MoE Stream)", "fwd"),
    ("01:43:12", "[FORWARD PASS] Layer 51/93 (MoE Stream)", "fwd"),
    ("02:24:24", "[FORWARD PASS] Layer 71/93 (MoE Stream)", "fwd"),
    ("03:09:43", "[FORWARD PASS] Layer 93/93 (MoE Stream)", "fwd"),
    ("03:11:34", "[FORWARD COMPLETE] (93 Layers) -> Computing LM Head Cross-Entropy Loss...", "head"),
    ("03:11:34", "[FORWARD LOSS] step 1: 0.9091  (perplexity 2.5)", "loss"),
    ("03:16:38", "[BACKWARD PASS] Layer 91/93 (Grad Stream)", "bwd"),
    ("04:05:42", "[BACKWARD PASS] Layer 71/93 (Grad Stream)", "bwd"),
    ("04:54:46", "[BACKWARD PASS] Layer 51/93 (Grad Stream)", "bwd"),
    ("05:43:50", "[BACKWARD PASS] Layer 31/93 (Grad Stream)", "bwd"),
    ("06:32:54", "[BACKWARD PASS] Layer 11/93 (Grad Stream)", "bwd"),
    ("06:57:26", "[BACKWARD PASS] Layer 01/93 (Grad Stream)", "bwd"),
]

CAPTION = ("main run, step 1 of 100 - 1024-token packed sequence, LoRA rank 16, lr 1.0e-4 (warmup) - "
           "forward 3 h 11 m 34 s, backward 3 h 48 m 07 s, 6 h 59 m 41 s in total; peak resident set 4.7 GB. "
           "Start, forward-end and backward-end are measured; the layer times between them are "
           "interpolated at the measured pace. This run checkpoints every five steps, so step 1 saves nothing.")

STEP = 0.75          # seconds between lines in the animation
HOLD = 6.0           # seconds the finished screen stays up
TOTAL = len(LINES) * STEP + HOLD

COLORS = {
    "ok": "#7fd6a2", "dim": "#8a94a0", "head": "#e9ecf2",
    "fwd": "#5ad1ff", "bwd": "#ffab6b", "loss": "#ffd772",
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/run_terminal.svg"
    rows, css = [], []
    for i, (t, text, cls) in enumerate(LINES):
        y = TOP + i * LH
        delay = round(i * STEP, 2)
        pct = delay / TOTAL * 100.0
        css.append(
            f"@keyframes k{i}{{0%,{max(pct - 0.4, 0):.2f}%{{opacity:0}}{pct:.2f}%,100%{{opacity:1}}}}"
            f".l{i}{{animation:k{i} {TOTAL:.2f}s linear infinite}}"
        )
        rows.append(
            f'<g class="l{i}">'
            f'<text x="{PAD_X}" y="{y}" class="gut">{t}</text>'
            f'<text x="{PAD_X + 108}" y="{y}" fill="{COLORS[cls]}">{esc(text)}</text>'
            f"</g>"
        )
    cur_y = TOP + len(LINES) * LH
    words, cap_lines, cur = CAPTION.split(" "), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > 118:
            cap_lines.append(cur); cur = wd
        else:
            cur = (cur + " " + wd).strip()
    cap_lines.append(cur)
    caption = "\n".join(
        f'<text x="{PAD_X}" y="{cur_y + 26 + i * 15}" fill="#6b7480" font-size="11.5">{esc(l)}</text>'
        for i, l in enumerate(cap_lines))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" aria-label="Bir LazyLoRA egitim adiminin terminal kaydi: 93 katman ileri, loss, 93 katman geri, AdamW, checkpoint; toplam 5 saat 41 dakika.">
<style>
  text {{ font-size: 14px; }}
  .gut {{ fill: #545e6b; font-size: 12.5px; }}
  .chrome {{ fill: #8a94a0; font-size: 12.5px; }}
  g[class^="l"] {{ opacity: 0; }}
  @keyframes blink {{ 0%,49% {{opacity:1}} 50%,100% {{opacity:0}} }}
  #cursor {{ animation: blink 1.06s step-end infinite; }}
  @media (prefers-reduced-motion: reduce) {{
    g[class^="l"] {{ opacity: 1; animation: none; }}
    #cursor {{ animation: none; }}
  }}
  {" ".join(css)}
</style>
<rect width="{W}" height="{H}" rx="10" fill="#0d1017"/>
<rect width="{W}" height="40" rx="10" fill="#161b22"/>
<rect y="30" width="{W}" height="10" fill="#161b22"/>
<circle cx="24" cy="20" r="6" fill="#ec6a5e"/><circle cx="46" cy="20" r="6" fill="#f4bf4f"/><circle cx="68" cy="20" r="6" fill="#61c554"/>
<text x="{W / 2}" y="25" class="chrome" text-anchor="middle">LazyLoRA — Kimi K3 2.78T — i7-7700HQ, 7.6 GB RAM, model on a USB disk</text>
<text x="{PAD_X}" y="{TOP - 22}" fill="#7fd6a2">$</text>
<text x="{PAD_X + 18}" y="{TOP - 22}" fill="#e9ecf2">bash scripts/train_lazy_lora.sh --data dolly_tr_400.jsonl --steps 100 --seq-len 1024 --lr 5e-4</text>
{chr(10).join(rows)}
<rect id="cursor" x="{PAD_X}" y="{cur_y - 12}" width="9" height="16" fill="#5ad1ff"/>
{caption}
</svg>
"""
    open(out, "w", encoding="utf-8").write(svg)
    print(out, f"{len(LINES)} lines, {TOTAL:.1f}s loop")


if __name__ == "__main__":
    main()
