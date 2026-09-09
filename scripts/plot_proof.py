#!/usr/bin/env python3
"""
Figures for the proof-of-learning run (Bulgular.md §18).

    ~/venvs/lazylora/bin/python scripts/plot_proof.py

Reads <workspace>/forward_loss.jsonl.proof (the losses of the 8-9 September run, one line
per step) and writes into docs/figures/:
    proof_loss_tr.png / _en.png   square cards for sharing (1200x1200)
    proof_loss_wide.png/.svg      two panels: the loss curve and what fits in RAM
Steps alternate between the two packed sequences A and B, so step k belongs to sequence
A when k is odd; the pass number of a sequence is (k+1)//2 for A and k//2 for B.
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "figures")

BG, FG, MUTED, GRID = "#0d1017", "#e9ecf2", "#8b93a7", "#232833"
CA, CB = "#5ad1ff", "#ffab6b"

TXT = {
    "tr": dict(
        title="2,78 trilyon parametreli bir modelin LoRA\nadaptörü, 7,6 GB RAM'li bir dizüstünde\neğitildi",
        sub="Kimi K3 · 93 katman · katman başına 896 uzman · 1,56 TB checkpoint bir USB diskte\nLoRA adaptörü çekirdek-dışı eğitim: her katman diskten akar, RAM'de tek katman durur",
        xlabel="aynı dizi üzerinde kaçıncı tur", ylabel="loss (yalnız cevap token'ları)",
        sa="A dizisi", sb="B dizisi",
        stats=[("5,5-5,8 sa", "adım (kanıt koşusu)"), ("4,5-4,7 GB", "yerleşik bellek"), ("590 MB", "eğitilen ağırlık"),
               ("1,56 TB", "diskteki model")],
        foot="Kanıt koşusu, 8-9 Eylül 2026 · beş örnek, iki paket dizi (1082 token, 528'i eğitilen) · github.com/heyobi/LazyLora"),
    "en": dict(
        title="A LoRA adapter on a 2.78-trillion-parameter\nmodel, trained out of core on a laptop\nwith 7.6 GB of RAM",
        sub="Kimi K3 · 93 layers · 896 experts per layer · a 1.56 TB checkpoint on a USB hard disk\nOut-of-core LoRA: every layer streams from disk, one layer at a time is resident",
        xlabel="pass over the same sequence", ylabel="loss (assistant tokens only)",
        sa="sequence A", sb="sequence B",
        stats=[("5.5-5.8 h", "per step (proof run)"), ("4.5-4.7 GB", "resident set"), ("590 MB", "trainable weights"),
               ("1.56 TB", "model on disk")],
        foot="Proof run, 8-9 Sep 2026 · five examples in two packed sequences (1082 tokens, 528 trained) · github.com/heyobi/LazyLora"),
}


def load():
    cfg = get_default_config()
    path = os.path.join(cfg.paths.workspace_dir, "forward_loss.jsonl.proof")
    rows = []
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("time", 0) > 1788850000:      # the proof run only
            rows.append(d)
    rows.sort(key=lambda d: d["time"])
    a = [(i // 2 + 1, d["loss"]) for i, d in enumerate(rows) if i % 2 == 0]
    b = [(i // 2 + 1, d["loss"]) for i, d in enumerate(rows) if i % 2 == 1]
    return a, b, rows


def curve(ax, a, b, t, label_size=11):
    for pts, c, name in ((a, CA, t["sa"]), (b, CB, t["sb"])):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=c, lw=2.6, ms=9, mec=BG, mew=2, label=name, zorder=3)
        for x, y in pts:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 13),
                        ha="center", color=c, fontsize=label_size, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(t["xlabel"], color=MUTED, fontsize=label_size + 0.5, labelpad=9)
    ax.set_ylabel(t["ylabel"], color=MUTED, fontsize=label_size + 0.5, labelpad=9)
    ax.grid(True, color=GRID, lw=1, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=label_size)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    leg = ax.legend(loc="upper right", frameon=False, fontsize=label_size + 1)
    for txt, c in zip(leg.get_texts(), (CA, CB)):
        txt.set_color(c)


def card(lang):
    t = TXT[lang]
    a, b, _ = load()
    fig = plt.figure(figsize=(6, 6), dpi=200, facecolor=BG)
    fig.text(0.07, 0.955, t["title"], color=FG, fontsize=15, fontweight="bold",
             va="top", linespacing=1.35)
    fig.text(0.07, 0.787, t["sub"], color=MUTED, fontsize=8.3, va="top", linespacing=1.5)
    ax = fig.add_axes([0.135, 0.305, 0.80, 0.435], facecolor=BG)
    curve(ax, a, b, t, label_size=10)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.205, 0.205], color=GRID, lw=1.2))
    for i, (big, small) in enumerate(t["stats"]):
        x = 0.085 + i * 0.222
        fig.text(x, 0.145, big, color=FG, fontsize=13.5, fontweight="bold")
        fig.text(x, 0.105, small, color=MUTED, fontsize=8.2)
    fig.text(0.07, 0.038, t["foot"], color=MUTED, fontsize=7.1)
    p = os.path.join(OUT, f"proof_loss_{lang}.png")
    fig.savefig(p, facecolor=BG)
    plt.close(fig)
    return p


def wide():
    t = TXT["en"]
    a, b, _ = load()
    fig = plt.figure(figsize=(11, 5.2), dpi=180, facecolor=BG)
    fig.text(0.045, 0.945, "Out-of-core LoRA on Kimi K3 (2.78T): the loss falls, and nothing fits in RAM",
             color=FG, fontsize=15, fontweight="bold", va="top")
    ax1 = fig.add_axes([0.075, 0.185, 0.35, 0.60], facecolor=BG)
    curve(ax1, a, b, t, label_size=9.5)
    ax1.set_title("proof run: the same sequence, pass after pass", color=MUTED,
                  fontsize=10.5, pad=26, loc="left")

    ax2 = fig.add_axes([0.645, 0.185, 0.325, 0.60], facecolor=BG)
    bars = [("checkpoint (USB disk)", 1560.0, "#3a4152"),
            ("shared weights (NVMe)", 108.8, "#3a4152"),
            ("machine RAM", 7.6, CB),
            ("LoRA adapter, trained", 0.59, CA)]
    ys = range(len(bars))
    ax2.barh(list(ys), [v for _, v, _ in bars], color=[c for *_, c in bars], height=0.62)
    ax2.set_yticks(list(ys), [n for n, *_ in bars], color=MUTED, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlim(0.3, 9000)
    for i, (_, v, _c) in enumerate(bars):
        ax2.text(v * 1.25, i, f"{v:g} GB", va="center", color=FG, fontsize=9.5, fontweight="bold")
    ax2.grid(True, axis="x", color=GRID, lw=1)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.set_title("what has to move past 7.6 GB of RAM", color=MUTED, fontsize=10.5, pad=26, loc="left")
    fig.text(0.045, 0.03, t["foot"], color=MUTED, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"proof_loss_wide.{ext}"), facecolor=BG)
    plt.close(fig)



def social():
    """1280x640 card for the GitHub social preview (Settings -> Social preview)."""
    t = TXT["en"]
    a, b, _ = load()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
    fig.text(0.055, 0.90, "LazyLoRA", color=FG, fontsize=40, fontweight="bold", va="top")
    fig.text(0.055, 0.735, "Out-of-core LoRA fine-tuning on a\n2.78-trillion-parameter MoE, on one laptop",
             color=FG, fontsize=19, va="top", linespacing=1.45)
    fig.text(0.055, 0.50, "Kimi K3 · 93 layers · 896 experts per layer\n1.56 TB checkpoint on a USB disk · 7.6 GB RAM\nthe 2.78 T weights stay frozen; a 590 MB adapter trains\nforward checked against an independent C engine\ngradients checked by finite differences",
             color=MUTED, fontsize=13, va="top", linespacing=1.75)
    fig.text(0.055, 0.085, "github.com/heyobi/LazyLora", color=CA, fontsize=13)
    ax = fig.add_axes([0.55, 0.20, 0.40, 0.60], facecolor=BG)
    curve(ax, a, b, t, label_size=10)
    ax.set_title("proof run: loss on the same sequence, pass after pass",
                 color=MUTED, fontsize=11, pad=24, loc="left")
    fig.savefig(os.path.join(OUT, "social_preview.png"), facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    print(card("tr")); print(card("en")); wide(); social()
    print(os.path.join(OUT, "proof_loss_wide.png")); print(os.path.join(OUT, "social_preview.png"))
