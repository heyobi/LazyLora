#!/usr/bin/env python3
"""
One-screen status of the measurement queue: which trace is running, how far it is, the
per-layer pace, ETA for the rest of the queue, disk/RAM, and the git state.

    python scripts/status.py
"""
import glob, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402

QUEUE = ["zh_paragraph", "code_python", "tr_news", "en_paragraph", "tr_paragraph"]
TOKENS = {"zh_paragraph": 111, "code_python": 167, "tr_news": 261, "en_paragraph": 159, "tr_paragraph": 264}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main():
    cfg = get_default_config()
    W = cfg.paths.workspace_dir
    now = time.time()
    print(f"== LazyLoRA status  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    running = sh("pgrep -af 'measure_routing|verify_backward|train_lazy|lazy_trainer' | grep -v 'bash -c' | cut -c1-140")
    print("running :", running or "nothing")
    print()
    pace = {}
    print(f"{'trace':26s} {'layers':>8s} {'pace s/layer':>13s} {'read GB':>8s} {'RSS GB':>7s} {'status'}")
    for tag in QUEUE:
        dirs = sorted(glob.glob(os.path.join(W, "traces", f"{tag}_L93_*")))
        if not dirs:
            print(f"{tag:26s} {'-':>8s} {'':>13s} {'':>8s} {'':>7s} queued")
            continue
        m = json.load(open(os.path.join(dirs[-1], "trace.json")))
        L = m["layers"]
        secs = [e["seconds"] for e in L[-10:]] or [0]
        p = sum(secs) / len(secs)
        pace[tag] = p
        done = m.get("total_seconds") is not None
        st = f"done in {m['total_seconds'] / 3600:.1f} h" if done else f"running, ETA {(92 - len(L)) * p / 60:.0f} min"
        print(f"{tag:26s} {len(L):>4d}/92  {p:>13.0f} {sum(e['bytes_read'] for e in L) / 1e9:>8.0f} {m.get('peak_rss_gb', 0) or '-':>7} {st}")
    # rough ETA for the rest
    ref = pace.get("zh_paragraph") or 55.0
    per_tok = ref / TOKENS["zh_paragraph"] * 0.45 + ref * 0.55 / TOKENS["zh_paragraph"]  # crude: cost ~ sweep, mild in tokens
    remaining = 0.0
    for tag in QUEUE:
        dirs = sorted(glob.glob(os.path.join(W, "traces", f"{tag}_L93_*")))
        if dirs and json.load(open(os.path.join(dirs[-1], "trace.json"))).get("total_seconds"):
            continue
        done_layers = len(json.load(open(os.path.join(dirs[-1], "trace.json")))["layers"]) if dirs else 0
        est_pace = ref * (0.6 + 0.4 * TOKENS[tag] / TOKENS["zh_paragraph"])
        remaining += (92 - done_layers) * est_pace
    print(f"\nqueue ETA ~ {remaining / 3600:.1f} h, then clean 1024/2048-token profiles (~1 h)")
    print()
    print("disk    :", sh("df -h /mnt/disk2tb /mnt/nvme | tail -2 | awk '{print $6, $4, \"free\"}' | tr '\\n' ' '"))
    print("ram     :", sh("free -g | awk 'NR==2{print $7\" GB available of \"$2}'"))
    print("git     :", sh("cd /home/ibox/calisma/LazyLora && git log --oneline -1 && git status --short | wc -l | xargs -I{} echo '{} uncommitted files'"))
    err = sh("sudo -n dmesg 2>/dev/null | grep -cE 'I/O error, dev sd|failed to read volume'")
    print("disk errs in kernel log:", err or "n/a")


if __name__ == "__main__":
    main()
