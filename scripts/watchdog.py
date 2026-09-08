#!/usr/bin/env python3
"""
Unattended-run watchdog. Meant to be run every 15 minutes by a systemd user timer.

Reads <workspace>/run_manifest.json, which describes the run that should be alive:
  {"name": "...", "args": ["--data", "...", "--steps", "16", ...], "steps": 16,
   "expected_step_seconds": 18000, "python": "/home/ibox/venvs/lazylora-cu/bin/python",
   "env": {"LAZYLORA_GPU": "1"}, "active": true}

Each tick it
  - checks whether the trainer process is alive; if not and steps remain, resumes it from
    the newest checkpoint (at most 3 automatic restarts in a row, 30 min apart);
  - reads forward_loss.jsonl for progress and detects stalls (no new step in 3x the
    expected step time);
  - checks the HDD mount, kernel disk/USB errors, NVMe free space, swap and GPU temperature;
  - writes <workspace>/status.json and status.txt (what `status.sh` shows);
  - sends a push notification to the phone (claude-code-server's ccs_push) on every
    completed step and on any problem, and once on completion.
State between ticks lives in <workspace>/watchdog_state.json.
"""
import json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402

CCS_BIN = os.path.expanduser("~/.local/share/claude-code-server/bin")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def push(title, body):
    try:
        sys.path.insert(0, CCS_BIN)
        import ccs_push  # type: ignore
        return ccs_push.send_all({"title": title, "body": body[:180], "session": "ziverbey-01"})
    except Exception as exc:
        return f"push unavailable: {exc}"


def load_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def main():
    cfg = get_default_config()
    W = cfg.paths.workspace_dir
    man = load_json(os.path.join(W, "run_manifest.json"), None)
    state = load_json(os.path.join(W, "watchdog_state.json"), {"restarts": 0, "last_restart": 0,
                                                              "last_step_seen": 0, "alerted": {}})
    now = time.time()
    alerts, info = [], {}

    # --- health
    hdd_ok = os.path.isdir(os.path.join(cfg.paths.base_model_dir)) and os.path.exists(
        os.path.join(cfg.paths.base_model_dir, "config.json"))
    disk_err = int(sh("sudo -n dmesg 2>/dev/null | grep -cE 'I/O error, dev sd|USB disconnect'") or 0)
    nvme_free = int(sh("df --output=avail -B1 /mnt/nvme | tail -1") or 0) / 1e9
    swap_used = int(sh("free -b | awk 'NR==3{print $3}'") or 0) / 1e9
    gpu_temp = int(sh("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader") or 0)
    info.update(hdd_ok=hdd_ok, disk_errors=disk_err, nvme_free_gb=round(nvme_free, 1),
                swap_used_gb=round(swap_used, 2), gpu_temp=gpu_temp)
    if not hdd_ok:
        alerts.append("HDD okunamıyor (model dizini yok)")
    if disk_err > state.get("disk_err_seen", disk_err):
        alerts.append(f"Yeni disk/USB hatası (+{disk_err - state.get('disk_err_seen', disk_err)})")
    state["disk_err_seen"] = disk_err
    if nvme_free < 3:
        alerts.append(f"NVMe boş alan {nvme_free:.1f} GB")
    if gpu_temp >= 85:
        alerts.append(f"GPU {gpu_temp} °C")
    if swap_used > 2:
        alerts.append(f"Swap {swap_used:.1f} GB")

    # --- progress
    steps = []
    try:
        for line in open(os.path.join(W, "forward_loss.jsonl")):
            try:
                steps.append(json.loads(line))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    run_start = man.get("started", 0) if man else 0
    steps = [s for s in steps if s.get("time", 0) >= run_start]
    last = steps[-1] if steps else None
    alive = bool(sh("pgrep -f 'lazy_lora.trainer.lazy_trainer' | grep -v $$"))
    info.update(alive=alive, steps_done=len(steps), last_step=last)

    if man and man.get("active"):
        total = int(man.get("steps", 0))
        exp = float(man.get("expected_step_seconds", 18000))
        if last and last["step"] > state.get("last_step_seen", 0):
            state["last_step_seen"] = last["step"]
            state["last_step_time"] = last["time"]
            push(f"LazyLoRA adım {last['step']}/{total}", f"loss {last['loss']:.4f}  ppl {last.get('perplexity', 0):.1f}")
        done = last is not None and last["step"] >= total
        if done and not state["alerted"].get("done"):
            push("LazyLoRA koşu tamamlandı", f"{man['name']}: {total} adım, son loss {last['loss']:.4f}")
            state["alerted"]["done"] = True
        if alive:
            ref = state.get("last_step_time") or run_start or now
            if now - ref > 3 * exp:
                alerts.append(f"Eğitim {(now - ref) / 3600:.1f} saattir adım atmadı (beklenen {exp / 3600:.1f} s)")
        elif not done:
            # crashed or was killed: resume from the newest checkpoint
            ck_dir = cfg.paths.checkpoints_dir
            latest = None
            try:
                latest = open(os.path.join(ck_dir, "latest.txt")).read().strip()
            except Exception:
                pass
            can = state["restarts"] < 3 and now - state.get("last_restart", 0) > 1800
            if latest and os.path.exists(os.path.join(ck_dir, latest)) and can:
                env = dict(os.environ, PYTHONPATH=REPO, LAZYLORA_PYTHON=man.get("python", ""), **man.get("env", {}))
                cmd = ["bash", os.path.join(REPO, "scripts", "train_lazy_lora.sh"),
                       "--resume", os.path.join(ck_dir, latest)] + list(man.get("args", []))
                log = open(os.path.join(W, f"{man['name']}_resume_{int(now)}.log"), "w")
                subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
                state["restarts"] += 1
                state["last_restart"] = now
                push("LazyLoRA yeniden başlatıldı", f"{latest} üzerinden devam ({state['restarts']}. kez)")
                info["resumed_from"] = latest
            elif not can:
                alerts.append("Eğitim ölü ve otomatik yeniden başlatma sınırı doldu")
            else:
                alerts.append("Eğitim ölü ve checkpoint yok")
        if alive and state["restarts"] and now - state.get("last_restart", 0) > 6 * 3600:
            state["restarts"] = 0     # a run that survived 6 h resets the restart budget

    for a in alerts:
        if not state["alerted"].get(a) or now - state["alerted"][a] > 6 * 3600:
            push("LazyLoRA uyarı", a)
            state["alerted"][a] = now

    status = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": man["name"] if man else None,
              "alerts": alerts, **info}
    json.dump(status, open(os.path.join(W, "status.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(W, "status.txt"), "w") as f:
        f.write(f"{status['time']}  run={status['run']}  alive={alive}  steps={len(steps)}"
                f"{'/' + str(man['steps']) if man else ''}\n")
        if last:
            f.write(f"last step {last['step']}: loss {last['loss']:.4f}  at {time.strftime('%H:%M', time.localtime(last['time']))}\n")
        f.write(f"hdd_ok={hdd_ok} disk_errors={disk_err} nvme_free={nvme_free:.1f}GB swap={swap_used:.1f}GB gpu={gpu_temp}C\n")
        for a in alerts:
            f.write(f"ALERT: {a}\n")
    json.dump(state, open(os.path.join(W, "watchdog_state.json"), "w"))
    print(open(os.path.join(W, "status.txt")).read())


if __name__ == "__main__":
    main()
