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
  - optionally sends a push notification (claude-code-server's ccs_push, if that is
    installed; set LAZYLORA_PUSH_SESSION to the session it should address) on every
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
        session = os.environ.get("LAZYLORA_PUSH_SESSION", "")
        return ccs_push.send_all({"title": title, "body": body[:180], "session": session})
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
    hdd_temp = int(sh("sudo -n smartctl -d sat -A /dev/$(lsblk -no PKNAME $(findmnt -no SOURCE /mnt/disk2tb 2>/dev/null) 2>/dev/null) 2>/dev/null | awk '/Temperature_Celsius/{print $10}'") or 0)
    usb_resets = int(sh("sudo -n dmesg 2>/dev/null | grep -c 'usb 2-2: reset'") or 0)
    info.update(hdd_temp=hdd_temp, usb_resets=usb_resets)
    if hdd_temp >= 58:
        alerts.append(f"HDD {hdd_temp} °C")
    # SMART wear counters: any increase is an alert, because the run reads about 3 TB per step
    # from a consumer disk for a month, which is far beyond its rated workload
    smart = sh("sudo -n smartctl -d sat -A /dev/$(lsblk -no PKNAME $(findmnt -no SOURCE /mnt/disk2tb 2>/dev/null) 2>/dev/null) 2>/dev/null | "
               "awk '/Reallocated_Sector_Ct|Current_Pending_Sector|Offline_Uncorrectable|UDMA_CRC_Error_Count/{print $2\"=\"$10}'")
    smart_now = {}
    for kv in smart.split():
        k, _, v = kv.partition("=")
        if v.isdigit():
            smart_now[k] = int(v)
    info["smart"] = smart_now
    smart_seen = state.get("smart_seen", {})
    for k, v in smart_now.items():
        if k in smart_seen and v > smart_seen[k]:
            alerts.append(f"SMART {k}: {smart_seen[k]} -> {v}")
    state["smart_seen"] = smart_now
    if usb_resets > state.get("usb_resets_seen", usb_resets) + 20:
        alerts.append(f"USB köprüsü sıfırlanıyor (+{usb_resets - state.get('usb_resets_seen', usb_resets)})")
    state["usb_resets_seen"] = usb_resets
    if not hdd_ok:
        alerts.append("HDD okunamıyor (model dizini yok)")
        # the enclosure dropped off the bus: try the reconnect procedure once per tick
        out = sh("sudo -n bash " + os.path.join(REPO, "scripts", "hdd_reconnect.sh") + " 2>&1 | tail -3", timeout=300)
        info["reconnect"] = out[-200:]
        hdd_ok = os.path.exists(os.path.join(cfg.paths.base_model_dir, "config.json"))
        if hdd_ok:
            push("LazyLoRA disk geri geldi", "USB disk yeniden bağlandı; koşu checkpoint'ten devam edecek")
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

    # copy every new checkpoint off the NVMe onto the root SSD, keeping the two newest; the
    # checkpoint is weeks of compute and the only artefact that cannot be re-downloaded
    try:
        import shutil, glob
        bdir = os.path.expanduser(os.environ.get("LAZYLORA_CKPT_BACKUP_DIR", "~/lazylora_ckpt_backup"))
        os.makedirs(bdir, exist_ok=True)
        cks = sorted(glob.glob(os.path.join(cfg.paths.checkpoints_dir, "lazy_lora_step_*.pt")), key=os.path.getmtime)
        for ck in cks[-2:]:
            dst = os.path.join(bdir, os.path.basename(ck))
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(ck):
                shutil.copy2(ck, dst + ".tmp"); os.replace(dst + ".tmp", dst)
                info["backed_up"] = os.path.basename(ck)
        for old_bk in sorted(glob.glob(os.path.join(bdir, "lazy_lora_step_*.pt")), key=os.path.getmtime)[:-2]:
            os.remove(old_bk)
    except Exception as exc:
        alerts.append(f"checkpoint yedeği alınamadı: {exc}")

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
        f.write(f"hdd_ok={hdd_ok} hdd={hdd_temp}C usb_resets={usb_resets} disk_errors={disk_err} nvme_free={nvme_free:.1f}GB swap={swap_used:.1f}GB gpu={gpu_temp}C smart={smart_now}\n")
        for a in alerts:
            f.write(f"ALERT: {a}\n")
    json.dump(state, open(os.path.join(W, "watchdog_state.json"), "w"))
    print(open(os.path.join(W, "status.txt")).read())


if __name__ == "__main__":
    main()
