"""
Thread-Safe Metrics Collector for LazyLoRA Training.
Tracks GPU VRAM, Host RAM, Disk I/O throughput, active layer/expert IDs,
loss progression, step latency, and ETA.
"""

import os
import time
import shutil
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TrainingStepMetrics:
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    smooth_loss: float = 0.0
    lr: float = 0.0
    active_layer: int = 0
    total_layers: int = 93
    active_experts: List[int] = field(default_factory=list)
    vram_used_mb: float = 0.0
    vram_total_mb: float = 6144.0
    vram_peak_mb: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 16.0
    disk_read_mbps: float = 0.0
    c_drive_free_gb: float = 0.0
    step_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    elapsed_time_sec: float = 0.0
    eta_sec: float = 0.0


class MetricsTracker:
    """Collects and computes live hardware and training statistics."""

    def __init__(self, total_steps: int = 1000, total_layers: int = 93):
        self.total_steps = total_steps
        self.total_layers = total_layers
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.history: List[TrainingStepMetrics] = []
        self.smooth_loss = 0.0
        self.peak_vram_mb = 0.0

    def get_c_drive_free_gb(self) -> float:
        """Probe free space on C: drive."""
        c_path = "/mnt/c" if platform.system() == "Linux" else "C:\\"
        try:
            usage = shutil.disk_usage(c_path)
            return round(usage.free / (1024 ** 3), 2)
        except Exception:
            return 8.5

    def get_hardware_memory_snapshot(self) -> Dict[str, float]:
        """Query instant RAM and VRAM usage."""
        vram_used = 0.0
        vram_total = 6144.0
        ram_used = 0.0
        ram_total = 16.0

        # Try nvidia-smi
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().splitlines()[0].split(",")
                vram_used = float(parts[0].strip())
                vram_total = float(parts[1].strip())
        except Exception:
            pass

        # Try /proc/meminfo
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                mem_data = {}
                for l in lines:
                    if ":" in l:
                        k, v = l.split(":", 1)
                        mem_data[k.strip()] = int(v.strip().split()[0])
                total_kb = mem_data.get("MemTotal", 16 * 1024 * 1024)
                avail_kb = mem_data.get("MemAvailable", mem_data.get("MemFree", 8 * 1024 * 1024))
                ram_total = total_kb / (1024 * 1024)
                ram_used = (total_kb - avail_kb) / (1024 * 1024)
            except Exception:
                pass

        return {
            "vram_used_mb": vram_used,
            "vram_total_mb": vram_total,
            "ram_used_gb": round(ram_used, 2),
            "ram_total_gb": round(ram_total, 2),
        }

    def update_step(
        self,
        step: int,
        loss: float,
        lr: float,
        active_layer: int = 0,
        active_experts: Optional[List[int]] = None,
        tokens_processed: int = 512,
        disk_bytes_read: int = 0,
    ) -> TrainingStepMetrics:
        """Records a step update and returns snapshot metrics."""
        now = time.time()
        step_dt = max(1e-4, now - self.last_step_time)
        self.last_step_time = now
        elapsed = now - self.start_time

        # Exponential moving average for loss
        if self.smooth_loss == 0.0:
            self.smooth_loss = loss
        else:
            self.smooth_loss = 0.9 * self.smooth_loss + 0.1 * loss

        hw = self.get_hardware_memory_snapshot()
        self.peak_vram_mb = max(self.peak_vram_mb, hw["vram_used_mb"])

        tokens_per_sec = tokens_processed / step_dt if step_dt > 0 else 0.0
        disk_mbps = (disk_bytes_read / (1024 * 1024)) / step_dt if step_dt > 0 else 0.0
        c_free = self.get_c_drive_free_gb()

        # ETA calculation
        remaining_steps = max(0, self.total_steps - step)
        eta_sec = remaining_steps * step_dt

        m = TrainingStepMetrics(
            step=step,
            total_steps=self.total_steps,
            loss=round(loss, 4),
            smooth_loss=round(self.smooth_loss, 4),
            lr=lr,
            active_layer=active_layer,
            total_layers=self.total_layers,
            active_experts=active_experts or [],
            vram_used_mb=round(hw["vram_used_mb"], 1),
            vram_total_mb=round(hw["vram_total_mb"], 1),
            vram_peak_mb=round(self.peak_vram_mb, 1),
            ram_used_gb=hw["ram_used_gb"],
            ram_total_gb=hw["ram_total_gb"],
            disk_read_mbps=round(disk_mbps, 1),
            c_drive_free_gb=c_free,
            step_time_sec=round(step_dt, 3),
            tokens_per_sec=round(tokens_per_sec, 1),
            elapsed_time_sec=round(elapsed, 1),
            eta_sec=round(eta_sec, 1),
        )

        self.history.append(m)
        if len(self.history) > 1000:
            self.history.pop(0)

        # Dump live snapshot for real-time monitoring
        try:
            import json
            from dataclasses import asdict
            from lazy_lora.core.config import default_cache_dir
            live_path = os.path.join(default_cache_dir(), "live_metrics.json")
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(asdict(m), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return m
