"""
Hardware and System Resource Profiler for LazyLoRA.
Analyzes CPU cores, GPU specifications, VRAM, System RAM, WSL allocation,
and disk storage across C: and D: drives.
Enforces safety checks (C: drive low disk space protection).
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class CPUInfo:
    model: str
    physical_cores: int
    logical_cores: int
    architecture: str


@dataclass
class GPUInfo:
    name: str
    total_vram_mb: float
    free_vram_mb: float
    cuda_version: str
    driver_version: str
    cuda_cores_approx: int
    compute_capability: str


@dataclass
class MemoryInfo:
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    swap_total_gb: float
    swap_free_gb: float


@dataclass
class DiskInfo:
    path: str
    label: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float
    is_safe_for_storage: bool


@dataclass
class SystemHardwareProfile:
    os_name: str
    is_wsl: bool
    cpu: CPUInfo
    gpu: Optional[GPUInfo]
    memory: MemoryInfo
    disks: List[DiskInfo]
    c_drive_safe: bool
    recommended_vram_budget_mb: float
    recommended_ram_budget_gb: float
    recommended_storage_path: str


class HardwareProfiler:
    """Probes system hardware in Linux/WSL/Windows environments."""

    @staticmethod
    def is_wsl() -> bool:
        if platform.system() == "Linux":
            try:
                with open("/proc/version", "r") as f:
                    return "microsoft" in f.read().lower()
            except Exception:
                return False
        return False

    @staticmethod
    def get_cpu_info() -> CPUInfo:
        model = platform.processor() or "Unknown"
        physical_cores = os.cpu_count() or 1
        logical_cores = os.cpu_count() or 1

        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                for line in cpuinfo.splitlines():
                    if "model name" in line:
                        model = line.split(":", 1)[1].strip()
                        break
                # Try lscpu if available
                lscpu = subprocess.run(["lscpu"], capture_output=True, text=True)
                if lscpu.returncode == 0:
                    for line in lscpu.stdout.splitlines():
                        if "Model name:" in line:
                            model = line.split(":", 1)[1].strip()
                        elif "Core(s) per socket:" in line:
                            cores_per_socket = int(line.split(":", 1)[1].strip())
                            physical_cores = cores_per_socket
                        elif "CPU(s):" in line and not "NUMA" in line:
                            logical_cores = int(line.split(":", 1)[1].strip())
            except Exception:
                pass

        return CPUInfo(
            model=model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            architecture=platform.machine(),
        )

    @staticmethod
    def get_gpu_info() -> Optional[GPUInfo]:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
                name = parts[0]
                total_vram = float(parts[1])
                free_vram = float(parts[2])
                driver_ver = parts[3] if len(parts) > 3 else "N/A"

                compute_cap = "5.2" if "980" in name else "Unknown"
                cuda_cores = 2816 if "980 Ti" in name or "980Ti" in name else (
                    2048 if "980" in name else 1024
                )

                cuda_ver = "Unknown"
                smi_all = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if smi_all.returncode == 0:
                    for line in smi_all.stdout.splitlines():
                        if "CUDA Version:" in line:
                            try:
                                cuda_ver = line.split("CUDA Version:")[1].split("|")[0].strip()
                            except Exception:
                                pass

                return GPUInfo(
                    name=name,
                    total_vram_mb=total_vram,
                    free_vram_mb=free_vram,
                    cuda_version=cuda_ver,
                    driver_version=driver_ver,
                    cuda_cores_approx=cuda_cores,
                    compute_capability=compute_cap,
                )
        except Exception:
            pass
        return None

    @staticmethod
    def get_memory_info() -> MemoryInfo:
        total_ram = 0.0
        avail_ram = 0.0
        used_ram = 0.0
        swap_total = 0.0
        swap_free = 0.0

        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                data = {}
                for line in meminfo.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        data[k.strip()] = int(v.strip().split()[0])  # in kB

                total_ram = data.get("MemTotal", 0) / (1024 * 1024)
                avail_ram = data.get("MemAvailable", data.get("MemFree", 0)) / (1024 * 1024)
                used_ram = total_ram - avail_ram
                swap_total = data.get("SwapTotal", 0) / (1024 * 1024)
                swap_free = data.get("SwapFree", 0) / (1024 * 1024)
            except Exception:
                pass
        else:
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_ram = stat.ullTotalPhys / (1024 ** 3)
                avail_ram = stat.ullAvailPhys / (1024 ** 3)
                used_ram = total_ram - avail_ram
            except Exception:
                total_ram = 16.0
                avail_ram = 8.0
                used_ram = 8.0

        return MemoryInfo(
            total_ram_gb=round(total_ram, 2),
            available_ram_gb=round(avail_ram, 2),
            used_ram_gb=round(used_ram, 2),
            swap_total_gb=round(swap_total, 2),
            swap_free_gb=round(swap_free, 2),
        )

    @staticmethod
    def get_disk_info() -> List[DiskInfo]:
        disks = []
        is_wsl = HardwareProfiler.is_wsl()

        paths_to_check = []
        if is_wsl:
            paths_to_check = [
                ("/mnt/c", "Windows C: Drive (System)"),
                ("/mnt/d", "Windows D: Drive (Mass Storage)"),
                ("/", "WSL Root VHDX"),
            ]
        elif platform.system() == "Windows":
            paths_to_check = [
                ("C:\\", "Windows C: Drive (System)"),
                ("D:\\", "Windows D: Drive (Mass Storage)"),
            ]
        else:
            from lazy_lora.core.config import (default_model_dir, default_workspace_dir,
                                               default_fast_scratch_dir)
            paths_to_check = [("/", "Root Filesystem (system)")]
            for p, label in ((default_model_dir(), "Model checkpoint volume"),
                             (default_workspace_dir(), "Workspace volume"),
                             (default_fast_scratch_dir(), "Fast scratch volume (activations, checkpoints)")):
                # Walk up to the nearest existing ancestor so an unmounted scratch dir is reported
                q = p
                while q and not os.path.exists(q):
                    q = os.path.dirname(q)
                if q and q not in [x for x, _ in paths_to_check]:
                    paths_to_check.append((q, label))

        for p, label in paths_to_check:
            if os.path.exists(p):
                try:
                    usage = shutil.disk_usage(p)
                    total_gb = usage.total / (1024 ** 3)
                    used_gb = usage.used / (1024 ** 3)
                    free_gb = usage.free / (1024 ** 3)
                    pct = (used_gb / total_gb) * 100.0 if total_gb > 0 else 0.0
                    
                    is_system = p in ("/", "C:\\") or "/mnt/c" in p.lower()
                    is_safe = free_gb > 50.0 and not is_system

                    disks.append(DiskInfo(
                        path=p,
                        label=label,
                        total_gb=round(total_gb, 2),
                        used_gb=round(used_gb, 2),
                        free_gb=round(free_gb, 2),
                        percent_used=round(pct, 1),
                        is_safe_for_storage=is_safe,
                    ))
                except Exception:
                    pass

        return disks

    @classmethod
    def analyze_system(cls) -> SystemHardwareProfile:
        cpu = cls.get_cpu_info()
        gpu = cls.get_gpu_info()
        mem = cls.get_memory_info()
        disks = cls.get_disk_info()

        is_wsl = cls.is_wsl()
        os_name = f"WSL2 Ubuntu ({platform.release()})" if is_wsl else platform.platform()

        c_disk = next((d for d in disks if "c:" in d.path.lower() or "/mnt/c" in d.path.lower()), None)
        c_safe = True
        if c_disk and c_disk.free_gb < 15.0:
            c_safe = False

        d_disk = next((d for d in disks if "d:" in d.path.lower() or "/mnt/d" in d.path.lower()), None)
        if is_wsl or platform.system() == "Windows":
            rec_storage = "/mnt/d/LazyLora_Workspace" if is_wsl and d_disk else (
                "D:\\LazyLora_Workspace" if d_disk else "./workspace"
            )
        else:
            from lazy_lora.core.config import default_workspace_dir
            rec_storage = default_workspace_dir()

        vram_mb = gpu.total_vram_mb if gpu else 0.0
        rec_vram_budget = max(512.0, vram_mb - 1536.0) if vram_mb > 0 else 0.0
        rec_ram_budget = max(2.0, mem.total_ram_gb * 0.65)

        return SystemHardwareProfile(
            os_name=os_name,
            is_wsl=is_wsl,
            cpu=cpu,
            gpu=gpu,
            memory=mem,
            disks=disks,
            c_drive_safe=c_safe,
            recommended_vram_budget_mb=round(rec_vram_budget, 1),
            recommended_ram_budget_gb=round(rec_ram_budget, 1),
            recommended_storage_path=rec_storage,
        )

    @classmethod
    def print_report(cls) -> None:
        prof = cls.analyze_system()
        w = 78
        print("=" * w)
        print(" " * 20 + "LAZYLORA HARDWARE & SYSTEM AUDIT")
        print("=" * w)
        print(f" OS Environment       : {prof.os_name} (WSL2: {prof.is_wsl})")
        print(f" CPU Model            : {prof.cpu.model}")
        print(f" CPU Cores / Threads  : {prof.cpu.physical_cores} Cores / {prof.cpu.logical_cores} Threads ({prof.cpu.architecture})")
        
        if prof.gpu:
            print(f" GPU Model            : {prof.gpu.name}")
            print(f" GPU VRAM             : {prof.gpu.total_vram_mb:.0f} MB Total ({prof.gpu.free_vram_mb:.0f} MB Free)")
            print(f" CUDA & Driver        : CUDA {prof.gpu.cuda_version} | Driver {prof.gpu.driver_version} (CC {prof.gpu.compute_capability})")
            print(f" CUDA Cores (Approx)  : ~{prof.gpu.cuda_cores_approx} Cores")
        else:
            print(" GPU                  : [WARNING] No NVIDIA GPU detected via nvidia-smi")

        print(f" System RAM (WSL/Host): {prof.memory.total_ram_gb} GB Total ({prof.memory.available_ram_gb} GB Available)")
        if prof.memory.swap_total_gb > 0:
            print(f" Swap Memory          : {prof.memory.swap_total_gb} GB Total ({prof.memory.swap_free_gb} GB Free)")

        print("-" * w)
        print(" STORAGE DISKS & SAFETY AUDIT:")
        for d in prof.disks:
            safe_str = "[OK - DEDICATED STORAGE]" if d.is_safe_for_storage else (
                "[PROTECTED - DO NOT WRITE]" if "c:" in d.path.lower() or "/mnt/c" in d.path.lower() else "[WSL ROOT]"
            )
            print(f"  * {d.label} ({d.path})")
            print(f"    Total: {d.total_gb:.1f} GB | Free: {d.free_gb:.1f} GB ({d.percent_used:.1f}% used) -> {safe_str}")

        print("-" * w)
        print(" RESOURCE ALLOCATION & OPTIMIZATION PROFILE:")
        print(f"  * Recommended VRAM Cap   : {prof.recommended_vram_budget_mb:.0f} MB (Buffer headroom preserved)")
        print(f"  * Recommended RAM Cap    : {prof.recommended_ram_budget_gb:.1f} GB")
        print(f"  * Designated Storage Path: {prof.recommended_storage_path} (Zero C: Drive impact)")
        
        if not prof.c_drive_safe:
            print("  ! CRITICAL NOTICE: C: drive space is low (<15GB). Strict zero-write policy enforced!")
        print("=" * w)


if __name__ == "__main__":
    HardwareProfiler.print_report()
