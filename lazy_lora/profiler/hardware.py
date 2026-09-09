"""
Hardware and system-resource profiler for LazyLoRA.

Reports the CPU, the NVIDIA GPU (when nvidia-smi answers), RAM and swap, and the free
space on every volume the engine actually uses: the system volume, the checkpoint volume,
the workspace volume and the fast scratch volume. The volume paths are not written here;
they come from lazy_lora.core.config, which reads LAZYLORA_MODEL_DIR,
LAZYLORA_WORKSPACE_DIR and LAZYLORA_FAST_SCRATCH_DIR from the environment.

The project began on a Windows/WSL machine and this file used to assume it: it named a
"C: drive" and a "D: drive" whether or not they existed, and it guessed CUDA core counts
from a short list of GeForce names. It now reports what an ordinary Linux machine really
has; WSL and native Windows are still supported, but as cases that must be *detected*
rather than as the default.

The one rule that survives from the WSL days, and the reason this file exists at all:
a volume is never recommended for model-scale storage if it is the system volume, or if
it has less than 50 GB free. A 1.56 TB checkpoint on the root filesystem fills it and
takes the machine down with it.

    python -m lazy_lora.profiler.hardware
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Free space below which a volume is not worth offering for checkpoint-scale data.
MIN_FREE_GB_FOR_STORAGE = 50.0
# Free space below which the system volume itself is in danger.
SYSTEM_VOLUME_WARN_GB = 15.0

# CUDA cores per streaming multiprocessor, keyed by compute capability. This is a property
# of the architecture, not of a particular card, which is why it can be tabulated at all;
# the previous version of this file guessed the *total* from the marketing name and was
# wrong on every GPU that was not in its list.
_CORES_PER_SM = {
    (2, 0): 32, (2, 1): 48,
    (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128,
    (10, 0): 128, (10, 1): 128,
    (12, 0): 128,
}


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
    cuda_cores_approx: int          # 0 when the SM count was not probed; never guessed
    compute_capability: str         # "unknown" when the driver does not report it


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
    is_system_volume: bool
    is_safe_for_storage: bool


@dataclass
class SystemHardwareProfile:
    os_name: str
    is_wsl: bool
    cpu: CPUInfo
    gpu: Optional[GPUInfo]
    memory: MemoryInfo
    disks: List[DiskInfo]
    system_volume_safe: bool
    recommended_vram_budget_mb: float
    recommended_ram_budget_gb: float
    recommended_storage_path: str

    @property
    def c_drive_safe(self) -> bool:
        """Old name for system_volume_safe, kept so existing callers do not break."""
        return self.system_volume_safe


class HardwareProfiler:
    """Probes system hardware on Linux (native or WSL) and on Windows."""

    # ------------------------------------------------------------------ environment

    @staticmethod
    def is_wsl() -> bool:
        """True only on a real WSL kernel. On native Linux this is False and stays False."""
        if platform.system() != "Linux":
            return False
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except Exception:
            return False

    @staticmethod
    def _distro_name() -> str:
        """PRETTY_NAME from /etc/os-release, or "" when there is none."""
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------------- cpu

    @staticmethod
    def get_cpu_info() -> CPUInfo:
        model = platform.processor() or "Unknown"
        physical_cores = os.cpu_count() or 1
        logical_cores = os.cpu_count() or 1

        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            model = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass

            # lscpu is read into an exact key -> value map, and asked for it in the C
            # locale. Both matter: the old code matched the substring "CPU(s):", which also
            # appears in "NUMA node0 CPU(s):" (it special-cased that one) and would match any
            # other such key a future lscpu adds; and on a machine whose lscpu is translated,
            # English substrings match nothing at all and the count silently stays at the
            # os.cpu_count() fallback.
            fields = {}
            try:
                res = subprocess.run(
                    ["lscpu"], capture_output=True, text=True,
                    env=dict(os.environ, LC_ALL="C", LANG="C"),
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if ":" in line:
                            key, val = line.split(":", 1)
                            fields[key.strip()] = val.strip()
            except Exception:
                pass

            def _field_int(key: str) -> int:
                """The integer at `key`, or 0 when absent or not a plain number."""
                try:
                    return int(fields[key])
                except (KeyError, ValueError):
                    return 0

            model = fields.get("Model name") or model

            # Physical cores are cores-per-socket x sockets. Reading only "Core(s) per
            # socket", as this file used to, reports half the machine on a two-socket box.
            per_socket = _field_int("Core(s) per socket")
            if per_socket > 0:
                physical_cores = per_socket * (_field_int("Socket(s)") or 1)
            logical = _field_int("CPU(s)")
            if logical > 0:
                logical_cores = logical

        return CPUInfo(
            model=model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            architecture=platform.machine(),
        )

    # ------------------------------------------------------------------------- gpu

    @staticmethod
    def _query_smi(fields: str) -> Optional[str]:
        """One nvidia-smi --query-gpu call; None when the tool or the field is unavailable."""
        try:
            res = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().splitlines()[0]
        except Exception:
            pass
        return None

    @staticmethod
    def _cuda_cores(compute_cap: str, probe_torch: bool) -> int:
        """
        CUDA cores = SM count x cores-per-SM. The SM count is only knowable from the CUDA
        runtime, so it costs a torch import and a CUDA context; that is off by default and
        only happens when the caller asks or LAZYLORA_PROFILE_TORCH=1 is set. Otherwise
        this returns 0, which the report prints as "not probed" rather than as a guess.
        """
        if not probe_torch:
            return 0
        try:
            major, minor = (int(x) for x in compute_cap.split(".")[:2])
        except Exception:
            return 0
        per_sm = _CORES_PER_SM.get((major, minor))
        if per_sm is None:
            return 0
        try:
            import torch  # noqa: PLC0415 - deliberately lazy: a hardware audit must not need torch
            if not torch.cuda.is_available():
                return 0
            return int(torch.cuda.get_device_properties(0).multi_processor_count) * per_sm
        except Exception:
            return 0

    @classmethod
    def get_gpu_info(cls, probe_torch: bool = False) -> Optional[GPUInfo]:
        row = cls._query_smi("name,memory.total,memory.free,driver_version")
        if row is None:
            return None
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 3:
            return None
        name = parts[0]
        try:
            total_vram = float(parts[1])
            free_vram = float(parts[2])
        except ValueError:
            return None
        driver_ver = parts[3] if len(parts) > 3 else "unknown"

        # compute_cap is a separate call: old drivers reject the field and would otherwise
        # take the whole query down with it.
        cap_row = cls._query_smi("compute_cap")
        compute_cap = cap_row.strip() if cap_row else "unknown"

        cuda_ver = "unknown"
        try:
            smi_all = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if smi_all.returncode == 0:
                for line in smi_all.stdout.splitlines():
                    if "CUDA Version:" in line:
                        cuda_ver = line.split("CUDA Version:")[1].split("|")[0].strip()
                        break
        except Exception:
            pass

        return GPUInfo(
            name=name,
            total_vram_mb=total_vram,
            free_vram_mb=free_vram,
            cuda_version=cuda_ver,
            driver_version=driver_ver,
            cuda_cores_approx=cls._cuda_cores(compute_cap, probe_torch),
            compute_capability=compute_cap,
        )

    # ---------------------------------------------------------------------- memory

    @staticmethod
    def get_memory_info() -> MemoryInfo:
        total_ram = avail_ram = used_ram = swap_total = swap_free = 0.0

        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    data = {}
                    for line in f:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            data[k.strip()] = int(v.strip().split()[0])  # kB
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
                pass

        return MemoryInfo(
            total_ram_gb=round(total_ram, 2),
            available_ram_gb=round(avail_ram, 2),
            used_ram_gb=round(used_ram, 2),
            swap_total_gb=round(swap_total, 2),
            swap_free_gb=round(swap_free, 2),
        )

    # ------------------------------------------------------------------------ disks

    @staticmethod
    def _system_volume() -> str:
        """The volume the operating system itself lives on."""
        if platform.system() == "Windows":
            return os.environ.get("SystemDrive", "C:") + "\\"
        return "/"

    @classmethod
    def _is_system_path(cls, path: str) -> bool:
        p = path.lower()
        if p in ("/", "c:\\", "c:/"):
            return True
        # A WSL guest's /mnt/c is the Windows system volume seen from inside Linux.
        return p == "/mnt/c" or p.startswith("/mnt/c/")

    @classmethod
    def _engine_paths(cls) -> List[Tuple[str, str]]:
        """(path, label) for the volumes the engine reads and writes, from the config."""
        from lazy_lora.core.config import (default_model_dir, default_workspace_dir,
                                           default_fast_scratch_dir)
        return [
            (default_model_dir(), "Model checkpoint volume"),
            (default_workspace_dir(), "Workspace volume"),
            (default_fast_scratch_dir(), "Fast scratch volume (activations, checkpoints)"),
        ]

    @classmethod
    def get_disk_info(cls) -> List[DiskInfo]:
        is_wsl = cls.is_wsl()
        paths_to_check: List[Tuple[str, str]] = []

        if platform.system() == "Windows":
            paths_to_check.append((cls._system_volume(), "System volume"))
        elif is_wsl:
            # Detected WSL: the Windows volumes are visible under /mnt and worth naming,
            # but the guest root is still the system volume as far as writes are concerned.
            paths_to_check.append(("/", "WSL guest root (system)"))
            for letter in ("c", "d", "e"):
                mnt = f"/mnt/{letter}"
                if os.path.isdir(mnt):
                    role = "Windows system volume" if letter == "c" else "Windows volume"
                    paths_to_check.append((mnt, f"{role} ({letter.upper()}:)"))
        else:
            paths_to_check.append(("/", "Root filesystem (system)"))

        # Every volume the engine actually uses, wherever the environment put it. Walk up to
        # the nearest existing ancestor so an unmounted or not-yet-created directory is still
        # reported against the volume it would land on.
        try:
            engine_paths = cls._engine_paths()
        except Exception:
            engine_paths = []
        for p, label in engine_paths:
            q = p
            while q and not os.path.exists(q):
                parent = os.path.dirname(q)
                if parent == q:
                    q = ""
                    break
                q = parent
            if q and q not in [x for x, _ in paths_to_check]:
                paths_to_check.append((q, label))

        disks = []
        for p, label in paths_to_check:
            if not os.path.exists(p):
                continue
            try:
                usage = shutil.disk_usage(p)
            except Exception:
                continue
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            pct = (used_gb / total_gb) * 100.0 if total_gb > 0 else 0.0
            is_system = cls._is_system_path(p)
            disks.append(DiskInfo(
                path=p,
                label=label,
                total_gb=round(total_gb, 2),
                used_gb=round(used_gb, 2),
                free_gb=round(free_gb, 2),
                percent_used=round(pct, 1),
                is_system_volume=is_system,
                is_safe_for_storage=(free_gb > MIN_FREE_GB_FOR_STORAGE and not is_system),
            ))
        return disks

    # ---------------------------------------------------------------------- profile

    @classmethod
    def analyze_system(cls, probe_torch: Optional[bool] = None) -> SystemHardwareProfile:
        """
        Probe the machine. probe_torch=True imports torch to turn the GPU's SM count into a
        CUDA core count; it defaults to LAZYLORA_PROFILE_TORCH=1, i.e. off, so that a
        hardware audit never needs the training stack installed.
        """
        if probe_torch is None:
            probe_torch = os.environ.get("LAZYLORA_PROFILE_TORCH", "") == "1"

        cpu = cls.get_cpu_info()
        gpu = cls.get_gpu_info(probe_torch=probe_torch)
        mem = cls.get_memory_info()
        disks = cls.get_disk_info()
        is_wsl = cls.is_wsl()

        if platform.system() == "Linux":
            distro = cls._distro_name() or platform.platform()
            os_name = f"{distro} [WSL2, kernel {platform.release()}]" if is_wsl else \
                      f"{distro} (kernel {platform.release()})"
        else:
            os_name = platform.platform()

        # The system volume is what must not fill up, whatever it is called on this OS.
        sys_disk = next((d for d in disks if d.is_system_volume), None)
        system_volume_safe = not (sys_disk and sys_disk.free_gb < SYSTEM_VOLUME_WARN_GB)

        if is_wsl:
            # Only on a detected WSL host is a /mnt/<letter> path a sensible recommendation.
            big = max((d for d in disks if d.is_safe_for_storage), key=lambda d: d.free_gb,
                      default=None)
            rec_storage = os.path.join(big.path, "LazyLora_Workspace") if big else "./workspace"
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
            system_volume_safe=system_volume_safe,
            recommended_vram_budget_mb=round(rec_vram_budget, 1),
            recommended_ram_budget_gb=round(rec_ram_budget, 1),
            recommended_storage_path=rec_storage,
        )

    # ----------------------------------------------------------------------- report

    @classmethod
    def print_report(cls) -> None:
        prof = cls.analyze_system()
        w = 78
        print("=" * w)
        print(" " * 20 + "LAZYLORA HARDWARE & SYSTEM AUDIT")
        print("=" * w)
        print(f" OS                   : {prof.os_name}")
        print(f" CPU Model            : {prof.cpu.model}")
        print(f" CPU Cores / Threads  : {prof.cpu.physical_cores} cores / "
              f"{prof.cpu.logical_cores} threads ({prof.cpu.architecture})")

        if prof.gpu:
            print(f" GPU Model            : {prof.gpu.name}")
            print(f" GPU VRAM             : {prof.gpu.total_vram_mb:.0f} MB total "
                  f"({prof.gpu.free_vram_mb:.0f} MB free)")
            print(f" CUDA & Driver        : CUDA {prof.gpu.cuda_version} | "
                  f"driver {prof.gpu.driver_version} | compute capability "
                  f"{prof.gpu.compute_capability}")
            if prof.gpu.cuda_cores_approx:
                print(f" CUDA Cores           : ~{prof.gpu.cuda_cores_approx} "
                      f"(SM count x cores per SM)")
            else:
                print(" CUDA Cores           : not probed "
                      "(set LAZYLORA_PROFILE_TORCH=1 to read the SM count from CUDA)")
        else:
            print(" GPU                  : none reported by nvidia-smi "
                  "(the engine runs on CPU)")

        print(f" System RAM           : {prof.memory.total_ram_gb} GB total "
              f"({prof.memory.available_ram_gb} GB available)")
        if prof.memory.swap_total_gb > 0:
            print(f" Swap                 : {prof.memory.swap_total_gb} GB total "
                  f"({prof.memory.swap_free_gb} GB free)")

        print("-" * w)
        print(" STORAGE VOLUMES:")
        for d in prof.disks:
            if d.is_system_volume:
                verdict = "[SYSTEM VOLUME - not for model data]"
            elif d.is_safe_for_storage:
                verdict = "[OK - usable for model-scale data]"
            else:
                verdict = f"[under {MIN_FREE_GB_FOR_STORAGE:.0f} GB free - too small]"
            print(f"  * {d.label} ({d.path})")
            print(f"    total {d.total_gb:.1f} GB | free {d.free_gb:.1f} GB "
                  f"({d.percent_used:.1f}% used) -> {verdict}")

        print("-" * w)
        print(" RESOURCE BUDGETS:")
        print(f"  * Recommended VRAM cap   : {prof.recommended_vram_budget_mb:.0f} MB")
        print(f"  * Recommended RAM cap    : {prof.recommended_ram_budget_gb:.1f} GB")
        print(f"  * Recommended workspace  : {prof.recommended_storage_path}")
        print("    (set LAZYLORA_WORKSPACE_DIR, LAZYLORA_MODEL_DIR and "
              "LAZYLORA_FAST_SCRATCH_DIR to move it)")

        if not prof.system_volume_safe:
            print(f"  ! WARNING: the system volume has under {SYSTEM_VOLUME_WARN_GB:.0f} GB "
                  "free. Keep checkpoints and scratch off it.")
        print("=" * w)


if __name__ == "__main__":
    HardwareProfiler.print_report()
