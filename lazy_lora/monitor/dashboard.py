"""
Rich Visual Terminal Dashboard for Real-Time Monitoring of LazyLoRA Training.
Displays live VRAM, RAM, NVMe I/O throughput, active layer & expert indices,
loss curves, and C: drive safety isolation status.
"""

import sys
import time
from typing import Optional, List
from lazy_lora.monitor.metrics import TrainingStepMetrics


class TerminalDashboard:
    """
    Renders clean, high-aesthetic terminal UI for LazyLoRA monitoring.
    """

    def __init__(self, title: str = "LazyLoRA Engine - Kimi K3 Turkish Training Monitor"):
        self.title = title

    @staticmethod
    def _render_bar(val: float, max_val: float, length: int = 24, fill_char: str = "█", empty_char: str = "░") -> str:
        pct = min(1.0, max(0.0, val / max_val)) if max_val > 0 else 0.0
        filled = int(round(pct * length))
        return f"[{fill_char * filled}{empty_char * (length - filled)}] {pct * 100:.1f}%"

    @staticmethod
    def _format_time(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def render(self, m: TrainingStepMetrics, clear_screen: bool = False) -> None:
        """Render complete dashboard snapshot."""
        w = 82
        lines = []
        if clear_screen:
            lines.append("\033[2J\033[H")

        lines.append("=" * w)
        lines.append(f"  🚀  {self.title.upper()}  🚀")
        lines.append("=" * w)

        # Step Progress
        step_bar = self._render_bar(m.step, m.total_steps, length=28)
        lines.append(f" PROGRESS      : Step {m.step:04d}/{m.total_steps:04d}  {step_bar}")
        
        # Layer & Expert Routing Info
        layer_bar = self._render_bar(m.active_layer + 1, m.total_layers, length=20)
        experts_str = ", ".join([str(e) for e in m.active_experts[:8]])
        if len(m.active_experts) > 8:
            experts_str += f"... (+{len(m.active_experts)-8} more)"
        lines.append(f" ACTIVE LAYER  : Layer {m.active_layer+1:02d}/{m.total_layers:02d} {layer_bar}")
        lines.append(f" ACTIVE EXPERTS: Top-16 MoE [{experts_str or 'Shared Only'}]")

        lines.append("-" * w)
        # Memory Gauges
        vram_bar = self._render_bar(m.vram_used_mb, m.vram_total_mb, length=20)
        ram_bar = self._render_bar(m.ram_used_gb, m.ram_total_gb, length=20)
        
        lines.append(f" GPU VRAM (GTX 980 Ti): {m.vram_used_mb:.0f} / {m.vram_total_mb:.0f} MB  {vram_bar} (Peak: {m.vram_peak_mb:.0f} MB)")
        lines.append(f" SYSTEM RAM (WSL/Host): {m.ram_used_gb:.1f} / {m.ram_total_gb:.1f} GB  {ram_bar}")
        lines.append(f" NVMe I/O (D: Drive)  : {m.disk_read_mbps:.1f} MB/s Streaming Throughput")
        
        # Safety Alert
        c_status = "[LOCKED & PROTECTED]" if m.c_drive_free_gb >= 8.0 else "[WARNING: LOW SPACE]"
        lines.append(f" C: DRIVE SAFETY GUARD: {m.c_drive_free_gb:.1f} GB Free  ->  {c_status} (Zero C: Writes)")

        lines.append("-" * w)
        # Training Performance & Metrics
        lines.append(
            f" LOSS: {m.loss:.4f} (EMA: {m.smooth_loss:.4f}) | LR: {m.lr:.2e} | SPEED: {m.tokens_per_sec:.0f} tok/s | STEP TIME: {m.step_time_sec:.2f}s"
        )
        lines.append(
            f" TIME ELAPSED: {self._format_time(m.elapsed_time_sec)} | ESTIMATED TIME TO COMPLETION (ETA): {self._format_time(m.eta_sec)}"
        )
        lines.append("=" * w)

        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    # Test dashboard rendering
    dash = TerminalDashboard()
    dummy_metric = TrainingStepMetrics(
        step=42,
        total_steps=1000,
        loss=2.341,
        smooth_loss=2.450,
        lr=2e-4,
        active_layer=15,
        total_layers=93,
        active_experts=[3, 14, 89, 120, 342, 511, 789, 810, 850],
        vram_used_mb=2150.0,
        vram_total_mb=6144.0,
        vram_peak_mb=2300.0,
        ram_used_gb=4.2,
        ram_total_gb=16.0,
        disk_read_mbps=168.4,
        c_drive_free_gb=8.5,
        step_time_sec=1.45,
        tokens_per_sec=353.0,
        elapsed_time_sec=60.9,
        eta_sec=1389.1,
    )
    dash.render(dummy_metric)
