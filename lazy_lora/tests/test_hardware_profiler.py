"""
Unit test for Hardware Profiler and C: Drive Zero-Write Guard.
"""

import os
import unittest
from lazy_lora.profiler.hardware import HardwareProfiler, SystemHardwareProfile


class TestHardwareProfiler(unittest.TestCase):

    def test_hardware_profiling(self):
        profile = HardwareProfiler.analyze_system()
        self.assertIsInstance(profile, SystemHardwareProfile)
        self.assertGreater(profile.cpu.physical_cores, 0)
        self.assertGreater(profile.memory.total_ram_gb, 0)
        self.assertGreater(len(profile.disks), 0)

        # Check D: drive presence and C: drive isolation
        has_d_drive = any("d:" in d.path.lower() or "/mnt/d" in d.path.lower() for d in profile.disks)
        self.assertTrue(has_d_drive, "D: Drive must be available for out-of-core storage")

        c_disk = next((d for d in profile.disks if "c:" in d.path.lower() or "/mnt/c" in d.path.lower()), None)
        if c_disk:
            # Assert C: drive is marked as protected
            self.assertFalse(c_disk.is_safe_for_storage, "C: drive must NOT be marked as safe for massive model storage")

    def test_recommended_storage_path(self):
        profile = HardwareProfiler.analyze_system()
        self.assertTrue(
            "/mnt/d" in profile.recommended_storage_path or "D:" in profile.recommended_storage_path,
            f"Recommended storage must be on D: drive, got: {profile.recommended_storage_path}",
        )


if __name__ == "__main__":
    unittest.main()
