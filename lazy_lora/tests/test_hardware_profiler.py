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

        # The system volume must never be marked as safe for model-scale storage
        sys_disk = next((d for d in profile.disks
                         if d.path in ("/", "C:\\") or "/mnt/c" in d.path.lower()), None)
        if sys_disk:
            self.assertFalse(sys_disk.is_safe_for_storage, "system volume must NOT be marked as safe for massive model storage")
        # At least one non-system volume must be listed
        self.assertTrue(any(d is not sys_disk for d in profile.disks), "no storage volume besides the system one was found")

    def test_recommended_storage_path(self):
        profile = HardwareProfiler.analyze_system()
        from lazy_lora.core.config import get_default_config
        if profile.is_wsl:
            self.assertIn("/mnt/d", profile.recommended_storage_path)
        else:
            self.assertEqual(profile.recommended_storage_path, get_default_config().paths.workspace_dir)
        parent = os.path.dirname(profile.recommended_storage_path.rstrip("/"))
        self.assertTrue(os.path.isdir(parent), f"recommended storage parent does not exist: {parent}")


if __name__ == "__main__":
    unittest.main()
