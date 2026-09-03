"""Unit tests for the edge-box readiness checks."""

from __future__ import annotations

import unittest

from edge.check_edge import (
    Check,
    check_cores,
    check_disk_gb,
    check_memory_gb,
    check_packages,
    check_python,
    format_report,
)


class PythonCheckTests(unittest.TestCase):
    def test_supported_version_passes(self):
        self.assertTrue(check_python((3, 11, 9)).ok)

    def test_too_old_fails_and_says_what_to_do(self):
        result = check_python((3, 8, 10))
        self.assertFalse(result.ok)
        self.assertIn("3.9", result.fix)


class CoreCheckTests(unittest.TestCase):
    def test_four_cores_pass(self):
        self.assertTrue(check_cores(4).ok)

    def test_two_cores_fail(self):
        self.assertFalse(check_cores(2).ok)

    def test_unknown_core_count_is_a_failure_not_a_pass(self):
        self.assertFalse(check_cores(0).ok)


class MemoryCheckTests(unittest.TestCase):
    def test_eight_gigabytes_pass(self):
        self.assertTrue(check_memory_gb(8.0).ok)

    def test_four_gigabytes_pass(self):
        self.assertTrue(check_memory_gb(4.0).ok)

    def test_reported_3_7_is_a_4gb_box_and_passes_with_warning(self):
        # OptiPlex 7040 with 4GB fitted reported 3.7GB (2026-09-03).
        check = check_memory_gb(3.7)
        self.assertTrue(check.ok)
        self.assertIn("빠듯", check.detail)

    def test_eight_gigabytes_has_no_warning(self):
        self.assertNotIn("빠듯", check_memory_gb(8.0).detail)

    def test_three_gigabytes_fail(self):
        self.assertFalse(check_memory_gb(3.0).ok)

    def test_two_gigabytes_fail(self):
        self.assertFalse(check_memory_gb(2.0).ok)


class DiskCheckTests(unittest.TestCase):
    def test_plenty_of_room_passes(self):
        self.assertTrue(check_disk_gb(50.0).ok)

    def test_no_room_for_the_models_fails(self):
        self.assertFalse(check_disk_gb(2.0).ok)


class PackageCheckTests(unittest.TestCase):
    def test_all_present_passes(self):
        checks = check_packages(
            {
                "numpy": "1.26.4",
                "cv2": "4.10.0",
                "torch": "2.2.2",
                "ultralytics": "8.4.82",
            }
        )
        self.assertTrue(all(check.ok for check in checks))

    def test_missing_package_names_the_install_command(self):
        checks = check_packages(
            {"numpy": "1.26.4", "cv2": "4.10.0", "torch": None, "ultralytics": None}
        )
        failed = [check for check in checks if not check.ok]
        self.assertEqual(len(failed), 2)
        self.assertIn("pip install", failed[0].fix)

    def test_numpy_2_is_rejected(self):
        checks = check_packages(
            {
                "numpy": "2.1.0",
                "cv2": "4.10.0",
                "torch": "2.2.2",
                "ultralytics": "8.4.82",
            }
        )
        numpy_check = [c for c in checks if c.name.startswith("numpy")][0]
        self.assertFalse(numpy_check.ok)
        self.assertIn("numpy<2", numpy_check.fix)


class ReportTests(unittest.TestCase):
    def test_report_marks_each_line_and_summarises(self):
        report = format_report(
            [
                Check("ffmpeg", True, "8.1.2", ""),
                Check("코어 수", False, "2개", "코어 4개 이상인 박스로 바꾼다"),
            ]
        )
        self.assertIn("ffmpeg", report)
        self.assertIn("코어 4개 이상인 박스로 바꾼다", report)
        self.assertIn("1개 불합격", report)

    def test_all_pass_says_so(self):
        report = format_report([Check("ffmpeg", True, "8.1.2", "")])
        self.assertIn("전부 합격", report)


if __name__ == "__main__":
    unittest.main()
