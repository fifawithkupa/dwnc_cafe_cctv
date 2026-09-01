"""Unit tests for hardware-decode selection.

These never invoke ffmpeg: the probe is injected so the decision logic can
be tested on every platform, including ones with no accelerator at all.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List

from engine.seatnow_hwaccel import (
    HWACCEL_CHOICES,
    HwaccelChoice,
    candidate_order,
    hwaccel_input_args,
    resolve_hwaccel,
)


SAMPLE = Path("sample.mp4")


def prober_accepting(*accepted: str):
    """Return a probe that succeeds only for the named accelerators."""
    seen: List[str] = []

    def probe(name: str, sample_path: Path) -> bool:
        seen.append(name)
        return name in accepted

    probe.seen = seen  # type: ignore[attr-defined]
    return probe


class CandidateOrderTests(unittest.TestCase):
    def test_windows_prefers_quick_sync(self):
        self.assertEqual(candidate_order("win32"), ("qsv", "d3d11va", "dxva2"))

    def test_linux_prefers_quick_sync_then_vaapi(self):
        self.assertEqual(candidate_order("linux"), ("qsv", "vaapi"))

    def test_macos_uses_videotoolbox(self):
        self.assertEqual(candidate_order("darwin"), ("videotoolbox",))

    def test_unknown_platform_has_no_candidates(self):
        self.assertEqual(candidate_order("freebsd14"), ())


class InputArgsTests(unittest.TestCase):
    def test_none_produces_no_arguments(self):
        self.assertEqual(hwaccel_input_args("none"), ())

    def test_named_accelerator_becomes_hwaccel_flag(self):
        self.assertEqual(hwaccel_input_args("qsv"), ("-hwaccel", "qsv"))

    def test_auto_is_not_an_ffmpeg_value(self):
        with self.assertRaises(ValueError):
            hwaccel_input_args("auto")


class ResolveTests(unittest.TestCase):
    def test_auto_picks_the_first_working_candidate(self):
        probe = prober_accepting("d3d11va")
        choice = resolve_hwaccel("auto", SAMPLE, platform_name="win32", prober=probe)
        self.assertEqual(choice.name, "d3d11va")
        self.assertEqual(choice.args, ("-hwaccel", "d3d11va"))
        self.assertFalse(choice.fallback)
        self.assertEqual(probe.seen, ["qsv", "d3d11va"])

    def test_auto_falls_back_to_software_and_says_so(self):
        probe = prober_accepting()
        choice = resolve_hwaccel("auto", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "none")
        self.assertEqual(choice.args, ())
        self.assertTrue(choice.fallback)
        self.assertEqual(choice.tried, ("qsv", "vaapi"))

    def test_none_never_probes(self):
        probe = prober_accepting("qsv")
        choice = resolve_hwaccel("none", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "none")
        self.assertFalse(choice.fallback)
        self.assertEqual(probe.seen, [])

    def test_explicit_request_is_verified_not_trusted(self):
        probe = prober_accepting("vaapi")
        choice = resolve_hwaccel("qsv", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "none")
        self.assertTrue(choice.fallback)
        self.assertEqual(probe.seen, ["qsv"])

    def test_explicit_request_that_works_is_used(self):
        probe = prober_accepting("qsv")
        choice = resolve_hwaccel("qsv", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "qsv")
        self.assertFalse(choice.fallback)

    def test_every_choice_is_offered_to_argparse(self):
        self.assertIn("auto", HWACCEL_CHOICES)
        self.assertIn("none", HWACCEL_CHOICES)
        for name in ("qsv", "vaapi", "d3d11va", "dxva2", "videotoolbox"):
            self.assertIn(name, HWACCEL_CHOICES)


class DescribeTests(unittest.TestCase):
    def test_software_fallback_is_reported_as_off(self):
        choice = HwaccelChoice(
            name="none", requested="auto", args=(), fallback=True, tried=("qsv",)
        )
        message = choice.describe()
        self.assertIn("꺼짐", message)
        self.assertIn("qsv", message)

    def test_enabled_accelerator_is_reported_as_on(self):
        choice = HwaccelChoice(
            name="qsv",
            requested="auto",
            args=("-hwaccel", "qsv"),
            fallback=False,
            tried=("qsv",),
        )
        self.assertIn("켜짐", choice.describe())
        self.assertIn("qsv", choice.describe())

    def test_explicit_software_is_not_a_fallback_message(self):
        choice = HwaccelChoice(
            name="none", requested="none", args=(), fallback=False, tried=()
        )
        self.assertIn("소프트웨어", choice.describe())
        self.assertNotIn("실패", choice.describe())


if __name__ == "__main__":
    unittest.main()
