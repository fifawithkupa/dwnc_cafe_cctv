"""Unit tests for the decode benchmark's model- and ffmpeg-independent logic."""

from __future__ import annotations

import unittest
from pathlib import Path

from bench_decode import (
    CLIP_SPECS,
    ClipSpec,
    DecodeMeasurement,
    build_clip_command,
    build_decode_command,
    combined_rows,
    cores_used,
    format_decode_table,
    grade_decode,
    parse_benchmark,
)


BENCHMARK_OUTPUT = (
    "some unrelated ffmpeg chatter\n"
    "bench: utime=3.734s stime=0.453s rtime=1.126s\n"
    "bench: maxrss=115480KiB\n"
)


class ParseBenchmarkTests(unittest.TestCase):
    def test_reads_the_three_times(self):
        self.assertEqual(parse_benchmark(BENCHMARK_OUTPUT), (3.734, 0.453, 1.126))

    def test_missing_line_is_none_not_zero(self):
        self.assertIsNone(parse_benchmark("ffmpeg said nothing useful\n"))

    def test_maxrss_line_alone_is_not_enough(self):
        self.assertIsNone(parse_benchmark("bench: maxrss=115480KiB\n"))


class CoresUsedTests(unittest.TestCase):
    def test_cpu_seconds_over_content_seconds(self):
        self.assertAlmostEqual(cores_used(4.187, 10.0), 0.4187)

    def test_zero_content_is_zero_not_a_crash(self):
        self.assertEqual(cores_used(4.0, 0.0), 0.0)


class GradeTests(unittest.TestCase):
    def test_quarter_of_the_machine_passes(self):
        self.assertEqual(grade_decode(1.0, 4), "PASS")

    def test_just_over_a_quarter_is_conditional(self):
        self.assertEqual(grade_decode(1.1, 4), "CONDITIONAL")

    def test_over_half_the_machine_fails(self):
        self.assertEqual(grade_decode(2.1, 4), "FAIL")

    def test_boundary_at_half_is_still_conditional(self):
        self.assertEqual(grade_decode(2.0, 4), "CONDITIONAL")

    def test_unknown_core_count_never_claims_a_pass(self):
        self.assertEqual(grade_decode(0.1, 0), "UNKNOWN")


class ClipSpecTests(unittest.TestCase):
    def test_megapixels_are_rounded_for_display(self):
        spec = ClipSpec("4mp_h264", 2560, 1440, "h264", 6000)
        self.assertAlmostEqual(spec.megapixels, 3.7, places=1)

    def test_filename_is_derived_from_the_name(self):
        spec = ClipSpec("4mp_h264", 2560, 1440, "h264", 6000)
        self.assertEqual(spec.filename, "4mp_h264.mp4")

    def test_every_resolution_is_measured_in_both_codecs(self):
        names = {spec.name for spec in CLIP_SPECS}
        for size in ("2mp", "4mp", "8mp"):
            self.assertIn(f"{size}_h264", names)
            self.assertIn(f"{size}_h265", names)

    def test_bitrates_are_cctv_sized_not_source_sized(self):
        # The source clips are 14.9 Mbps; a real camera streams far less.
        for spec in CLIP_SPECS:
            self.assertLessEqual(spec.bitrate_kbps, 10000)


class ClipCommandTests(unittest.TestCase):
    def test_scales_and_reencodes_at_the_requested_bitrate(self):
        spec = ClipSpec("4mp_h265", 2560, 1440, "h265", 6000)
        command = build_clip_command(
            "ffmpeg", Path("src.mov"), spec, Path("out.mp4"), duration=30.0
        )
        self.assertIn("libx265", command)
        self.assertIn("scale=2560:1440", " ".join(command))
        self.assertIn("6000k", command)
        self.assertEqual(command[-1], "out.mp4")

    def test_h264_uses_the_h264_encoder(self):
        spec = ClipSpec("2mp_h264", 1920, 1080, "h264", 4000)
        command = build_clip_command(
            "ffmpeg", Path("src.mov"), spec, Path("out.mp4"), duration=30.0
        )
        self.assertIn("libx264", command)

    def test_duration_limits_the_clip(self):
        spec = ClipSpec("2mp_h264", 1920, 1080, "h264", 4000)
        command = build_clip_command(
            "ffmpeg", Path("src.mov"), spec, Path("out.mp4"), duration=30.0
        )
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "30.000")


class DecodeCommandTests(unittest.TestCase):
    def test_asks_ffmpeg_for_its_own_cpu_accounting(self):
        command = build_decode_command("ffmpeg", Path("clip.mp4"), ())
        self.assertIn("-benchmark", command)

    def test_discards_output_so_only_decoding_is_measured(self):
        command = build_decode_command("ffmpeg", Path("clip.mp4"), ())
        self.assertIn("null", command)
        self.assertNotIn("libx264", command)

    def test_hwaccel_precedes_the_input(self):
        command = build_decode_command("ffmpeg", Path("clip.mp4"), ("-hwaccel", "qsv"))
        self.assertLess(command.index("-hwaccel"), command.index("-i"))


def measurement(clip: str, hwaccel: str, cores: float) -> DecodeMeasurement:
    return DecodeMeasurement(
        clip=clip,
        hwaccel=hwaccel,
        cpu_seconds=cores * 30.0,
        wall_seconds=10.0,
        content_seconds=30.0,
        cores_used=cores,
        realtime_factor=3.0,
    )


class CombinedRowTests(unittest.TestCase):
    BENCH_REPORT = {
        "tick_budgets": [
            {"profile": "balanced", "backend": "ov-int8", "tick_utilization": 0.46},
            {
                "profile": "accuracy_default",
                "backend": "ov-int8",
                "tick_utilization": 0.90,
            },
        ]
    }

    def test_adds_decode_share_to_inference_share(self):
        rows = combined_rows(
            [measurement("4mp_h265", "qsv", 0.72)], self.BENCH_REPORT, total_cores=4
        )
        balanced = [r for r in rows if r["profile"] == "balanced"][0]
        self.assertAlmostEqual(balanced["decode_share"], 0.18)
        self.assertAlmostEqual(balanced["total_share"], 0.64)
        self.assertEqual(balanced["grade"], "PASS")

    def test_over_the_machine_is_a_fail(self):
        rows = combined_rows(
            [measurement("8mp_h265", "none", 2.4)], self.BENCH_REPORT, total_cores=4
        )
        heavy = [r for r in rows if r["profile"] == "accuracy_default"][0]
        self.assertEqual(heavy["grade"], "FAIL")

    def test_no_bench_report_yields_no_rows(self):
        self.assertEqual(combined_rows([measurement("2mp_h264", "qsv", 0.1)], None, 4), [])

    def test_software_and_hardware_rows_are_kept_apart(self):
        rows = combined_rows(
            [
                measurement("4mp_h265", "qsv", 0.72),
                measurement("4mp_h265", "none", 2.0),
            ],
            self.BENCH_REPORT,
            total_cores=4,
        )
        pairs = {(r["clip"], r["hwaccel"], r["profile"]) for r in rows}
        self.assertIn(("4mp_h265", "qsv", "balanced"), pairs)
        self.assertIn(("4mp_h265", "none", "balanced"), pairs)


class DecodeTableTests(unittest.TestCase):
    def test_table_names_the_grade_and_the_cost(self):
        table = format_decode_table([measurement("4mp_h265", "qsv", 0.72)], 4)
        self.assertIn("4mp_h265", table)
        self.assertIn("qsv", table)
        self.assertIn("PASS", table)

    def test_unknown_core_count_is_shown_not_hidden(self):
        table = format_decode_table([measurement("4mp_h265", "qsv", 0.72)], 0)
        self.assertIn("UNKNOWN", table)


if __name__ == "__main__":
    unittest.main()
