"""Tests for the parameter-sweep harness (plan.md T7)."""

from __future__ import annotations

import unittest
from pathlib import Path

from bench_sweep import (
    GridPoint,
    PointResult,
    build_run_command,
    expand_grid,
    rank,
    tick_utilization,
)


def result(name, accuracy, utilization, error=None):
    return PointResult(
        point={},
        name=name,
        runtime_seconds=1.0,
        frames_total=10,
        frames_exact=int(accuracy * 10),
        frame_accuracy=accuracy,
        ignore_ratio=0.0,
        tick_seconds=utilization * 15.0,
        tick_utilization=utilization,
        fits_tick=utilization <= 1.0,
        error=error,
    )


class ExpandGridTests(unittest.TestCase):
    def test_full_cartesian_product(self):
        grid = expand_grid([640, 1280], [640], [0, 2], [15.0], [True, False])
        self.assertEqual(len(grid), 8)
        self.assertEqual(len(set(point.name for point in grid)), 8)

    def test_point_name_encodes_every_swept_axis(self):
        point = GridPoint(1280, 960, 2, 15.0, True)
        self.assertEqual(point.name, "i1280_p960_m2_s15_crop")
        self.assertEqual(
            GridPoint(640, 640, 0, 5.0, False).name, "i640_p640_m0_s5_nocrop"
        )


class RunCommandTests(unittest.TestCase):
    POINT = GridPoint(960, 640, 2, 15.0, False)

    def command(self, extra=()):
        return build_run_command(
            "python", Path("cafe.mp4"), self.POINT, Path("out.jsonl"), "cpu", extra
        )

    def test_never_writes_an_annotated_video(self):
        """Encoding an MP4 per grid point would dominate the timing."""
        self.assertIn("--no-video", self.command())

    def test_crop_flag_follows_the_grid_point(self):
        self.assertIn("--no-table-crops", self.command())
        crops = build_run_command(
            "python",
            Path("cafe.mp4"),
            GridPoint(960, 640, 2, 15.0, True),
            Path("out.jsonl"),
            "cpu",
        )
        self.assertIn("--table-crops", crops)
        self.assertNotIn("--no-table-crops", crops)

    def test_every_swept_axis_reaches_the_cli(self):
        command = self.command()
        self.assertEqual(command[command.index("--imgsz") + 1], "960")
        self.assertEqual(command[command.index("--pose-imgsz") + 1], "640")
        self.assertEqual(command[command.index("--median-frames") + 1], "2")
        self.assertEqual(command[command.index("--sample-seconds") + 1], "15.0")

    def test_extra_flags_are_appended(self):
        command = self.command(("--layout", "layouts/cafe.json"))
        self.assertEqual(command[-2:], ["--layout", "layouts/cafe.json"])


class TickUtilizationTests(unittest.TestCase):
    POINT = GridPoint(1280, 960, 2, 15.0, True)

    def test_utilization_is_per_tick_not_per_run(self):
        # 20 samples in 150 s = 7.5 s per tick = half of a 15 s tick.
        self.assertAlmostEqual(tick_utilization(self.POINT, 150.0, 20), 0.5)

    def test_over_budget_exceeds_one(self):
        self.assertGreater(tick_utilization(self.POINT, 600.0, 20), 1.0)

    def test_no_samples_does_not_divide_by_zero(self):
        self.assertEqual(tick_utilization(self.POINT, 150.0, 0), 0.0)


class RankTests(unittest.TestCase):
    def test_accuracy_that_misses_the_tick_never_wins(self):
        ordered = rank(
            [
                result("slow_but_accurate", 0.95, 1.4),
                result("fits", 0.80, 0.4),
            ]
        )
        self.assertEqual(ordered[0].name, "fits")

    def test_among_fitting_points_the_most_accurate_wins(self):
        ordered = rank(
            [
                result("cheap", 0.70, 0.2),
                result("accurate", 0.88, 0.9),
            ]
        )
        self.assertEqual(ordered[0].name, "accurate")

    def test_failed_runs_sort_last(self):
        ordered = rank(
            [
                result("broken", 0.0, 0.0, error="boom"),
                result("ok", 0.5, 0.3),
            ]
        )
        self.assertEqual(ordered[-1].name, "broken")


if __name__ == "__main__":
    unittest.main()
