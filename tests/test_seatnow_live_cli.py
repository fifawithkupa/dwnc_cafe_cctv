"""CLI plumbing for live mode: URL inputs, the wall-clock tick schedule, and
the per-record measurements the 24/7 run is judged on."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from engine.seatnow import _validate_args, build_parser, live_log_default
from engine.seatnow_live import (
    TickSchedule,
    build_hwaccel_probe_command,
    build_probe_command,
    burst_period_frames,
    process_rss_mb,
    resolve_max_frame_age,
)


class UrlInputTests(unittest.TestCase):
    def test_parser_keeps_an_rtsp_url_as_a_string(self) -> None:
        args = build_parser().parse_args(["rtsp://192.168.0.5:554/stream"])
        self.assertEqual(args.input, "rtsp://192.168.0.5:554/stream")

    def test_parser_turns_a_file_argument_into_a_path(self) -> None:
        args = build_parser().parse_args(["sample_raw/cafe.mov"])
        self.assertIsInstance(args.input, Path)

    def test_validate_does_not_require_a_url_to_exist_on_disk(self) -> None:
        args = build_parser().parse_args(["rtsp://cam/stream", "--run-seconds", "60"])
        _validate_args(args)  # must not raise FileNotFoundError

    def test_run_seconds_must_be_positive(self) -> None:
        args = build_parser().parse_args(["rtsp://cam/stream", "--run-seconds", "0"])
        with self.assertRaises(ValueError):
            _validate_args(args)

    def test_run_seconds_is_rejected_for_file_input(self) -> None:
        args = build_parser().parse_args([__file__, "--run-seconds", "10"])
        with self.assertRaises(ValueError):
            _validate_args(args)

    def test_live_burst_seconds_defaults_to_five_and_rejects_negative(self) -> None:
        args = build_parser().parse_args(["rtsp://cam/stream"])
        self.assertEqual(args.live_burst_seconds, 5.0)
        args = build_parser().parse_args(["rtsp://cam/stream", "--live-burst-seconds", "-1"])
        with self.assertRaises(ValueError):
            _validate_args(args)

    def test_burst_period_in_frames_follows_the_stream_fps(self) -> None:
        self.assertEqual(burst_period_frames(29.97, 5.0, 5), 150)
        self.assertEqual(burst_period_frames(20.0, 5.0, 5), 100)
        self.assertEqual(burst_period_frames(0.0, 5.0, 5), 150)   # unknown fps: assume 30
        self.assertEqual(burst_period_frames(30.0, 0.0, 5), 0)    # 0 = convert every frame
        self.assertEqual(burst_period_frames(30.0, 0.1, 5), 5)    # never shorter than the burst

    def test_max_frame_age_defaults_to_one_interval(self) -> None:
        args = build_parser().parse_args(["rtsp://cam/stream"])
        self.assertIsNone(args.max_frame_age_seconds)
        self.assertEqual(resolve_max_frame_age(args.max_frame_age_seconds, 15.0), 15.0)
        self.assertEqual(resolve_max_frame_age(None, 30.0), 30.0)

    def test_max_frame_age_can_be_widened_or_switched_off(self) -> None:
        self.assertEqual(resolve_max_frame_age(45.0, 15.0), 45.0)
        self.assertIsNone(resolve_max_frame_age(0.0, 15.0))

    def test_negative_max_frame_age_is_rejected(self) -> None:
        args = build_parser().parse_args(
            ["rtsp://cam/stream", "--max-frame-age-seconds", "-3"]
        )
        with self.assertRaises(ValueError):
            _validate_args(args)

    def test_live_log_default_is_under_results_live(self) -> None:
        path = live_log_default("rtsp://192.168.0.5:554/Streaming/Channels/101")
        self.assertEqual(path.parts[-3:], ("results", "live", "log.jsonl"))


class TickScheduleTests(unittest.TestCase):
    def test_first_tick_is_due_immediately(self) -> None:
        schedule = TickSchedule(15.0, clock=lambda: 100.0)
        self.assertEqual(schedule.wait_seconds(), 0.0)
        self.assertEqual(schedule.scheduled, 100.0)

    def test_ticks_advance_by_the_interval_when_on_time(self) -> None:
        now = [100.0]
        schedule = TickSchedule(15.0, clock=lambda: now[0])
        schedule.advance()
        now[0] = 104.0
        self.assertEqual(schedule.scheduled, 115.0)
        self.assertAlmostEqual(schedule.wait_seconds(), 11.0)
        self.assertEqual(schedule.skipped, 0)

    def test_a_slow_tick_skips_the_missed_slots_and_counts_them(self) -> None:
        now = [100.0]
        schedule = TickSchedule(15.0, clock=lambda: now[0])
        schedule.advance()          # tick 1 done, next at 115
        now[0] = 147.0              # 32 s late: slots 115 and 130 are gone
        schedule.advance()
        self.assertEqual(schedule.scheduled, 160.0)
        self.assertEqual(schedule.skipped, 2)
        self.assertEqual(schedule.wait_seconds(), 13.0)

    def test_late_seconds_reports_how_far_behind_the_tick_started(self) -> None:
        now = [100.0]
        schedule = TickSchedule(15.0, clock=lambda: now[0])
        schedule.advance()
        now[0] = 118.5
        self.assertAlmostEqual(schedule.late_seconds(), 3.5)
        now[0] = 110.0
        self.assertEqual(schedule.late_seconds(), 0.0)


class ProbeCommandTests(unittest.TestCase):
    def test_probe_command_uses_tcp_transport_before_the_input(self) -> None:
        command = build_probe_command("ffprobe", "rtsp://cam/stream")
        input_index = command.index("-i")
        self.assertEqual(command[input_index + 1], "rtsp://cam/stream")
        self.assertLess(command.index("-rtsp_transport"), input_index)
        self.assertIn("codec_name,width,height", " ".join(command))

    def test_probe_command_for_a_file_has_no_rtsp_flags(self) -> None:
        command = build_probe_command("ffprobe", "clip.mp4")
        self.assertNotIn("-rtsp_transport", command)

    def test_hwaccel_probe_decodes_one_frame_over_tcp(self) -> None:
        command = build_hwaccel_probe_command("ffmpeg", "vaapi", "rtsp://cam/stream")
        input_index = command.index("-i")
        self.assertEqual(command[command.index("-hwaccel") + 1], "vaapi")
        self.assertLess(command.index("-hwaccel"), input_index)
        self.assertLess(command.index("-rtsp_transport"), input_index)
        self.assertEqual(command[command.index("-frames:v") + 1], "1")
        self.assertEqual(command[-1], "pipe:1")


class MemoryHelperTests(unittest.TestCase):
    def test_own_rss_is_a_positive_number_of_megabytes(self) -> None:
        rss = process_rss_mb()
        self.assertIsNotNone(rss)
        self.assertGreater(rss, 1.0)

    def test_unknown_pid_reports_none(self) -> None:
        self.assertIsNone(process_rss_mb(pid=2_000_000_000))


if __name__ == "__main__":
    unittest.main()
