"""Summarising a live run: tick timing, memory trend, and the sampler's CPU split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from edge.live_report import (
    memory_trend,
    render,
    summarize_records,
    summarize_samples,
)


def _record(index: int, duration: float, rss: float, late: float = 0.0, skipped: int = 0) -> dict:
    return {
        "frame_index": index,
        "inference_ms": duration * 1000 - 50,
        "tick": {"duration_s": duration, "late_s": late, "skipped_total": skipped, "burst_age_s": 0.02},
        "live": {"reconnects": 0, "hwaccel_suspect": False, "decode_fps": 30.0},
        "process": {"rss_mb": rss, "ffmpeg_rss_mb": 300.0, "elapsed_s": index * 15.0},
    }


class RecordSummaryTests(unittest.TestCase):
    def test_publish_counts_come_from_the_last_record(self) -> None:
        records = [_record(0, 30.0, 600), _record(1, 5.0, 900), _record(2, 5.0, 900)]
        records[-1]["publish"] = {"sent": 40, "failed": 2, "last_error": "HTTP 503"}
        summary = summarize_records(records, interval_seconds=15.0)
        self.assertEqual(summary["publish"], {"sent": 40, "failed": 2, "last_error": "HTTP 503"})
        self.assertIn("전송 성공 40회 · 실패 2회", render(summary, None, "x"))

    def test_no_publish_field_means_publishing_was_off(self) -> None:
        summary = summarize_records([_record(0, 30.0, 600), _record(1, 5.0, 900)], interval_seconds=15.0)
        self.assertIsNone(summary["publish"])
        self.assertIn("Supabase 전송: 꺼짐", render(summary, None, "x"))

    def test_first_tick_is_excluded_from_timing_but_reported(self) -> None:
        records = [_record(0, 40.0, 600), _record(1, 5.0, 1000), _record(2, 7.0, 1010), _record(3, 6.0, 1020)]
        summary = summarize_records(records, interval_seconds=15.0)
        self.assertEqual(summary["ticks"], 4)
        self.assertEqual(summary["first_tick_s"], 40.0)
        self.assertAlmostEqual(summary["tick_s"]["mean"], 6.0)
        self.assertEqual(summary["tick_s"]["max"], 7.0)
        self.assertEqual(summary["tick_s"]["over_budget"], 0)

    def test_ticks_over_half_the_interval_are_counted(self) -> None:
        records = [_record(0, 30.0, 600), _record(1, 8.0, 900), _record(2, 7.4, 900), _record(3, 9.1, 900)]
        summary = summarize_records(records, interval_seconds=15.0)
        self.assertEqual(summary["tick_s"]["over_budget"], 2)
        self.assertEqual(summary["budget_s"], 7.5)

    def test_skipped_slots_and_late_come_from_the_last_record(self) -> None:
        records = [_record(0, 30.0, 600), _record(1, 5.0, 900, late=0.4, skipped=1), _record(2, 5.0, 900, late=2.5, skipped=3)]
        summary = summarize_records(records, interval_seconds=15.0)
        self.assertEqual(summary["skipped_slots"], 3)
        self.assertAlmostEqual(summary["late_s"]["max"], 2.5)


class MemoryTrendTests(unittest.TestCase):
    def test_slope_is_megabytes_per_hour_from_a_linear_fit(self) -> None:
        # 10 MB every 15 s = 2400 MB/h
        points = [(i * 15.0, 1000.0 + 10.0 * i) for i in range(40)]
        trend = memory_trend(points)
        self.assertAlmostEqual(trend["mb_per_hour"], 2400.0, places=3)
        self.assertEqual(trend["first"], 1000.0)
        self.assertEqual(trend["last"], 1390.0)

    def test_flat_memory_has_zero_slope(self) -> None:
        points = [(i * 15.0, 1200.0) for i in range(20)]
        self.assertAlmostEqual(memory_trend(points)["mb_per_hour"], 0.0)

    def test_warmup_can_be_skipped(self) -> None:
        points = [(0.0, 500.0), (15.0, 1100.0)] + [(15.0 * i, 1100.0) for i in range(2, 30)]
        self.assertGreater(memory_trend(points)["mb_per_hour"], 100.0)
        self.assertAlmostEqual(memory_trend(points, skip=2)["mb_per_hour"], 0.0)


class SampleSummaryTests(unittest.TestCase):
    def test_reads_the_sampler_csv_and_averages_cores(self) -> None:
        header = "ts,load1,mem_used_mb,mem_avail_mb,swap_used_mb,cpu_total_cores,py_cores,py_rss_mb,dec_cores,dec_rss_mb,pub_cores,pub_rss_mb,mtx_cores,mtx_rss_mb,temp_c,cpu_mhz"
        rows = [
            "2026-09-06T05:00:00,1.2,1800,1900,0,1.50,1.00,1100,0.30,330,0.10,60,0.05,40,55,3100",
            "2026-09-06T05:00:10,1.3,1810,1890,0,1.70,1.20,1110,0.30,330,0.10,60,0.05,40,57,3100",
            "2026-09-06T05:00:20,1.1,1820,1880,0,,,1120,,330,,60,,40,59,3100",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
            summary = summarize_samples(path)
        self.assertEqual(summary["samples"], 3)
        self.assertAlmostEqual(summary["cores"]["py"]["mean"], 1.1)
        self.assertAlmostEqual(summary["cores"]["dec"]["mean"], 0.3)
        self.assertAlmostEqual(summary["cores"]["total"]["mean"], 1.6)
        self.assertEqual(summary["mem_used_mb"]["max"], 1820)
        self.assertEqual(summary["swap_used_mb"]["max"], 0)
        self.assertEqual(summary["temp_c"]["max"], 59)


if __name__ == "__main__":
    unittest.main()
