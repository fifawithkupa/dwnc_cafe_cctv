"""Daily log files for the 24/7 live loop: append across restarts, roll at
midnight, and throw away files older than the retention window."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from engine.seatnow import _validate_args, build_parser
from engine.seatnow_live import LiveLogRotator


def _at(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> dt.datetime:
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc)


class LogRotatorTests(unittest.TestCase):
    def test_file_is_named_by_local_date_and_opened_for_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            now = [_at(2026, 9, 6, 23, 50)]
            rotator = LiveLogRotator(Path(tmp), keep_days=14, now=lambda: now[0])
            handle = rotator.open()
            handle.write("a\n")
            rotator.close()
            rotator2 = LiveLogRotator(Path(tmp), keep_days=14, now=lambda: now[0])
            handle2 = rotator2.open()
            handle2.write("b\n")
            rotator2.close()
            path = Path(tmp) / "2026-09-06.jsonl"
            self.assertEqual(rotator.path, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "a\nb\n")

    def test_rolls_to_a_new_file_when_the_date_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            now = [_at(2026, 9, 6, 23, 59)]
            rotator = LiveLogRotator(Path(tmp), keep_days=14, now=lambda: now[0])
            first = rotator.open()
            first.write("late\n")
            self.assertIs(rotator.maybe_rotate(), first)
            now[0] = _at(2026, 9, 7, 0, 1)
            second = rotator.maybe_rotate()
            self.assertIsNot(second, first)
            second.write("early\n")
            rotator.close()
            self.assertEqual((Path(tmp) / "2026-09-06.jsonl").read_text(encoding="utf-8"), "late\n")
            self.assertEqual((Path(tmp) / "2026-09-07.jsonl").read_text(encoding="utf-8"), "early\n")

    def test_prune_deletes_only_dated_logs_older_than_keep_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("2026-08-01.jsonl", "2026-08-23.jsonl", "2026-09-05.jsonl", "notes.txt", "2026-08-01_summary.json"):
                (root / name).write_text("x", encoding="utf-8")
            rotator = LiveLogRotator(root, keep_days=14, now=lambda: _at(2026, 9, 6, 12))
            removed = rotator.prune()
            self.assertEqual(sorted(p.name for p in removed), ["2026-08-01.jsonl", "2026-08-01_summary.json"])
            self.assertTrue((root / "2026-08-23.jsonl").exists())
            self.assertTrue((root / "2026-09-05.jsonl").exists())
            self.assertTrue((root / "notes.txt").exists())

    def test_keep_days_zero_never_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2020-01-01.jsonl").write_text("x", encoding="utf-8")
            rotator = LiveLogRotator(root, keep_days=0, now=lambda: _at(2026, 9, 6))
            self.assertEqual(rotator.prune(), [])
            self.assertTrue((root / "2020-01-01.jsonl").exists())


class LogDirFlagTests(unittest.TestCase):
    def test_log_dir_and_keep_days_parse_with_defaults(self) -> None:
        args = build_parser().parse_args(["rtsp://cam/x", "--log-dir", "results/live"])
        self.assertEqual(args.log_dir, Path("results/live"))
        self.assertEqual(args.keep_days, 14)

    def test_log_dir_and_log_are_mutually_exclusive(self) -> None:
        args = build_parser().parse_args(
            ["rtsp://cam/x", "--log-dir", "results/live", "--log", "results/x.jsonl"]
        )
        with self.assertRaises(ValueError):
            _validate_args(args)

    def test_log_dir_is_live_only_and_keep_days_non_negative(self) -> None:
        args = build_parser().parse_args([__file__, "--log-dir", "results/live"])
        with self.assertRaises(ValueError):
            _validate_args(args)
        args = build_parser().parse_args(["rtsp://cam/x", "--log-dir", "r", "--keep-days", "-1"])
        with self.assertRaises(ValueError):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
