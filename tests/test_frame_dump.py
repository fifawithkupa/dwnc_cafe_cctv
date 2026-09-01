"""Unit tests for the per-tick still writer.

No video and no model: the naming rules are pure, and the writing is
checked against a 2x2 array in a temporary directory.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.frame_dump import CLEAN_DIR, MARKED_DIR, frame_paths, frame_stem, save_frame_pair


class FrameStemTests(unittest.TestCase):
    def test_zero_is_padded(self):
        self.assertEqual(frame_stem(0.0), "t0000.0s")

    def test_whole_second_keeps_one_decimal(self):
        self.assertEqual(frame_stem(15.0), "t0015.0s")

    def test_fractional_second_is_kept(self):
        self.assertEqual(frame_stem(1234.5), "t1234.5s")

    def test_stems_sort_in_time_order(self):
        stems = [frame_stem(t) for t in (5.0, 15.0, 105.0, 1005.0)]
        self.assertEqual(stems, sorted(stems))

    def test_negative_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            frame_stem(-0.1)


class FramePathsTests(unittest.TestCase):
    def test_clean_comes_first_and_lives_in_its_own_directory(self):
        clean, marked = frame_paths(Path("frames/angle1"), 15.0)
        self.assertEqual(clean, Path("frames/angle1") / CLEAN_DIR / "t0015.0s.jpg")
        self.assertEqual(marked, Path("frames/angle1") / MARKED_DIR / "t0015.0s.jpg")

    def test_the_two_directories_are_different(self):
        # The blinding rule depends on a grader being able to be pointed at
        # clean/ alone; if they shared a directory that would be impossible.
        self.assertNotEqual(CLEAN_DIR, MARKED_DIR)


class SaveFramePairTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((2, 2, 3), dtype=np.uint8)
        self.marked = np.full((2, 2, 3), 255, dtype=np.uint8)

    def test_both_files_are_written(self):
        with tempfile.TemporaryDirectory() as raw:
            clean_path, marked_path = save_frame_pair(
                Path(raw), 15.0, self.clean, self.marked
            )
            self.assertTrue(clean_path.exists())
            self.assertTrue(marked_path.exists())

    def test_missing_directories_are_created(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "does" / "not" / "exist"
            clean_path, _ = save_frame_pair(target, 0.0, self.clean, self.marked)
            self.assertTrue(clean_path.exists())

    def test_unwritable_destination_raises(self):
        # Silently skipping a still would be indistinguishable from "that tick
        # was never judged", which is the one thing the harness must not blur.
        with tempfile.TemporaryDirectory() as raw:
            blocker = Path(raw) / CLEAN_DIR
            blocker.write_text("not a directory", encoding="utf-8")
            with self.assertRaises((RuntimeError, OSError, NotADirectoryError, FileExistsError)):
                save_frame_pair(Path(raw), 0.0, self.clean, self.marked)


class SeatnowArgumentTests(unittest.TestCase):
    """--frame-dir must be independent of --no-video."""

    def _parse(self, argv):
        from engine import seatnow

        return seatnow.build_parser().parse_args(argv)

    def test_frame_dir_defaults_to_none(self):
        args = self._parse(["input.mov"])
        self.assertIsNone(args.frame_dir)

    def test_frame_dir_is_a_path(self):
        args = self._parse(["input.mov", "--frame-dir", "frames/angle1"])
        self.assertEqual(args.frame_dir, Path("frames/angle1"))

    def test_frame_dir_combines_with_no_video(self):
        args = self._parse(["input.mov", "--frame-dir", "frames/a", "--no-video"])
        self.assertEqual(args.frame_dir, Path("frames/a"))
        self.assertTrue(args.no_video)


if __name__ == "__main__":
    unittest.main()
