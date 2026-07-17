"""Integration tests for SeatNow's FFmpeg-backed video I/O.

These tests intentionally use the system ``ffmpeg`` binary to create and
decode a tiny H.264 fixture.  They do not depend on OpenCV's video backend,
which is unavailable in the target macOS environment.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from seatnow_core import (
    FFmpegBurstReader,
    FFmpegSampleReader,
    FFmpegVideoWriter,
    probe_video,
    require_ffmpeg,
)


class FFmpegVideoIOTests(unittest.TestCase):
    WIDTH = 160
    HEIGHT = 96
    SOURCE_FPS = 10.0
    SOURCE_DURATION = 2.0
    SAMPLE_SECONDS = 0.5

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.ffmpeg, _ = require_ffmpeg()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        cls._temporary_directory = tempfile.TemporaryDirectory(
            prefix="seatnow-video-io-",
        )
        cls.directory = Path(cls._temporary_directory.name)
        cls.source_path = cls.directory / "synthetic_source.mp4"
        command = [
            cls.ffmpeg,
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={cls.WIDTH}x{cls.HEIGHT}:rate={cls.SOURCE_FPS:g}",
            "-t",
            f"{cls.SOURCE_DURATION:g}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(cls.source_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            cls._temporary_directory.cleanup()
            raise RuntimeError(f"Could not create synthetic test video: {completed.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()
        super().tearDownClass()

    def test_probe_video_reports_synthetic_metadata(self) -> None:
        info = probe_video(self.source_path)

        self.assertEqual(info.width, self.WIDTH)
        self.assertEqual(info.height, self.HEIGHT)
        self.assertEqual(info.codec, "h264")
        self.assertAlmostEqual(info.fps, self.SOURCE_FPS, places=3)
        self.assertAlmostEqual(info.duration, self.SOURCE_DURATION, delta=0.1)
        self.assertEqual(
            info.source_frames,
            round(self.SOURCE_FPS * self.SOURCE_DURATION),
        )

    def test_sample_reader_returns_expected_frames_shapes_and_timestamps(self) -> None:
        reader = FFmpegSampleReader(self.source_path, self.SAMPLE_SECONDS)
        samples = list(reader)

        self.assertEqual(len(samples), 4)
        self.assertEqual([sample[0] for sample in samples], [0, 1, 2, 3])
        self.assertEqual([sample[1] for sample in samples], [0.0, 0.5, 1.0, 1.5])
        for _, _, frame in samples:
            self.assertEqual(frame.shape, (self.HEIGHT, self.WIDTH, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertTrue(frame.flags.c_contiguous)

        self.assertFalse(np.array_equal(samples[0][2], samples[-1][2]))
        self.assertIsNotNone(reader.process)
        assert reader.process is not None
        self.assertEqual(reader.process.poll(), 0)

    def test_video_writer_outputs_h264_with_expected_frames_and_redecodes(self) -> None:
        source_samples = list(
            FFmpegSampleReader(self.source_path, self.SAMPLE_SECONDS)
        )
        output_path = self.directory / "writer_output.mp4"
        output_fps = 1.0 / self.SAMPLE_SECONDS

        with FFmpegVideoWriter(
            output_path,
            width=self.WIDTH,
            height=self.HEIGHT,
            fps=output_fps,
        ) as writer:
            for _, _, frame in source_samples:
                writer.write(frame)

        self.assertTrue(writer.closed)
        self.assertEqual(writer.process.poll(), 0)
        self.assertTrue(output_path.is_file())
        self.assertGreater(output_path.stat().st_size, 0)

        info = probe_video(output_path)
        self.assertEqual(info.codec, "h264")
        self.assertEqual((info.width, info.height), (self.WIDTH, self.HEIGHT))
        self.assertAlmostEqual(info.fps, output_fps, places=3)
        self.assertEqual(info.source_frames, len(source_samples))
        self.assertAlmostEqual(
            info.duration,
            len(source_samples) / output_fps,
            delta=0.1,
        )

        playback = subprocess.run(
            [
                self.ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(output_path),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(playback.returncode, 0, playback.stderr)

        decoded = list(FFmpegSampleReader(output_path, self.SAMPLE_SECONDS))
        self.assertEqual(len(decoded), len(source_samples))
        for _, _, frame in decoded:
            self.assertEqual(frame.shape, (self.HEIGHT, self.WIDTH, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_sample_reader_early_close_is_silent_and_reaps_process(self) -> None:
        """A max-samples-style early break must be normal, not a decode error."""
        reader = FFmpegSampleReader(self.source_path, sample_seconds=0.1)
        iterator = iter(reader)
        first = next(iterator)
        self.assertEqual(first[0], 0)
        process = reader.process
        self.assertIsNotNone(process)
        assert process is not None
        self.assertIsNone(process.poll())

        close_error = None
        try:
            iterator.close()
        except Exception as exc:  # retain the process-reaping assertion below
            close_error = exc
        finally:
            deadline = time.monotonic() + 2.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

        self.assertIsNotNone(process.poll(), "ffmpeg process leaked")
        self.assertIsNone(close_error, f"early reader close raised: {close_error}")

    def test_burst_reader_returns_centered_native_fps_frames(self) -> None:
        reader = FFmpegBurstReader(self.source_path)
        center_index, burst = reader.read_burst(1.0, n=2)

        self.assertEqual(len(burst), 5)
        self.assertEqual(center_index, 2)
        expected = [0.8, 0.9, 1.0, 1.1, 1.2]
        for (timestamp, frame), expected_ts in zip(burst, expected):
            self.assertAlmostEqual(timestamp, expected_ts, places=6)
            self.assertEqual(frame.shape, (self.HEIGHT, self.WIDTH, 3))
            self.assertEqual(frame.dtype, np.uint8)
        # testsrc2 animates, so consecutive frames must differ.
        self.assertFalse(np.array_equal(burst[0][1], burst[-1][1]))

    def test_burst_reader_head_shifts_instead_of_truncating(self) -> None:
        reader = FFmpegBurstReader(self.source_path)
        center_index, burst = reader.read_burst(0.0, n=2)

        self.assertEqual(len(burst), 5)
        self.assertEqual(center_index, 0)
        self.assertAlmostEqual(burst[0][0], 0.0, places=6)

    def test_burst_reader_truncates_at_end_of_video(self) -> None:
        reader = FFmpegBurstReader(self.source_path)
        center_index, burst = reader.read_burst(1.9, n=2)

        self.assertLess(len(burst), 5)
        self.assertGreaterEqual(len(burst), 1)
        self.assertAlmostEqual(burst[center_index][0], 1.9, delta=0.11)

    def test_burst_reader_n_zero_returns_single_frame(self) -> None:
        reader = FFmpegBurstReader(self.source_path)
        center_index, burst = reader.read_burst(1.0, n=0)

        self.assertEqual(len(burst), 1)
        self.assertEqual(center_index, 0)
        self.assertAlmostEqual(burst[0][0], 1.0, places=6)

    def test_burst_reader_past_end_returns_empty(self) -> None:
        reader = FFmpegBurstReader(self.source_path)
        center_index, burst = reader.read_burst(5.0, n=2)

        self.assertEqual(burst, [])
        self.assertEqual(center_index, 0)

    def test_writer_context_does_not_mask_body_exception(self) -> None:
        output_path = self.directory / "aborted_writer.mp4"
        with self.assertRaisesRegex(ValueError, "original body failure"):
            with FFmpegVideoWriter(
                output_path,
                width=self.WIDTH,
                height=self.HEIGHT,
                fps=2.0,
            ):
                raise ValueError("original body failure")


if __name__ == "__main__":
    unittest.main()
