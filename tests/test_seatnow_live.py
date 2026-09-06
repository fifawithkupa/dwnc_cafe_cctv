"""Live (RTSP) reader: continuous decode, trailing burst window, reconnect.

A camera stream cannot be seeked, so the reader keeps one long-lived ffmpeg
process decoding in the background and the tick takes the *most recent*
frames.  Unit tests drive the reader with fake processes; one integration
test decodes a real synthetic file through the real ffmpeg binary.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from engine.seatnow_core import require_ffmpeg
from engine.seatnow_live import (
    FFmpegLiveReader,
    HWACCEL_FAILURE_PATTERNS,
    coherent_tail,
    is_live_source,
    redact_url,
    should_disable_hwaccel,
)


WIDTH, HEIGHT = 4, 3
FRAME_BYTES = WIDTH * HEIGHT * 3


def _frame_payload(value: int) -> bytes:
    return bytes([value % 256]) * FRAME_BYTES


class FakeStream:
    """Yields ``payload`` then either reports EOF or blocks until closed.

    ``pause_after`` inserts one real pause of ``pause_s`` once that many bytes
    have been read, to simulate the gap between two frame bursts.
    """

    def __init__(
        self,
        payload: bytes,
        block_after: bool,
        pause_after: Optional[int] = None,
        pause_s: float = 0.0,
    ) -> None:
        self._buffer = io.BytesIO(payload)
        self._block_after = block_after
        self._closed = threading.Event()
        self._pause_after = pause_after
        self._pause_s = pause_s
        self._served = 0
        self._paused = False

    def read(self, size: int) -> bytes:
        if (
            self._pause_after is not None
            and not self._paused
            and self._served >= self._pause_after
        ):
            self._paused = True
            self._closed.wait(self._pause_s)
        chunk = self._buffer.read(size)
        if chunk:
            self._served += len(chunk)
            return chunk
        if self._block_after:
            self._closed.wait()
        return b""

    def close(self) -> None:
        self._closed.set()


class FakeProcess:
    def __init__(self, stream: FakeStream) -> None:
        self.stdout = stream
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.close()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: Optional[float] = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _spawner(
    scripts: List[tuple],
) -> Callable[[List[str], object], FakeProcess]:
    """Each script is (payload, block_after, stderr_text); last one repeats."""

    calls: List[List[str]] = []

    def spawn(command: List[str], stderr_file) -> FakeProcess:
        index = min(len(calls), len(scripts) - 1)
        calls.append(command)
        payload, block_after, stderr_text = scripts[index]
        if stderr_text:
            stderr_file.write(stderr_text.encode("utf-8"))
            stderr_file.flush()
        return FakeProcess(FakeStream(payload, block_after))

    spawn.calls = calls  # type: ignore[attr-defined]
    return spawn


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class LiveSourceDetectionTests(unittest.TestCase):
    def test_rtsp_urls_are_live_sources(self) -> None:
        self.assertTrue(is_live_source("rtsp://192.168.0.10:554/stream"))
        self.assertTrue(is_live_source("rtsps://cam.local/stream"))
        self.assertTrue(is_live_source("RTSP://CAM/upper"))

    def test_file_paths_are_not_live_sources(self) -> None:
        self.assertFalse(is_live_source("sample_raw/cafe.mov"))
        self.assertFalse(is_live_source(r"C:\videos\cafe.mp4"))
        self.assertFalse(is_live_source("/home/hugo/seatnow/cafe.mov"))


class LiveReaderCommandTests(unittest.TestCase):
    def test_hwaccel_and_tcp_transport_come_before_the_input(self) -> None:
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, hwaccel_args=("-hwaccel", "vaapi")
        )
        command = reader.build_command("ffmpeg")
        input_index = command.index("-i")
        self.assertEqual(command[input_index + 1], "rtsp://cam/stream")
        self.assertLess(command.index("-hwaccel"), input_index)
        self.assertLess(command.index("-rtsp_transport"), input_index)
        self.assertEqual(command[command.index("-rtsp_transport") + 1], "tcp")
        self.assertEqual(command[-1], "pipe:1")
        self.assertIn("bgr24", command)
        self.assertNotIn("-ss", command)

    def test_software_decoding_has_no_hwaccel_flag(self) -> None:
        reader = FFmpegLiveReader("rtsp://cam/stream", WIDTH, HEIGHT)
        self.assertNotIn("-hwaccel", reader.build_command("ffmpeg"))

    def test_burst_selection_converts_only_a_few_frames_per_period(self) -> None:
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, burst_every_frames=150, burst_frames=5
        )
        command = reader.build_command("ffmpeg")
        self.assertIn("-vf", command)
        vf = command[command.index("-vf") + 1]
        self.assertIn("select=", vf)
        self.assertIn("150", vf)
        self.assertIn("5", vf)
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")
        self.assertLess(command.index("-i"), command.index("-vf"))

    def test_no_burst_selection_means_every_frame_is_converted(self) -> None:
        reader = FFmpegLiveReader("rtsp://cam/stream", WIDTH, HEIGHT)
        command = reader.build_command("ffmpeg")
        self.assertNotIn("-vf", command)
        self.assertNotIn("-fps_mode", command)


class LiveReaderBurstTests(unittest.TestCase):
    def test_read_burst_returns_the_newest_frames_with_the_last_as_center(self) -> None:
        payload = b"".join(_frame_payload(i) for i in range(7))
        spawn = _spawner([(payload, True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn, reconnect_delays=(0.0,)
        )
        reader.start()
        try:
            center_index, burst = reader.read_burst(2, timeout=3.0)
        finally:
            reader.close()

        self.assertEqual(len(burst), 5)
        self.assertEqual(center_index, 4)
        values = [int(frame[0, 0, 0]) for _, frame in burst]
        self.assertEqual(values, [2, 3, 4, 5, 6])
        self.assertEqual(burst[0][1].shape, (HEIGHT, WIDTH, 3))
        timestamps = [timestamp for timestamp, _ in burst]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertGreaterEqual(timestamps[0], 0.0)

    def test_read_burst_with_fewer_frames_than_requested_returns_what_arrived(self) -> None:
        payload = b"".join(_frame_payload(i) for i in range(2))
        spawn = _spawner([(payload, True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn, reconnect_delays=(0.0,)
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["frames_decoded"] >= 2))
            center_index, burst = reader.read_burst(2, timeout=0.3)
        finally:
            reader.close()
        self.assertEqual(len(burst), 2)
        self.assertEqual(center_index, 1)

    def test_read_burst_with_no_frames_times_out_empty(self) -> None:
        spawn = _spawner([(b"", True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn, reconnect_delays=(0.0,)
        )
        reader.start()
        try:
            center_index, burst = reader.read_burst(2, timeout=0.2)
        finally:
            reader.close()
        self.assertEqual(burst, [])
        self.assertEqual(center_index, -1)

    def test_partial_trailing_frame_is_dropped_not_returned(self) -> None:
        payload = _frame_payload(1) + _frame_payload(2)[: FRAME_BYTES // 2]
        spawn = _spawner([(payload, False, ""), (b"", True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn, reconnect_delays=(0.0,)
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["reconnects"] >= 1))
            _, burst = reader.read_burst(0, timeout=0.5)
        finally:
            reader.close()
        self.assertEqual(len(burst), 1)
        self.assertEqual(int(burst[0][1][0, 0, 0]), 1)


class CoherentWindowTests(unittest.TestCase):
    def test_full_window_within_span_is_returned(self) -> None:
        frames = [(0.0, "a"), (0.03, "b"), (0.07, "c"), (5.0, "d"), (5.03, "e"), (5.07, "f")]
        self.assertEqual(coherent_tail(frames, 3, 1.0), [(5.0, "d"), (5.03, "e"), (5.07, "f")])

    def test_window_straddling_two_bursts_is_cut_to_the_newest_burst(self) -> None:
        frames = [(0.0, "a"), (0.03, "b"), (0.07, "c"), (5.0, "d"), (5.03, "e")]
        self.assertEqual(coherent_tail(frames, 5, 1.0), [(5.0, "d"), (5.03, "e")])

    def test_no_span_limit_returns_the_newest_frames(self) -> None:
        frames = [(0.0, "a"), (5.0, "b"), (9.0, "c")]
        self.assertEqual(coherent_tail(frames, 2, None), [(5.0, "b"), (9.0, "c")])

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(coherent_tail([], 3, 1.0), [])

    def test_read_burst_with_span_limit_waits_for_the_new_burst_to_complete(self) -> None:
        old = b"".join(_frame_payload(i) for i in range(3))
        new = b"".join(_frame_payload(i) for i in range(10, 15))
        stream = FakeStream(old + new, True, pause_after=len(old), pause_s=0.5)
        spawn = lambda command, stderr_file: FakeProcess(stream)  # noqa: E731
        reader = FFmpegLiveReader("rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn)
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["frames_decoded"] >= 3))
            center_index, burst = reader.read_burst(2, timeout=3.0, max_span_s=0.2)
        finally:
            reader.close()
        self.assertEqual([int(f[0, 0, 0]) for _, f in burst], [10, 11, 12, 13, 14])
        self.assertEqual(center_index, 4)


class StaleFrameTests(unittest.TestCase):
    def test_frames_older_than_max_age_are_treated_as_absent(self) -> None:
        # Real clock plus a jumpable offset: the wait loop needs time to move.
        offset = [0.0]
        payload = b"".join(_frame_payload(i) for i in range(5))
        spawn = _spawner([(payload, True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream",
            WIDTH,
            HEIGHT,
            spawn=spawn,
            clock=lambda: time.monotonic() + offset[0],
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["frames_decoded"] >= 5))
            _, fresh = reader.read_burst(2, timeout=0.1, max_age_s=30.0)
            offset[0] = 45.0  # 45 s later, camera silent
            center_index, stale = reader.read_burst(2, timeout=0.1, max_age_s=30.0)
        finally:
            reader.close()
        self.assertEqual(len(fresh), 5)
        self.assertEqual(stale, [])
        self.assertEqual(center_index, -1)


class HwaccelFallbackTests(unittest.TestCase):
    def test_disable_hwaccel_restarts_ffmpeg_without_the_flag(self) -> None:
        spawn = _spawner([(b"", True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT,
            hwaccel_args=("-hwaccel", "vaapi"), spawn=spawn, reconnect_delays=(0.0,),
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: len(spawn.calls) >= 1))
            self.assertIn("-hwaccel", spawn.calls[0])
            reader.disable_hwaccel()
            self.assertTrue(_wait_until(lambda: len(spawn.calls) >= 2))
        finally:
            reader.close()
        self.assertNotIn("-hwaccel", spawn.calls[-1])
        self.assertEqual(reader.hwaccel_args, ())
        self.assertTrue(reader.stats()["hwaccel_disabled"])

    def test_policy_falls_back_after_repeated_dead_connections_with_hwaccel_errors(self) -> None:
        healthy = {"hwaccel_suspect": False, "reconnects": 0, "frames_decoded": 100, "hwaccel_disabled": False}
        self.assertFalse(should_disable_hwaccel(healthy, frames_at_last_tick=95))
        # reconnecting, no new frames, and ffmpeg blamed the accelerator
        broken = {"hwaccel_suspect": True, "reconnects": 3, "frames_decoded": 100, "hwaccel_disabled": False}
        self.assertTrue(should_disable_hwaccel(broken, frames_at_last_tick=100))
        # frames are still arriving -> not dead, do not switch
        alive = {"hwaccel_suspect": True, "reconnects": 3, "frames_decoded": 130, "hwaccel_disabled": False}
        self.assertFalse(should_disable_hwaccel(alive, frames_at_last_tick=100))
        # already switched -> never again
        done = dict(broken, hwaccel_disabled=True)
        self.assertFalse(should_disable_hwaccel(done, frames_at_last_tick=100))

    def test_patterns_match_what_this_ffmpeg_actually_prints(self) -> None:
        real = (
            "[AVHWDeviceContext @ 0x6131] Failed to initialise VAAPI connection: -1 (unknown libva error).\n"
            "Device creation failed: -5.\n"
            "No device available for decoder: device type vaapi needed for codec hevc.\n"
            "[vist#0:0/hevc @ 0x6131] Hardware device setup failed for decoder: Input/output error\n"
        )
        hits = [
            line for line in real.splitlines()
            if any(p.lower() in line.lower() for p in HWACCEL_FAILURE_PATTERNS)
        ]
        self.assertGreaterEqual(len(hits), 3)
        self.assertTrue(any("Hardware device setup failed" in h for h in hits))


class RedactUrlTests(unittest.TestCase):
    def test_password_is_hidden_but_host_and_path_kept(self) -> None:
        self.assertEqual(
            redact_url("rtsp://admin:S3cret!@192.168.0.50:554/Streaming/Channels/101"),
            "rtsp://admin:***@192.168.0.50:554/Streaming/Channels/101",
        )

    def test_url_without_credentials_is_unchanged(self) -> None:
        self.assertEqual(redact_url("rtsp://192.168.0.50/x"), "rtsp://192.168.0.50/x")
        self.assertEqual(redact_url("sample_raw/cafe.mov"), "sample_raw/cafe.mov")


class LiveReaderReconnectTests(unittest.TestCase):
    def test_stream_end_triggers_reconnect_and_frames_keep_flowing(self) -> None:
        first = b"".join(_frame_payload(i) for i in range(3))
        second = b"".join(_frame_payload(i) for i in range(10, 13))
        spawn = _spawner([(first, False, ""), (second, True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn, reconnect_delays=(0.0,)
        )
        reader.start()
        try:
            self.assertTrue(
                _wait_until(lambda: reader.stats()["frames_decoded"] >= 6)
            )
            stats = reader.stats()
            _, burst = reader.read_burst(1, timeout=1.0)
        finally:
            reader.close()
        self.assertEqual(stats["reconnects"], 1)
        self.assertEqual(len(spawn.calls), 2)
        self.assertEqual([int(f[0, 0, 0]) for _, f in burst], [10, 11, 12])

    def test_reconnect_delay_backs_off_and_caps_at_the_last_value(self) -> None:
        sleeps: List[float] = []
        spawn = _spawner([(b"", False, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream",
            WIDTH,
            HEIGHT,
            spawn=spawn,
            reconnect_delays=(0.01, 0.02, 0.05),
            sleep=lambda seconds: sleeps.append(seconds),
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["reconnects"] >= 5))
        finally:
            reader.close()
        self.assertEqual(sleeps[:5], [0.01, 0.02, 0.05, 0.05, 0.05])

    def test_frames_arriving_reset_the_backoff(self) -> None:
        sleeps: List[float] = []
        spawn = _spawner(
            [(b"", False, ""), (b"", False, ""), (_frame_payload(1), False, ""), (b"", False, "")]
        )
        reader = FFmpegLiveReader(
            "rtsp://cam/stream",
            WIDTH,
            HEIGHT,
            spawn=spawn,
            reconnect_delays=(0.01, 0.02, 0.05),
            sleep=lambda seconds: sleeps.append(seconds),
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["reconnects"] >= 4))
        finally:
            reader.close()
        # two empty connections escalate, a frame resets, the next drop starts over
        self.assertEqual(sleeps[:4], [0.01, 0.02, 0.01, 0.02])


class LiveReaderHwaccelTests(unittest.TestCase):
    def test_hwaccel_failure_on_stderr_is_flagged(self) -> None:
        stderr = "[h264 @ 0x55] Failed to initialise VAAPI connection: -1 (unknown libva error).\n"
        spawn = _spawner([(b"", False, stderr), (b"", True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream",
            WIDTH,
            HEIGHT,
            hwaccel_args=("-hwaccel", "vaapi"),
            spawn=spawn,
            reconnect_delays=(0.0,),
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["hwaccel_suspect"]))
            stats = reader.stats()
        finally:
            reader.close()
        self.assertIn("VAAPI", stats["hwaccel_messages"][0])

    def test_clean_stderr_is_not_flagged(self) -> None:
        spawn = _spawner([(_frame_payload(1), False, "frame=1 fps=0\n"), (b"", True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream",
            WIDTH,
            HEIGHT,
            hwaccel_args=("-hwaccel", "vaapi"),
            spawn=spawn,
            reconnect_delays=(0.0,),
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["reconnects"] >= 1))
            stats = reader.stats()
        finally:
            reader.close()
        self.assertFalse(stats["hwaccel_suspect"])
        self.assertEqual(stats["hwaccel_messages"], [])

    def test_patterns_cover_the_known_ffmpeg_fallback_messages(self) -> None:
        for line in (
            "Failed to initialise VAAPI connection",
            "No device available for decoder: device type vaapi needed for codec h264",
            "Failed setup for format vaapi: hwaccel initialisation returned error",
            "Error creating a QSV session",
        ):
            self.assertTrue(
                any(pattern.lower() in line.lower() for pattern in HWACCEL_FAILURE_PATTERNS),
                line,
            )


class LiveReaderStatsTests(unittest.TestCase):
    def test_stats_report_buffer_and_freshness(self) -> None:
        payload = b"".join(_frame_payload(i) for i in range(4))
        spawn = _spawner([(payload, True, "")])
        reader = FFmpegLiveReader(
            "rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn, buffer_frames=3
        )
        reader.start()
        try:
            self.assertTrue(_wait_until(lambda: reader.stats()["frames_decoded"] >= 4))
            stats = reader.stats()
        finally:
            reader.close()
        self.assertEqual(stats["frames_decoded"], 4)
        self.assertEqual(stats["buffered"], 3)
        self.assertEqual(stats["reconnects"], 0)
        self.assertGreaterEqual(stats["newest_age_s"], 0.0)
        self.assertIsNone(stats["last_error"])

    def test_close_is_idempotent_and_stops_the_thread(self) -> None:
        spawn = _spawner([(b"", True, "")])
        reader = FFmpegLiveReader("rtsp://cam/stream", WIDTH, HEIGHT, spawn=spawn)
        reader.start()
        reader.close()
        reader.close()
        self.assertFalse(reader.is_running())


class LiveReaderRealFfmpegTests(unittest.TestCase):
    """The reader is source-agnostic: a plain file behaves like a stream that
    ends, which exercises decode, framing and reconnect through real ffmpeg."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.ffmpeg, _ = require_ffmpeg()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls._tmp = tempfile.TemporaryDirectory(prefix="seatnow-live-")
        cls.source = Path(cls._tmp.name) / "src.mp4"
        completed = subprocess.run(
            [
                cls.ffmpeg, "-y", "-nostdin", "-v", "error", "-f", "lavfi",
                "-i", "testsrc2=size=160x96:rate=10", "-t", "1",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.source),
            ],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            cls._tmp.cleanup()
            raise RuntimeError(completed.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_decodes_real_frames_and_reconnects_at_end_of_file(self) -> None:
        reader = FFmpegLiveReader(
            str(self.source), 160, 96, reconnect_delays=(0.05,), buffer_frames=8
        )
        reader.start()
        try:
            self.assertTrue(
                _wait_until(lambda: reader.stats()["reconnects"] >= 1, timeout=10.0)
            )
            center_index, burst = reader.read_burst(2, timeout=5.0)
            stats = reader.stats()
        finally:
            reader.close()
        self.assertEqual(len(burst), 5)
        self.assertEqual(center_index, 4)
        self.assertEqual(burst[0][1].shape, (96, 160, 3))
        self.assertGreaterEqual(stats["frames_decoded"], 10)
        self.assertFalse(stats["hwaccel_suspect"])
        self.assertFalse(reader.is_running())

    def test_burst_selection_filter_is_accepted_by_real_ffmpeg(self) -> None:
        # 10 source frames; 2 frames every 5 -> frames 0,1,5,6 = 4 per pass
        reader = FFmpegLiveReader(
            str(self.source), 160, 96, reconnect_delays=(0.05,),
            burst_every_frames=5, burst_frames=2,
        )
        reader.start()
        try:
            self.assertTrue(
                _wait_until(lambda: reader.stats()["reconnects"] >= 1, timeout=10.0)
            )
            stats = reader.stats()
            _, burst = reader.read_burst(0, timeout=2.0, max_span_s=0.5)
        finally:
            reader.close()
        self.assertGreater(stats["frames_decoded"], 0)
        self.assertEqual(stats["frames_decoded"] % 4, 0, stats)
        self.assertEqual(len(burst), 1)
        self.assertEqual(stats["hwaccel_messages"], [])


if __name__ == "__main__":
    unittest.main()
