"""Read a live camera stream (RTSP) for the tick loop.

A camera cannot be seeked, so the file readers' "jump to t, decode 2n+1
frames" shape does not apply.  This reader keeps one ffmpeg process decoding
continuously in a background thread into a small ring buffer, and a tick
takes the *newest* 2n+1 frames.  The window is therefore trailing (past
frames only) rather than centred: on a live stream there is no future frame
to vote with (docs/superpowers/specs/2026-09-01-detection-inspection-harness-design.md §7).

Two things the file path never had to worry about:

* The stream drops.  Cameras reboot, switches hiccup, Wi-Fi bridges flap.
  When ffmpeg exits the reader waits with a capped backoff and reconnects;
  frames already buffered stay available so the next tick still judges.
* Hardware decoding can silently stop being hardware.  ffmpeg logs the
  failure as a *warning* and carries on in software, so the process exit
  code says nothing.  The reader keeps ffmpeg's stderr and flags known
  fallback messages in ``stats()``; the tick loop prints that loudly and
  writes it into every record (plan.md §2 "T8에 딸려 갈 것").
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np

from engine.seatnow_core import (
    _read_exact,
    _read_stderr_file,
    _wait_process,
    require_ffmpeg,
)


LIVE_SCHEMES: Tuple[str, ...] = ("rtsp://", "rtsps://")

# Substrings ffmpeg prints (as warnings) when an accelerator was requested but
# decoding fell back to software.  Matched case-insensitively.
HWACCEL_FAILURE_PATTERNS: Tuple[str, ...] = (
    "Failed to initialise VAAPI",
    "No device available for decoder",
    "hwaccel initialisation returned error",
    "Failed setup for format",
    "Error creating a QSV session",
    "Error initializing an internal MFX session",
    "falling back to software",
)

# Socket I/O timeout for the RTSP demuxer.  A camera that stalls without
# closing the connection would otherwise block the reader forever; with this
# ffmpeg exits and the reconnect loop takes over.
RTSP_SOCKET_TIMEOUT_US = 10_000_000

Frame = Tuple[float, np.ndarray]
Spawn = Callable[[List[str], object], object]


def is_live_source(value: object) -> bool:
    """True for stream URLs the live reader handles (RTSP), False for files."""
    return str(value).lower().startswith(LIVE_SCHEMES)


def _default_spawn(command: List[str], stderr_file) -> subprocess.Popen:
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)


class FFmpegLiveReader:
    """Continuous decoder with a trailing-window ``read_burst``.

    ``width``/``height`` must be known up front (probe the stream first):
    rawvideo frames are fixed-size and the reader frames the pipe by size.
    """

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        *,
        hwaccel_args: Sequence[str] = (),
        buffer_frames: int = 8,
        rtsp_transport: str = "tcp",
        reconnect_delays: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 15.0),
        spawn: Optional[Spawn] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], None]] = None,
        ffmpeg: Optional[str] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if buffer_frames < 1:
            raise ValueError("buffer_frames must be at least 1")
        if not reconnect_delays:
            raise ValueError("reconnect_delays cannot be empty")
        self.url = str(url)
        self.width = int(width)
        self.height = int(height)
        self.hwaccel_args = tuple(hwaccel_args)
        self.rtsp_transport = rtsp_transport
        self.reconnect_delays = tuple(float(d) for d in reconnect_delays)
        self._spawn: Spawn = spawn or _default_spawn
        self._clock = clock
        self._sleep = sleep
        self._ffmpeg = ffmpeg

        self._frame_bytes = self.width * self.height * 3
        self._buffer: Deque[Frame] = deque(maxlen=buffer_frames)
        self._arrivals: Deque[float] = deque(maxlen=600)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[object] = None
        self._t0 = self._clock()

        self._frames_decoded = 0
        self._reconnects = 0
        self._last_error: Optional[str] = None
        self._hwaccel_suspect = False
        self._hwaccel_messages: List[str] = []
        self._connected = False

    # ----------------------------------------------------------------- command

    def build_command(self, ffmpeg: str) -> List[str]:
        """ffmpeg invocation: accelerator and transport flags before ``-i``."""
        command: List[str] = [ffmpeg, "-nostdin", "-v", "warning", *self.hwaccel_args]
        if is_live_source(self.url):
            command += [
                "-rtsp_transport",
                self.rtsp_transport,
                "-timeout",
                str(RTSP_SOCKET_TIMEOUT_US),
            ]
        command += [
            "-i",
            self.url,
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        return command

    # --------------------------------------------------------------- lifecycle

    def start(self) -> "FFmpegLiveReader":
        if self._thread is not None:
            return self
        self._t0 = self._clock()
        self._thread = threading.Thread(
            target=self._run, name="seatnow-live-reader", daemon=True
        )
        self._thread.start()
        return self

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:  # type: ignore[attr-defined]
                    process.terminate()  # type: ignore[attr-defined]
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)

    def __enter__(self) -> "FFmpegLiveReader":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------- reads

    def read_burst(self, n: int, timeout: float = 5.0) -> Tuple[int, List[Frame]]:
        """Newest ``2n+1`` frames, oldest first; the last one is the centre.

        Waits up to ``timeout`` for the buffer to hold that many.  If fewer
        arrived, returns what there is (still newest-last); if nothing has
        arrived, returns ``(-1, [])`` so the caller can skip the tick.
        """
        if n < 0:
            raise ValueError("n cannot be negative")
        wanted = 2 * n + 1
        deadline = self._clock() + max(0.0, timeout)
        with self._condition:
            while len(self._buffer) < wanted and not self._stop.is_set():
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.05))
            snapshot = list(self._buffer)[-wanted:]
        if not snapshot:
            return -1, []
        return len(snapshot) - 1, snapshot

    def stats(self) -> dict:
        now = self._clock()
        with self._condition:
            buffered = len(self._buffer)
            newest = self._buffer[-1][0] if self._buffer else None
            arrivals = list(self._arrivals)
            frames_decoded = self._frames_decoded
            reconnects = self._reconnects
            last_error = self._last_error
            suspect = self._hwaccel_suspect
            messages = list(self._hwaccel_messages)
            connected = self._connected
        window = 5.0
        recent = [t for t in arrivals if now - t <= window]
        decode_fps = (len(recent) / window) if len(recent) > 1 else 0.0
        return {
            "frames_decoded": frames_decoded,
            "buffered": buffered,
            "reconnects": reconnects,
            "connected": connected,
            "newest_age_s": (now - self._t0 - newest) if newest is not None else None,
            "decode_fps": round(decode_fps, 2),
            "last_error": last_error,
            "hwaccel_suspect": suspect,
            "hwaccel_messages": messages,
            "uptime_s": now - self._t0,
        }

    # ------------------------------------------------------------ background

    def _pause(self, seconds: float) -> None:
        if self._sleep is not None:
            self._sleep(seconds)
        else:
            self._stop.wait(seconds)

    def _scan_stderr(self, text: str) -> None:
        if not text:
            return
        hits: List[str] = []
        for line in text.splitlines():
            lowered = line.lower()
            if any(pattern.lower() in lowered for pattern in HWACCEL_FAILURE_PATTERNS):
                hits.append(line.strip())
        if not hits:
            return
        with self._condition:
            self._hwaccel_suspect = True
            for line in hits:
                if len(self._hwaccel_messages) >= 10:
                    break
                if line not in self._hwaccel_messages:
                    self._hwaccel_messages.append(line)

    def _run(self) -> None:
        ffmpeg = self._ffmpeg
        if ffmpeg is None and self._spawn is _default_spawn:
            ffmpeg, _ = require_ffmpeg()
        command = self.build_command(ffmpeg or "ffmpeg")
        attempt = 0
        while not self._stop.is_set():
            stderr_file = tempfile.TemporaryFile(mode="w+b")
            process = None
            try:
                process = self._spawn(command, stderr_file)
            except OSError as exc:
                with self._condition:
                    self._last_error = str(exc)
            if process is not None:
                self._process = process
                got_frames = self._pump(process)
                self._process = None
                if got_frames:
                    attempt = 0
                self._reap(process)
            try:
                stderr_text = _read_stderr_file(stderr_file).decode("utf-8", errors="replace")
            finally:
                stderr_file.close()
            self._scan_stderr(stderr_text)
            with self._condition:
                self._connected = False
                if stderr_text.strip():
                    self._last_error = stderr_text.strip().splitlines()[-1][:300]
            if self._stop.is_set():
                break
            with self._condition:
                self._reconnects += 1
            delay = self.reconnect_delays[min(attempt, len(self.reconnect_delays) - 1)]
            attempt += 1
            self._pause(delay)

    def _pump(self, process) -> bool:
        """Move frames from the process into the buffer until it ends."""
        stream = process.stdout
        got_frames = False
        while not self._stop.is_set():
            payload = _read_exact(stream, self._frame_bytes)
            if len(payload) != self._frame_bytes:
                break  # EOF, or a torn frame at the end of a dropped stream
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
            now = self._clock()
            with self._condition:
                self._buffer.append((now - self._t0, frame))
                self._arrivals.append(now)
                self._frames_decoded += 1
                self._connected = True
                self._condition.notify_all()
            got_frames = True
        return got_frames

    @staticmethod
    def _reap(process) -> None:
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass
        try:
            _wait_process(process, timeout=5.0)
        except Exception:
            pass
        stream = getattr(process, "stdout", None)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def child_pid(self) -> Optional[int]:
        process = self._process
        return getattr(process, "pid", None) if process is not None else None


# ------------------------------------------------------------------ schedule


class TickSchedule:
    """Wall-clock tick slots: t0, t0+interval, t0+2·interval, …

    A tick that overruns its slot does not shift the grid; the missed slots
    are skipped and counted, so a 24/7 log shows *when* the box fell behind
    instead of silently drifting.
    """

    def __init__(self, interval_seconds: float, clock: Callable[[], float] = time.monotonic):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval = float(interval_seconds)
        self._clock = clock
        self.t0 = clock()
        self.scheduled = self.t0
        self.skipped = 0
        self.ticks = 0

    def wait_seconds(self) -> float:
        return max(0.0, self.scheduled - self._clock())

    def late_seconds(self) -> float:
        return max(0.0, self._clock() - self.scheduled)

    def advance(self) -> None:
        self.ticks += 1
        self.scheduled += self.interval
        now = self._clock()
        while self.scheduled < now:
            self.scheduled += self.interval
            self.skipped += 1


# --------------------------------------------------------------------- probe


def build_probe_command(ffprobe: str, url: str) -> List[str]:
    command = [ffprobe, "-v", "error"]
    if is_live_source(url):
        command += ["-rtsp_transport", "tcp"]
    command += [
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        "-i",
        str(url),
    ]
    return command


def probe_stream(
    url: str,
    *,
    attempts: int = 5,
    delays: Sequence[float] = (2.0, 4.0, 8.0, 15.0),
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
):
    """Width/height/fps/codec of a live stream, retrying while the camera boots."""
    import json as _json

    from engine.seatnow_core import VideoInfo, _parse_rate

    _, ffprobe = require_ffmpeg()
    command = build_probe_command(ffprobe, url)
    last_error = "unknown"
    for attempt in range(max(1, attempts)):
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_error = f"ffprobe timed out after {timeout:.0f}s"
        else:
            if completed.returncode == 0:
                payload = _json.loads(completed.stdout or "{}")
                streams = payload.get("streams") or []
                if streams:
                    stream = streams[0]
                    return VideoInfo(
                        width=int(stream["width"]),
                        height=int(stream["height"]),
                        fps=_parse_rate(stream.get("avg_frame_rate", "0/0")),
                        duration=0.0,
                        codec=str(stream.get("codec_name") or "unknown"),
                        source_frames=None,
                    )
                last_error = "no video stream in probe output"
            else:
                last_error = (completed.stderr or "").strip().splitlines()[-1:] or ["ffprobe failed"]
                last_error = last_error[0]
        if attempt + 1 < attempts:
            sleep(delays[min(attempt, len(delays) - 1)])
    raise RuntimeError(f"스트림을 열 수 없습니다: {url} — {last_error}")


# -------------------------------------------------------------------- memory


def process_rss_mb(pid: Optional[int] = None) -> Optional[float]:
    """Resident memory of a process in MB (own process when ``pid`` is None)."""
    import os
    import sys as _sys

    target = os.getpid() if pid is None else int(pid)
    statm = f"/proc/{target}/statm"
    try:
        with open(statm, "r", encoding="ascii") as handle:
            fields = handle.read().split()
        resident_pages = int(fields[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError, AttributeError):
        pass
    if _sys.platform.startswith("win"):
        return _windows_rss_mb(target)
    if pid is None:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024 * 1024 if _sys.platform == "darwin" else 1024
            return usage / divisor
        except Exception:
            return None
    return None


def _windows_rss_mb(pid: int) -> Optional[float]:
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize / (1024 * 1024)
    finally:
        kernel32.CloseHandle(handle)


# ------------------------------------------------------------- hwaccel probe


def build_hwaccel_probe_command(ffmpeg: str, name: str, url: str) -> List[str]:
    """Decode exactly one frame of the stream through ``name``.

    Mirrors ``seatnow_hwaccel.probe_hwaccel`` but speaks RTSP over TCP with a
    socket timeout, so a probe against a camera neither hangs nor fails for
    a transport reason and gets blamed on the accelerator.
    """
    command = [ffmpeg, "-nostdin", "-v", "error", "-hwaccel", name]
    if is_live_source(url):
        command += ["-rtsp_transport", "tcp", "-timeout", str(RTSP_SOCKET_TIMEOUT_US)]
    command += [
        "-i",
        str(url),
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    return command


def probe_live_hwaccel(name: str, url: object, ffmpeg: Optional[str] = None) -> bool:
    """True only if a real frame came out of the accelerator (exit code lies)."""
    binary = ffmpeg
    if binary is None:
        try:
            binary, _ = require_ffmpeg()
        except RuntimeError:
            return False
    command = build_hwaccel_probe_command(binary, name, str(url))
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=45.0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and len(completed.stdout) > 0
