"""Serve a local video file as a live RTSP stream, for testing without a camera.

plan.md T5: the camera is not bought until the edge bench (T6) fixes the
resolution, but T8's stream reader and T9's Quick Sync decoding have to be
verifiable before then.  This republishes an existing sample video as a real
RTSP stream so the whole pipeline — decode, infer, tick — runs against a live
source that cannot be seeked, exactly like the real camera.

    python rtsp_republish.py sample_raw/cafe.mp4
    # -> rtsp://127.0.0.1:8554/seatnow

    # in another terminal
    ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/seatnow
    python seatnow.py rtsp://127.0.0.1:8554/seatnow      # after T8

Requires an RTSP server.  ``mediamtx`` is the one this project standardises on:

    brew install mediamtx          # macOS
    winget install bluenviron.mediamtx
    # Linux: download from github.com/bluenviron/mediamtx/releases

Pass ``--no-server`` when a server is already running (on this machine or the
edge box) and only the ffmpeg publisher is wanted.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


DEFAULT_PORT = 8554
DEFAULT_PATH = "seatnow"


def require_binary(name: str, hint: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} not found on PATH. {hint}")
    return found


def build_publisher_command(
    ffmpeg: str,
    source: Path,
    url: str,
    loop: bool,
    transcode: bool,
    fps: Optional[float],
) -> List[str]:
    """Build the ffmpeg command that pushes ``source`` to ``url``.

    ``-re`` paces the file at wall-clock speed, which is the whole point: a
    stream published faster than real time would let a seek-based reader keep
    up and hide the very bug T8 exists to fix.
    """
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-re"]
    if loop:
        command += ["-stream_loop", "-1"]
    command += ["-i", str(source)]
    if transcode or fps:
        # A CCTV camera emits a steady H.264 stream with regular keyframes; a
        # sample file may not, and a reader that joins mid-GOP then waits.
        command += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"]
        command += ["-g", "30", "-keyint_min", "30", "-sc_threshold", "0"]
        if fps:
            command += ["-r", str(fps)]
    else:
        command += ["-c:v", "copy"]
    command += ["-an", "-f", "rtsp", "-rtsp_transport", "tcp", url]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Video file to republish")
    parser.add_argument("--host", default="127.0.0.1", help="RTSP host to publish to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="RTSP port")
    parser.add_argument("--path", default=DEFAULT_PATH, help="RTSP stream path")
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Publish once instead of looping forever",
    )
    parser.add_argument(
        "--transcode",
        action="store_true",
        help="Re-encode with regular keyframes instead of copying the stream",
    )
    parser.add_argument("--fps", type=float, help="Force an output frame rate (implies --transcode)")
    parser.add_argument(
        "--no-server",
        dest="start_server",
        action="store_false",
        help="Do not start mediamtx; publish to an already-running server",
    )
    parser.add_argument(
        "--server-startup-seconds",
        type=float,
        default=1.5,
        help="How long to wait for mediamtx to bind its port",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.source.exists():
        raise FileNotFoundError(f"Source video not found: {args.source}")

    ffmpeg = require_binary(
        "ffmpeg", "Install it first (see ONBOARDING.md)."
    )
    url = f"rtsp://{args.host}:{args.port}/{args.path}"

    server: Optional[subprocess.Popen] = None
    if args.start_server:
        mediamtx = require_binary(
            "mediamtx",
            "Install it (brew install mediamtx) or pass --no-server to publish "
            "to a server you started yourself.",
        )
        print(f"Starting RTSP server: {mediamtx}", flush=True)
        server = subprocess.Popen(
            [mediamtx],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(args.server_startup_seconds)
        if server.poll() is not None:
            raise RuntimeError(
                "mediamtx exited immediately — port "
                f"{args.port} is probably already in use. "
                "Use --no-server if a server is already running."
            )

    command = build_publisher_command(
        ffmpeg, args.source, url, args.loop, args.transcode, args.fps
    )
    print(f"Publishing {args.source} -> {url}", flush=True)
    print(f"  {' '.join(command)}\n", flush=True)
    print(f"Receive with:  ffplay -rtsp_transport tcp {url}", flush=True)
    print("Ctrl-C to stop.\n", flush=True)

    publisher = subprocess.Popen(command)
    try:
        return publisher.wait()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
        return 0
    finally:
        for process in (publisher, server):
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    sys.exit(main())
