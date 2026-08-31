"""Pick a hardware video decoder, and be honest when there isn't one.

The pilot box is a used mini-PC with an Intel iGPU, so Quick Sync is the
decoder we care about.  Decoding runs 24/7 while inference only runs once
per tick, which makes this the cost that decides how many camera pixels the
box can afford.

ffmpeg will happily warn about a failed accelerator and keep decoding in
software.  A benchmark that believes the flag instead of the frames reports
software numbers as hardware ones, so every candidate here is accepted only
after a real frame comes out of it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


HWACCEL_AUTO = "auto"
HWACCEL_NONE = "none"
HWACCEL_CHOICES: Tuple[str, ...] = (
    HWACCEL_AUTO,
    HWACCEL_NONE,
    "qsv",
    "vaapi",
    "d3d11va",
    "dxva2",
    "videotoolbox",
)

# Best first.  Quick Sync leads on both desktop platforms because it is the
# decoder the mini-PC was chosen for; the others are fallbacks that still
# beat software decoding.
_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "win32": ("qsv", "d3d11va", "dxva2"),
    "linux": ("qsv", "vaapi"),
    "darwin": ("videotoolbox",),
}

_PROBE_CACHE: Dict[Tuple[str, str], bool] = {}


def candidate_order(platform_name: str) -> Tuple[str, ...]:
    """Hardware decoders worth trying on this OS, best first."""
    for prefix, candidates in _CANDIDATES.items():
        if platform_name.startswith(prefix):
            return candidates
    return ()


def hwaccel_input_args(name: str) -> Tuple[str, ...]:
    """ffmpeg arguments that must appear before ``-i``."""
    if name == HWACCEL_NONE:
        return ()
    if name == HWACCEL_AUTO:
        raise ValueError("'auto' must be resolved before building ffmpeg arguments")
    return ("-hwaccel", name)


@dataclass(frozen=True)
class HwaccelChoice:
    """What was asked for, what was probed, and what is actually in use."""

    name: str
    requested: str
    args: Tuple[str, ...]
    fallback: bool
    tried: Tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.name != HWACCEL_NONE

    def describe(self) -> str:
        if self.enabled:
            return f"하드웨어 디코딩 켜짐 ({self.name})"
        if self.fallback:
            tried = ", ".join(self.tried) if self.tried else "후보 없음"
            return (
                "하드웨어 디코딩 꺼짐 — 소프트웨어로 돌립니다. "
                f"시도한 것: {tried}"
            )
        return "하드웨어 디코딩 꺼짐 (소프트웨어 디코딩을 직접 지정함)"


def probe_hwaccel(name: str, sample_path: Path, ffmpeg: Optional[str] = None) -> bool:
    """Decode exactly one frame through ``name``; True only if a frame came out.

    Returning frames is the test, not the exit code: ffmpeg can report success
    for a run that quietly fell back to software.
    """
    key = (name, str(sample_path))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    binary = ffmpeg or shutil.which("ffmpeg")
    if not binary:
        _PROBE_CACHE[key] = False
        return False
    command = [
        binary,
        "-nostdin",
        "-v",
        "error",
        "-hwaccel",
        name,
        "-i",
        str(sample_path),
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
    try:
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=30.0
        )
    except (OSError, subprocess.TimeoutExpired):
        _PROBE_CACHE[key] = False
        return False
    result = completed.returncode == 0 and len(completed.stdout) > 0
    _PROBE_CACHE[key] = result
    return result


def reset_hwaccel_cache() -> None:
    """Forget probe results (tests, and re-probing a different input)."""
    _PROBE_CACHE.clear()


def resolve_hwaccel(
    requested: str,
    sample_path: Path,
    platform_name: str = sys.platform,
    prober: Callable[[str, Path], bool] = probe_hwaccel,
) -> HwaccelChoice:
    """Turn a ``--hwaccel`` value into arguments that are known to work."""
    if requested == HWACCEL_NONE:
        return HwaccelChoice(
            name=HWACCEL_NONE, requested=requested, args=(), fallback=False, tried=()
        )
    if requested == HWACCEL_AUTO:
        candidates = candidate_order(platform_name)
    else:
        candidates = (requested,)

    tried: List[str] = []
    for candidate in candidates:
        tried.append(candidate)
        if prober(candidate, sample_path):
            return HwaccelChoice(
                name=candidate,
                requested=requested,
                args=hwaccel_input_args(candidate),
                fallback=False,
                tried=tuple(tried),
            )
    return HwaccelChoice(
        name=HWACCEL_NONE,
        requested=requested,
        args=(),
        fallback=True,
        tried=tuple(tried),
    )
