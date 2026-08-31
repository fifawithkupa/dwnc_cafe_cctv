# 엣지 디코딩 측정 + 하드웨어 디코딩 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 엣지 박스에서 "어떤 카메라를 살 것인가"에 답하는 디코딩 측정 도구를 만들고, 그 측정이 의미를 갖도록 하드웨어 디코딩(Quick Sync)을 파이프라인에 넣는다.

**Architecture:** 하드웨어 디코딩 선택 로직을 신규 순수 모듈 `seatnow_hwaccel.py`로 분리한다 (`seatnow_core.py`는 이미 3,469줄이고, `seatnow_report.py` 분리 선례를 따른다). 리더는 ffmpeg 명령을 만드는 메서드를 밖으로 빼서 ffmpeg 없이도 명령 구성을 검증할 수 있게 한다. 측정은 ffmpeg의 `-benchmark`가 보고하는 CPU 시간을 파싱해서 쓰므로 새 의존성이 없다.

**Tech Stack:** Python 3.9~3.11, `unittest`(표준 라이브러리), ffmpeg/ffprobe 시스템 바이너리, numpy<2.

## Global Constraints

- 스펙: `docs/superpowers/specs/2026-08-31-edge-decode-bench-design.md`
- 프로젝트 규칙: `CLAUDE.md` — 카메라는 우리가 1대 설치, 3상태 정직 출력, **설명은 비개발자 기준**
- Python 3.9 호환: `Optional[X]` / `Tuple[X, ...]`를 쓰고 `X | Y` 문법을 쓰지 않는다
- 테스트는 `unittest`. 실행: `./venv/Scripts/python.exe -m unittest discover tests`
- **기존 테스트 229개는 계속 통과해야 한다.** 리더의 기존 호출부(`bench.py:85`, `make_labels.py:75`, `seatnow.py:397`·`:469`)는 인자를 안 넘겨도 지금과 똑같이 동작해야 한다
- 새 파이썬 의존성을 추가하지 않는다 (psutil 금지 — ffmpeg `-benchmark`로 충분)
- **하드웨어 디코딩이 안 켜졌으면 안 켜졌다고 출력에 찍는다.** 조용한 폴백 금지
- 사용자 대면 출력 문구는 한국어. 코드 주석·docstring은 기존 파일들과 같이 영어
- 생성한 벤치 클립(`bench/clips/`)은 `.gitignore` 대상

---

### Task 1: `seatnow_hwaccel.py` — 하드웨어 디코딩 선택 로직

**Files:**
- Create: `seatnow_hwaccel.py`
- Test: `tests/test_hwaccel.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 없음 (표준 라이브러리만)
- Produces:
  - `HWACCEL_CHOICES: Tuple[str, ...]` — argparse `choices`용
  - `candidate_order(platform_name: str) -> Tuple[str, ...]`
  - `hwaccel_input_args(name: str) -> Tuple[str, ...]`
  - `class HwaccelChoice` — `.name`, `.requested`, `.args: Tuple[str, ...]`, `.fallback: bool`, `.tried: Tuple[str, ...]`, `.enabled: bool`, `.describe() -> str`
  - `resolve_hwaccel(requested, sample_path, platform_name=sys.platform, prober=probe_hwaccel) -> HwaccelChoice`
  - `probe_hwaccel(name, sample_path, ffmpeg=None) -> bool`
  - `reset_hwaccel_cache() -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_hwaccel.py`:

```python
"""Unit tests for hardware-decode selection.

These never invoke ffmpeg: the probe is injected so the decision logic can
be tested on every platform, including ones with no accelerator at all.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List, Optional, Tuple

from seatnow_hwaccel import (
    HWACCEL_CHOICES,
    HwaccelChoice,
    candidate_order,
    hwaccel_input_args,
    resolve_hwaccel,
)


SAMPLE = Path("sample.mp4")


def prober_accepting(*accepted: str):
    """Return a probe that succeeds only for the named accelerators."""
    seen: List[str] = []

    def probe(name: str, sample_path: Path) -> bool:
        seen.append(name)
        return name in accepted

    probe.seen = seen  # type: ignore[attr-defined]
    return probe


class CandidateOrderTests(unittest.TestCase):
    def test_windows_prefers_quick_sync(self):
        self.assertEqual(candidate_order("win32"), ("qsv", "d3d11va", "dxva2"))

    def test_linux_prefers_quick_sync_then_vaapi(self):
        self.assertEqual(candidate_order("linux"), ("qsv", "vaapi"))

    def test_macos_uses_videotoolbox(self):
        self.assertEqual(candidate_order("darwin"), ("videotoolbox",))

    def test_unknown_platform_has_no_candidates(self):
        self.assertEqual(candidate_order("freebsd14"), ())


class InputArgsTests(unittest.TestCase):
    def test_none_produces_no_arguments(self):
        self.assertEqual(hwaccel_input_args("none"), ())

    def test_named_accelerator_becomes_hwaccel_flag(self):
        self.assertEqual(hwaccel_input_args("qsv"), ("-hwaccel", "qsv"))

    def test_auto_is_not_an_ffmpeg_value(self):
        with self.assertRaises(ValueError):
            hwaccel_input_args("auto")


class ResolveTests(unittest.TestCase):
    def test_auto_picks_the_first_working_candidate(self):
        probe = prober_accepting("d3d11va")
        choice = resolve_hwaccel("auto", SAMPLE, platform_name="win32", prober=probe)
        self.assertEqual(choice.name, "d3d11va")
        self.assertEqual(choice.args, ("-hwaccel", "d3d11va"))
        self.assertFalse(choice.fallback)
        self.assertEqual(probe.seen, ["qsv", "d3d11va"])

    def test_auto_falls_back_to_software_and_says_so(self):
        probe = prober_accepting()
        choice = resolve_hwaccel("auto", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "none")
        self.assertEqual(choice.args, ())
        self.assertTrue(choice.fallback)
        self.assertEqual(choice.tried, ("qsv", "vaapi"))

    def test_none_never_probes(self):
        probe = prober_accepting("qsv")
        choice = resolve_hwaccel("none", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "none")
        self.assertFalse(choice.fallback)
        self.assertEqual(probe.seen, [])

    def test_explicit_request_is_verified_not_trusted(self):
        probe = prober_accepting("vaapi")
        choice = resolve_hwaccel("qsv", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "none")
        self.assertTrue(choice.fallback)
        self.assertEqual(probe.seen, ["qsv"])

    def test_explicit_request_that_works_is_used(self):
        probe = prober_accepting("qsv")
        choice = resolve_hwaccel("qsv", SAMPLE, platform_name="linux", prober=probe)
        self.assertEqual(choice.name, "qsv")
        self.assertFalse(choice.fallback)

    def test_every_choice_is_offered_to_argparse(self):
        self.assertIn("auto", HWACCEL_CHOICES)
        self.assertIn("none", HWACCEL_CHOICES)
        for name in ("qsv", "vaapi", "d3d11va", "dxva2", "videotoolbox"):
            self.assertIn(name, HWACCEL_CHOICES)


class DescribeTests(unittest.TestCase):
    def test_software_fallback_is_reported_as_off(self):
        choice = HwaccelChoice(
            name="none", requested="auto", args=(), fallback=True, tried=("qsv",)
        )
        message = choice.describe()
        self.assertIn("꺼짐", message)
        self.assertIn("qsv", message)

    def test_enabled_accelerator_is_reported_as_on(self):
        choice = HwaccelChoice(
            name="qsv", requested="auto", args=("-hwaccel", "qsv"), fallback=False,
            tried=("qsv",),
        )
        self.assertIn("켜짐", choice.describe())
        self.assertIn("qsv", choice.describe())

    def test_explicit_software_is_not_a_fallback_message(self):
        choice = HwaccelChoice(
            name="none", requested="none", args=(), fallback=False, tried=()
        )
        self.assertIn("소프트웨어", choice.describe())
        self.assertNotIn("실패", choice.describe())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m unittest tests.test_hwaccel -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'seatnow_hwaccel'`

- [ ] **Step 3: Write minimal implementation**

`seatnow_hwaccel.py`:

```python
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
from typing import Callable, Dict, Optional, Tuple


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

    tried = []
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m unittest tests.test_hwaccel -v`
Expected: PASS — `Ran 15 tests ... OK`

- [ ] **Step 5: Add `bench/clips/` to `.gitignore`**

Append to `.gitignore`:

```
bench/clips/
bench/*.json
```

- [ ] **Step 6: Run the whole suite**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: `Ran 244 tests ... OK` (229 + 15)

- [ ] **Step 7: Commit**

```bash
git add seatnow_hwaccel.py tests/test_hwaccel.py .gitignore
git commit -m "feat: 하드웨어 디코딩 선택 로직 - 프레임이 나온 것만 채택"
```

---

### Task 2: 리더에 하드웨어 디코딩 배선

**Files:**
- Modify: `seatnow_core.py:2879-2921` (`FFmpegSampleReader`), `seatnow_core.py:2975-3019` (`FFmpegBurstReader`)
- Test: `tests/test_video_io.py`

**Interfaces:**
- Consumes: Task 1의 `hwaccel_input_args`
- Produces:
  - `FFmpegSampleReader(path, sample_seconds, info=None, start_seconds=0.0, hwaccel_args=())`
  - `FFmpegSampleReader.build_command(ffmpeg: str) -> List[str]`
  - `FFmpegBurstReader(path, info=None, hwaccel_args=())`
  - `FFmpegBurstReader.build_command(ffmpeg: str, start_seconds: float, frame_count: int) -> List[str]`

`hwaccel_args`를 기본값 `()`로 두는 것이 하위 호환의 핵심이다. 기존 호출부 4곳은 수정 없이 지금과 완전히 같은 명령을 만든다.

- [ ] **Step 1: Write the failing test**

`tests/test_video_io.py` 맨 아래에 추가 (기존 `import` 줄에 아무것도 더하지 않아도 되도록 클래스 안에서 필요한 것만 쓴다):

```python
class HwaccelCommandTests(unittest.TestCase):
    """Command construction only — these never launch ffmpeg."""

    def _info(self):
        from seatnow_core import VideoInfo

        return VideoInfo(
            width=1920, height=1080, fps=30.0, duration=60.0,
            codec="h264", source_frames=1800,
        )

    def test_sample_reader_defaults_to_software_decoding(self):
        reader = FFmpegSampleReader(
            Path("clip.mp4"), sample_seconds=1.0, info=self._info()
        )
        command = reader.build_command("ffmpeg")
        self.assertNotIn("-hwaccel", command)

    def test_sample_reader_puts_hwaccel_before_the_input(self):
        reader = FFmpegSampleReader(
            Path("clip.mp4"), sample_seconds=1.0, info=self._info(),
            hwaccel_args=("-hwaccel", "qsv"),
        )
        command = reader.build_command("ffmpeg")
        self.assertLess(command.index("-hwaccel"), command.index("-i"))
        self.assertEqual(command[command.index("-hwaccel") + 1], "qsv")

    def test_burst_reader_defaults_to_software_decoding(self):
        reader = FFmpegBurstReader(Path("clip.mp4"), info=self._info())
        command = reader.build_command("ffmpeg", start_seconds=1.5, frame_count=5)
        self.assertNotIn("-hwaccel", command)
        self.assertIn("-ss", command)

    def test_burst_reader_puts_hwaccel_before_the_input(self):
        reader = FFmpegBurstReader(
            Path("clip.mp4"), info=self._info(), hwaccel_args=("-hwaccel", "vaapi"),
        )
        command = reader.build_command("ffmpeg", start_seconds=1.5, frame_count=5)
        self.assertLess(command.index("-hwaccel"), command.index("-i"))
        self.assertEqual(command[command.index("-hwaccel") + 1], "vaapi")

    def test_burst_reader_keeps_input_seeking(self):
        """-ss before -i is input seeking; moving it would change behaviour."""
        reader = FFmpegBurstReader(
            Path("clip.mp4"), info=self._info(), hwaccel_args=("-hwaccel", "qsv"),
        )
        command = reader.build_command("ffmpeg", start_seconds=1.5, frame_count=5)
        self.assertLess(command.index("-ss"), command.index("-i"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m unittest tests.test_video_io.HwaccelCommandTests -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'hwaccel_args'`

- [ ] **Step 3: Write the implementation**

`seatnow_core.py`, `FFmpegSampleReader.__init__` 시그니처와 본문에 한 줄 추가:

```python
    def __init__(
        self,
        path: Path,
        sample_seconds: float,
        info: Optional[VideoInfo] = None,
        start_seconds: float = 0.0,
        hwaccel_args: Sequence[str] = (),
    ):
        if sample_seconds <= 0:
            raise ValueError("sample_seconds must be positive")
        if start_seconds < 0:
            raise ValueError("start_seconds cannot be negative")
        self.path = Path(path)
        self.sample_seconds = float(sample_seconds)
        self.start_seconds = float(start_seconds)
        self.info = info or probe_video(self.path)
        self.hwaccel_args = tuple(hwaccel_args)
        self.process: Optional[subprocess.Popen] = None
        self.stderr_file = None

    def build_command(self, ffmpeg: str) -> List[str]:
        """Assemble the decode command.  Hardware flags are input options,
        so they must precede ``-i``."""
        return [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            *self.hwaccel_args,
            "-i",
            str(self.path),
            "-ss",
            f"{self.start_seconds:.9f}",
            "-vf",
            f"fps=1/{self.sample_seconds:.9f}",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
```

그리고 `__iter__`의 인라인 `command = [...]` 블록(현재 `seatnow_core.py:2900-2920`)을 아래 두 줄로 교체한다:

```python
        ffmpeg, _ = require_ffmpeg()
        command = self.build_command(ffmpeg)
```

`FFmpegBurstReader`도 같은 방식으로:

```python
    def __init__(
        self,
        path: Path,
        info: Optional[VideoInfo] = None,
        hwaccel_args: Sequence[str] = (),
    ):
        self.path = Path(path)
        self.info = info or probe_video(self.path)
        self.hwaccel_args = tuple(hwaccel_args)
        if self.info.fps <= 0:
            raise ValueError("Video fps must be positive for burst reading")

    def build_command(
        self, ffmpeg: str, start_seconds: float, frame_count: int
    ) -> List[str]:
        """Assemble the burst command.  ``-ss`` stays before ``-i`` (input
        seeking) and the hardware flags go in front of both."""
        return [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            *self.hwaccel_args,
            "-ss",
            f"{start_seconds:.9f}",
            "-i",
            str(self.path),
            "-frames:v",
            str(frame_count),
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
```

`read_burst` 안의 인라인 `command = [...]` 블록(현재 `seatnow_core.py:2999-3019`)을 교체:

```python
        ffmpeg, _ = require_ffmpeg()
        frame_interval = 1.0 / self.info.fps
        start_seconds = max(0.0, center_seconds - n * frame_interval)
        frame_count = 2 * n + 1
        command = self.build_command(ffmpeg, start_seconds, frame_count)
```

`Sequence`가 `seatnow_core.py`의 `typing` import에 없으면 추가한다.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m unittest tests.test_video_io -v`
Expected: PASS — 기존 통합 테스트 11개 + 신규 5개 전부 통과

- [ ] **Step 5: Run the whole suite (하위 호환 확인)**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: `Ran 249 tests ... OK`

- [ ] **Step 6: Commit**

```bash
git add seatnow_core.py tests/test_video_io.py
git commit -m "feat: 리더가 하드웨어 디코딩 인자를 받도록 명령 조립 분리"
```

---

### Task 3: `seatnow.py --hwaccel`

**Files:**
- Modify: `seatnow.py:18-33` (imports), `seatnow.py:99` 부근 (argparse), `seatnow.py:390-400`·`:465-470` (리더 생성)
- Test: 없음 (CLI 배선. Task 1·2가 로직을 덮는다)

**Interfaces:**
- Consumes: `resolve_hwaccel`, `HWACCEL_CHOICES`, `HWACCEL_AUTO`; Task 2의 `hwaccel_args` 인자
- Produces: `args.hwaccel` 문자열

- [ ] **Step 1: import 추가**

`seatnow.py`의 `from seatnow_layout import load_layout` 바로 위에:

```python
from seatnow_hwaccel import HWACCEL_AUTO, HWACCEL_CHOICES, resolve_hwaccel
```

- [ ] **Step 2: argparse 옵션 추가**

`parser.add_argument("--layout", ...)` 바로 앞에:

```python
    parser.add_argument(
        "--hwaccel",
        default=HWACCEL_AUTO,
        choices=HWACCEL_CHOICES,
        help="영상 디코딩에 쓸 하드웨어 가속기 (auto = OS에 맞는 것을 실제로 시험해보고 고름)",
    )
```

- [ ] **Step 3: `process_video`에서 한 번 해석하고 화면에 찍기**

`seatnow.py`의 `process_video` 안, `print(f"Video: {info.width}x...")` 블록 바로 다음에:

```python
    hwaccel = resolve_hwaccel(args.hwaccel, args.input)
    print(hwaccel.describe(), flush=True)
```

- [ ] **Step 4: 두 리더 생성부에 인자 전달**

`FFmpegSampleReader(...)` 호출(현재 `seatnow.py:397-402`)에 인자 추가:

```python
                reader = FFmpegSampleReader(
                    args.input,
                    args.sample_seconds,
                    info,
                    start_seconds=args.start_seconds,
                    hwaccel_args=hwaccel.args,
                )
```

`FFmpegBurstReader(...)` 호출(현재 `seatnow.py:469`):

```python
                reader = FFmpegBurstReader(args.input, info, hwaccel_args=hwaccel.args)
```

- [ ] **Step 5: `--help`가 뜨는지 확인**

Run: `./venv/Scripts/python.exe seatnow.py --help`
Expected: 출력에 `--hwaccel {auto,none,qsv,vaapi,d3d11va,dxva2,videotoolbox}` 가 보인다

- [ ] **Step 6: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: `Ran 249 tests ... OK`

- [ ] **Step 7: Commit**

```bash
git add seatnow.py
git commit -m "feat: seatnow.py --hwaccel (T9)"
```

---

### Task 4: `bench_decode.py` 계산 로직

**Files:**
- Create: `bench_decode.py`
- Test: `tests/test_bench_decode.py`

**Interfaces:**
- Consumes: `seatnow_hwaccel.hwaccel_input_args`
- Produces:
  - `class ClipSpec` — `.name`, `.width`, `.height`, `.codec`, `.bitrate_kbps`, `.megapixels`, `.filename`
  - `CLIP_SPECS: Tuple[ClipSpec, ...]`
  - `class DecodeMeasurement` — `.clip`, `.hwaccel`, `.cpu_seconds`, `.wall_seconds`, `.content_seconds`, `.cores_used`, `.realtime_factor`
  - `parse_benchmark(text: str) -> Optional[Tuple[float, float, float]]`
  - `cores_used(cpu_seconds: float, content_seconds: float) -> float`
  - `grade_decode(cores: float, total_cores: int) -> str`
  - `build_clip_command(ffmpeg, source, spec, destination, duration) -> List[str]`
  - `build_decode_command(ffmpeg, clip, hwaccel_args) -> List[str]`

- [ ] **Step 1: Write the failing test**

`tests/test_bench_decode.py`:

```python
"""Unit tests for the decode benchmark's model- and ffmpeg-independent logic."""

from __future__ import annotations

import unittest
from pathlib import Path

from bench_decode import (
    CLIP_SPECS,
    ClipSpec,
    build_clip_command,
    build_decode_command,
    cores_used,
    grade_decode,
    parse_benchmark,
)


BENCHMARK_OUTPUT = (
    "some unrelated ffmpeg chatter\n"
    "bench: utime=3.734s stime=0.453s rtime=1.126s\n"
    "bench: maxrss=115480KiB\n"
)


class ParseBenchmarkTests(unittest.TestCase):
    def test_reads_the_three_times(self):
        self.assertEqual(
            parse_benchmark(BENCHMARK_OUTPUT), (3.734, 0.453, 1.126)
        )

    def test_missing_line_is_none_not_zero(self):
        self.assertIsNone(parse_benchmark("ffmpeg said nothing useful\n"))

    def test_maxrss_line_alone_is_not_enough(self):
        self.assertIsNone(parse_benchmark("bench: maxrss=115480KiB\n"))


class CoresUsedTests(unittest.TestCase):
    def test_cpu_seconds_over_content_seconds(self):
        self.assertAlmostEqual(cores_used(4.187, 10.0), 0.4187)

    def test_zero_content_is_zero_not_a_crash(self):
        self.assertEqual(cores_used(4.0, 0.0), 0.0)


class GradeTests(unittest.TestCase):
    def test_quarter_of_the_machine_passes(self):
        self.assertEqual(grade_decode(1.0, 4), "PASS")

    def test_just_over_a_quarter_is_conditional(self):
        self.assertEqual(grade_decode(1.1, 4), "CONDITIONAL")

    def test_over_half_the_machine_fails(self):
        self.assertEqual(grade_decode(2.1, 4), "FAIL")

    def test_boundary_at_half_is_still_conditional(self):
        self.assertEqual(grade_decode(2.0, 4), "CONDITIONAL")

    def test_unknown_core_count_never_claims_a_pass(self):
        self.assertEqual(grade_decode(0.1, 0), "UNKNOWN")


class ClipSpecTests(unittest.TestCase):
    def test_megapixels_are_rounded_for_display(self):
        spec = ClipSpec("4mp_h264", 2560, 1440, "h264", 6000)
        self.assertAlmostEqual(spec.megapixels, 3.7, places=1)

    def test_filename_is_derived_from_the_name(self):
        spec = ClipSpec("4mp_h264", 2560, 1440, "h264", 6000)
        self.assertEqual(spec.filename, "4mp_h264.mp4")

    def test_every_resolution_is_measured_in_both_codecs(self):
        names = {spec.name for spec in CLIP_SPECS}
        for size in ("2mp", "4mp", "8mp"):
            self.assertIn(f"{size}_h264", names)
            self.assertIn(f"{size}_h265", names)

    def test_bitrates_are_cctv_sized_not_source_sized(self):
        # The source clips are 14.9 Mbps; a real camera streams far less.
        for spec in CLIP_SPECS:
            self.assertLessEqual(spec.bitrate_kbps, 10000)


class ClipCommandTests(unittest.TestCase):
    def test_scales_and_reencodes_at_the_requested_bitrate(self):
        spec = ClipSpec("4mp_h265", 2560, 1440, "h265", 6000)
        command = build_clip_command(
            "ffmpeg", Path("src.mov"), spec, Path("out.mp4"), duration=30.0
        )
        self.assertIn("libx265", command)
        self.assertIn("scale=2560:1440", " ".join(command))
        self.assertIn("6000k", command)
        self.assertEqual(command[-1], "out.mp4")

    def test_h264_uses_the_h264_encoder(self):
        spec = ClipSpec("2mp_h264", 1920, 1080, "h264", 4000)
        command = build_clip_command(
            "ffmpeg", Path("src.mov"), spec, Path("out.mp4"), duration=30.0
        )
        self.assertIn("libx264", command)

    def test_duration_limits_the_clip(self):
        spec = ClipSpec("2mp_h264", 1920, 1080, "h264", 4000)
        command = build_clip_command(
            "ffmpeg", Path("src.mov"), spec, Path("out.mp4"), duration=30.0
        )
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "30.000")


class DecodeCommandTests(unittest.TestCase):
    def test_asks_ffmpeg_for_its_own_cpu_accounting(self):
        command = build_decode_command("ffmpeg", Path("clip.mp4"), ())
        self.assertIn("-benchmark", command)

    def test_discards_output_so_only_decoding_is_measured(self):
        command = build_decode_command("ffmpeg", Path("clip.mp4"), ())
        self.assertIn("null", command)
        self.assertNotIn("libx264", command)

    def test_hwaccel_precedes_the_input(self):
        command = build_decode_command(
            "ffmpeg", Path("clip.mp4"), ("-hwaccel", "qsv")
        )
        self.assertLess(command.index("-hwaccel"), command.index("-i"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m unittest tests.test_bench_decode -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench_decode'`

- [ ] **Step 3: Write the implementation (계산 부분만)**

`bench_decode.py` (이 단계에서는 아래 내용까지만; `main`은 Task 5):

```python
"""Measure what 24/7 video decoding costs, so the camera can be chosen.

``bench.py`` measures inference, and inference resizes every frame to
``imgsz`` before the model sees it — which means a 2MP camera and an 8MP
camera cost the same there.  What actually scales with camera resolution is
decoding: unpacking the compressed stream, every frame, all day.  Nothing in
this repository measured that until now.

The source clips are 1080p at 14.9 Mbps, which is far heavier than a real
CCTV stream, so this script re-encodes them to camera-sized resolutions AND
camera-sized bitrates first.  The picture is upscaled and therefore fake; the
decoding cost is not, because it follows pixel count and bitrate.

    python bench_decode.py --source sample_raw/cafe_sample_angle1.mov
    python bench_decode.py --clips 4mp_h265 8mp_h265   # just the doubts
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from seatnow_hwaccel import (
    HWACCEL_AUTO,
    HWACCEL_CHOICES,
    hwaccel_input_args,
    resolve_hwaccel,
)


PROJECT_DIR = Path(__file__).resolve().parent
CLIP_DIR = PROJECT_DIR / "bench" / "clips"

_BENCHMARK_RE = re.compile(
    r"bench:\s*utime=([\d.]+)s\s+stime=([\d.]+)s\s+rtime=([\d.]+)s"
)

ENCODERS = {"h264": "libx264", "h265": "libx265"}

# A decoder that eats a quarter of the machine still leaves the tick budget
# (which bench.py grades at 50%) plus a quarter for the OS, log rotation and
# stream reconnects.
PASS_SHARE = 0.25
CONDITIONAL_SHARE = 0.50


@dataclass(frozen=True)
class ClipSpec:
    """One camera we might buy, expressed as something ffmpeg can produce."""

    name: str
    width: int
    height: int
    codec: str
    bitrate_kbps: int

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000.0

    @property
    def filename(self) -> str:
        return f"{self.name}.mp4"


CLIP_SPECS: Tuple[ClipSpec, ...] = (
    ClipSpec("2mp_h264", 1920, 1080, "h264", 4000),
    ClipSpec("2mp_h265", 1920, 1080, "h265", 4000),
    ClipSpec("4mp_h264", 2560, 1440, "h264", 6000),
    ClipSpec("4mp_h265", 2560, 1440, "h265", 6000),
    ClipSpec("8mp_h264", 3840, 2160, "h264", 10000),
    ClipSpec("8mp_h265", 3840, 2160, "h265", 10000),
)


@dataclass(frozen=True)
class DecodeMeasurement:
    clip: str
    hwaccel: str
    cpu_seconds: float
    wall_seconds: float
    content_seconds: float
    cores_used: float
    realtime_factor: float


def parse_benchmark(text: str) -> Optional[Tuple[float, float, float]]:
    """Read ``utime``/``stime``/``rtime`` out of ffmpeg's -benchmark line."""
    match = _BENCHMARK_RE.search(text)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def cores_used(cpu_seconds: float, content_seconds: float) -> float:
    """CPU seconds spent per second of video = cores held continuously."""
    if content_seconds <= 0:
        return 0.0
    return cpu_seconds / content_seconds


def grade_decode(cores: float, total_cores: int) -> str:
    """Grade continuous decoding against the whole machine."""
    if total_cores <= 0:
        return "UNKNOWN"
    share = cores / total_cores
    if share <= PASS_SHARE:
        return "PASS"
    if share <= CONDITIONAL_SHARE:
        return "CONDITIONAL"
    return "FAIL"


def build_clip_command(
    ffmpeg: str,
    source: Path,
    spec: ClipSpec,
    destination: Path,
    duration: float,
) -> List[str]:
    """Re-encode the source into one camera-shaped clip."""
    return [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={spec.width}:{spec.height}",
        "-c:v",
        ENCODERS[spec.codec],
        "-b:v",
        f"{spec.bitrate_kbps}k",
        "-maxrate",
        f"{spec.bitrate_kbps}k",
        "-bufsize",
        f"{spec.bitrate_kbps * 2}k",
        "-g",
        "60",
        "-preset",
        "medium",
        str(destination),
    ]


def build_decode_command(
    ffmpeg: str, clip: Path, hwaccel_args: Sequence[str]
) -> List[str]:
    """Decode the whole clip and throw the pixels away, reporting CPU time."""
    return [
        ffmpeg,
        "-nostdin",
        "-benchmark",
        "-v",
        "info",
        *hwaccel_args,
        "-i",
        str(clip),
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m unittest tests.test_bench_decode -v`
Expected: PASS — `Ran 18 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add bench_decode.py tests/test_bench_decode.py
git commit -m "feat: 디코딩 벤치의 클립 규격과 등급 판정"
```

---

### Task 5: `bench_decode.py` 실행부와 표

**Files:**
- Modify: `bench_decode.py`
- Test: `tests/test_bench_decode.py`

**Interfaces:**
- Consumes: Task 4의 모든 것
- Produces:
  - `combined_rows(measurements, bench_report, total_cores) -> List[Dict[str, object]]`
  - `format_decode_table(measurements, total_cores) -> str`
  - `format_combined_table(rows) -> str`
  - `main(argv=None) -> int`

`combined_rows`는 `bench.py`가 남긴 `bench/bench_report.json`의 `tick_budgets`와 디코딩 측정을 합친다. **이건 근사치다** — 추론이 도는 동안 코어를 전부 쓴다고 가정한다. 출력에 그 사실을 각주로 적는다 (`CLAUDE.md`: 트레이드오프를 숨기지 않는다).

- [ ] **Step 1: Write the failing test**

`tests/test_bench_decode.py`에 추가 (파일 상단 import에 `DecodeMeasurement`, `combined_rows`, `format_decode_table` 추가):

```python
def measurement(clip: str, hwaccel: str, cores: float) -> DecodeMeasurement:
    return DecodeMeasurement(
        clip=clip,
        hwaccel=hwaccel,
        cpu_seconds=cores * 30.0,
        wall_seconds=10.0,
        content_seconds=30.0,
        cores_used=cores,
        realtime_factor=3.0,
    )


class CombinedRowTests(unittest.TestCase):
    BENCH_REPORT = {
        "tick_budgets": [
            {"profile": "balanced", "backend": "ov-int8", "tick_utilization": 0.46},
            {"profile": "accuracy_default", "backend": "ov-int8", "tick_utilization": 0.90},
        ]
    }

    def test_adds_decode_share_to_inference_share(self):
        rows = combined_rows(
            [measurement("4mp_h265", "qsv", 0.72)], self.BENCH_REPORT, total_cores=4
        )
        balanced = [r for r in rows if r["profile"] == "balanced"][0]
        self.assertAlmostEqual(balanced["decode_share"], 0.18)
        self.assertAlmostEqual(balanced["total_share"], 0.64)
        self.assertEqual(balanced["grade"], "PASS")

    def test_over_the_machine_is_a_fail(self):
        rows = combined_rows(
            [measurement("8mp_h265", "none", 2.4)], self.BENCH_REPORT, total_cores=4
        )
        heavy = [r for r in rows if r["profile"] == "accuracy_default"][0]
        self.assertEqual(heavy["grade"], "FAIL")

    def test_no_bench_report_yields_no_rows(self):
        self.assertEqual(
            combined_rows([measurement("2mp_h264", "qsv", 0.1)], None, 4), []
        )

    def test_software_and_hardware_rows_are_kept_apart(self):
        rows = combined_rows(
            [
                measurement("4mp_h265", "qsv", 0.72),
                measurement("4mp_h265", "none", 2.0),
            ],
            self.BENCH_REPORT,
            total_cores=4,
        )
        pairs = {(r["clip"], r["hwaccel"], r["profile"]) for r in rows}
        self.assertIn(("4mp_h265", "qsv", "balanced"), pairs)
        self.assertIn(("4mp_h265", "none", "balanced"), pairs)


class DecodeTableTests(unittest.TestCase):
    def test_table_names_the_grade_and_the_cost(self):
        table = format_decode_table([measurement("4mp_h265", "qsv", 0.72)], 4)
        self.assertIn("4mp_h265", table)
        self.assertIn("qsv", table)
        self.assertIn("PASS", table)

    def test_unknown_core_count_is_shown_not_hidden(self):
        table = format_decode_table([measurement("4mp_h265", "qsv", 0.72)], 0)
        self.assertIn("UNKNOWN", table)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m unittest tests.test_bench_decode -v`
Expected: FAIL — `ImportError: cannot import name 'combined_rows'`

- [ ] **Step 3: Write the implementation**

`bench_decode.py` 끝에 추가:

```python
def combined_rows(
    measurements: Sequence[DecodeMeasurement],
    bench_report: Optional[Dict[str, object]],
    total_cores: int,
) -> List[Dict[str, object]]:
    """Put decoding and inference on one line per (clip, accelerator, profile).

    This is an approximation: it assumes inference saturates every core while
    it runs, so its tick utilisation can be read as a share of the machine.
    Close enough to reject a camera, not precise enough to defend a 3% margin.
    """
    if not bench_report:
        return []
    budgets = bench_report.get("tick_budgets") or []
    if not budgets or total_cores <= 0:
        return []
    rows: List[Dict[str, object]] = []
    for entry in measurements:
        decode_share = entry.cores_used / total_cores
        for budget in budgets:
            inference_share = float(budget.get("tick_utilization", 0.0))
            total_share = decode_share + inference_share
            if total_share <= PASS_SHARE + CONDITIONAL_SHARE:
                grade = "PASS"
            elif total_share <= 1.0:
                grade = "CONDITIONAL"
            else:
                grade = "FAIL"
            rows.append(
                {
                    "clip": entry.clip,
                    "hwaccel": entry.hwaccel,
                    "profile": budget.get("profile"),
                    "backend": budget.get("backend"),
                    "decode_share": round(decode_share, 3),
                    "inference_share": round(inference_share, 3),
                    "total_share": round(total_share, 3),
                    "grade": grade,
                }
            )
    return rows


def format_decode_table(
    measurements: Sequence[DecodeMeasurement], total_cores: int
) -> str:
    lines = [
        "",
        "## 디코딩 비용 (24시간 상시)",
        "",
        "| 클립 | 디코딩 방식 | 점유 코어 | 전체 대비 | 실시간 배속 | 판정 |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for entry in measurements:
        share = (
            f"{entry.cores_used / total_cores * 100:.0f}%" if total_cores > 0 else "?"
        )
        lines.append(
            f"| {entry.clip} | {entry.hwaccel} | {entry.cores_used:.2f} | "
            f"{share} | {entry.realtime_factor:.1f}x | "
            f"{grade_decode(entry.cores_used, total_cores)} |"
        )
    lines.append("")
    lines.append(
        f"'점유 코어' = 영상 1초를 푸는 데 드는 CPU 초. 이 박스의 코어 수는 "
        f"{total_cores if total_cores > 0 else '알 수 없음'}."
    )
    lines.append(
        "PASS = 전체의 25% 이내 / CONDITIONAL = 25~50% / FAIL = 50% 초과. "
        "나머지는 추론(bench.py 기준 50%)과 OS 몫이다."
    )
    return "\n".join(lines)


def format_combined_table(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return (
            "\n합산 판정: bench/bench_report.json 이 없어 건너뛴다. "
            "`python bench.py` 를 먼저 돌리면 추론까지 합친 표가 나온다."
        )
    lines = [
        "",
        "## 합산 판정 (디코딩 + 추론)",
        "",
        "| 클립 | 디코딩 방식 | 프로파일 | 디코딩 | 추론 | 합계 | 판정 |",
        "|---|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['clip']} | {row['hwaccel']} | {row['profile']} | "
            f"{float(row['decode_share']) * 100:.0f}% | "
            f"{float(row['inference_share']) * 100:.0f}% | "
            f"{float(row['total_share']) * 100:.0f}% | {row['grade']} |"
        )
    lines.append("")
    lines.append(
        "⚠️ 근사치다. 추론이 도는 동안 코어를 전부 쓴다고 가정했다. "
        "카메라를 탈락시키는 근거로는 충분하지만 3% 차이를 다투는 데는 못 쓴다."
    )
    return "\n".join(lines)


def ensure_clips(
    ffmpeg: str, source: Path, specs: Sequence[ClipSpec], duration: float,
    rebuild: bool,
) -> Dict[str, Path]:
    """Create the camera-shaped clips once and reuse them afterwards."""
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clips: Dict[str, Path] = {}
    for spec in specs:
        destination = CLIP_DIR / spec.filename
        if destination.exists() and not rebuild:
            print(f"  재사용: {destination.name}")
            clips[spec.name] = destination
            continue
        print(
            f"  생성 중: {destination.name} "
            f"({spec.width}x{spec.height}, {spec.codec}, {spec.bitrate_kbps}kbps)",
            flush=True,
        )
        command = build_clip_command(ffmpeg, source, spec, destination, duration)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not destination.exists():
            print(f"    실패: {completed.stderr.strip()[:400]}")
            continue
        clips[spec.name] = destination
    return clips


def measure_clip(
    ffmpeg: str, clip: Path, hwaccel_name: str, hwaccel_args: Sequence[str]
) -> Optional[DecodeMeasurement]:
    """Decode one clip once and turn ffmpeg's own accounting into cores."""
    from seatnow_core import probe_video

    info = probe_video(clip)
    command = build_decode_command(ffmpeg, clip, hwaccel_args)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        print(f"    실패: {completed.stderr.strip()[:400]}")
        return None
    parsed = parse_benchmark(completed.stderr)
    if parsed is None:
        print("    실패: ffmpeg가 -benchmark 결과를 내지 않았다")
        return None
    utime, stime, rtime = parsed
    cpu_seconds = utime + stime
    content_seconds = info.duration
    return DecodeMeasurement(
        clip=clip.stem,
        hwaccel=hwaccel_name,
        cpu_seconds=round(cpu_seconds, 3),
        wall_seconds=round(wall_seconds, 3),
        content_seconds=round(content_seconds, 3),
        cores_used=round(cores_used(cpu_seconds, content_seconds), 4),
        realtime_factor=round(
            content_seconds / rtime if rtime > 0 else 0.0, 2
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="영상 디코딩 비용을 재서 살 카메라의 해상도·코덱을 정한다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_DIR / "sample_raw" / "cafe_sample_angle1.mov",
        help="벤치 클립을 만들 원본 영상",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=[spec.name for spec in CLIP_SPECS],
        choices=[spec.name for spec in CLIP_SPECS],
        help="측정할 클립",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="클립 길이(초)"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="캐시된 클립을 버리고 다시 만든다"
    )
    parser.add_argument(
        "--hwaccel",
        default=HWACCEL_AUTO,
        choices=HWACCEL_CHOICES,
        help="하드웨어 디코딩 방식 (auto = 실제로 시험해보고 고름)",
    )
    parser.add_argument(
        "--software-only",
        action="store_true",
        help="하드웨어 디코딩 측정을 건너뛴다",
    )
    parser.add_argument(
        "--bench-report",
        type=Path,
        default=PROJECT_DIR / "bench" / "bench_report.json",
        help="bench.py가 남긴 추론 결과 (있으면 합산 판정에 쓴다)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "bench" / "decode_report.json",
        help="측정 결과를 남길 경로",
    )
    parser.add_argument(
        "--label", default=platform.node(), help="보고서에 남길 장비 이름"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg를 찾지 못했다. docs/edge-setup.md 의 설치 절차를 볼 것.")
        return 1
    if not args.source.exists():
        print(f"원본 영상이 없다: {args.source}")
        return 1

    total_cores = os.cpu_count() or 0
    print(f"장비: {args.label}  ({platform.platform()})")
    print(f"코어 수: {total_cores if total_cores else '알 수 없음'}")
    print(f"원본: {args.source}\n")

    specs = [spec for spec in CLIP_SPECS if spec.name in set(args.clips)]
    print("### 벤치 클립 준비")
    clips = ensure_clips(ffmpeg, args.source, specs, args.duration, args.rebuild)
    if not clips:
        print("만들어진 클립이 없다.")
        return 1

    modes: List[Tuple[str, Tuple[str, ...]]] = [("none", ())]
    if not args.software_only:
        first_clip = next(iter(clips.values()))
        choice = resolve_hwaccel(args.hwaccel, first_clip)
        print(f"\n{choice.describe()}")
        if choice.enabled:
            modes.append((choice.name, choice.args))
        else:
            print("→ 하드웨어 측정은 건너뛴다. 소프트웨어 숫자만 나온다.")

    print("\n### 측정")
    measurements: List[DecodeMeasurement] = []
    for name, clip in clips.items():
        for mode_name, mode_args in modes:
            print(f"  {name:10s} {mode_name:12s}", end=" ", flush=True)
            entry = measure_clip(ffmpeg, clip, mode_name, mode_args)
            if entry is None:
                continue
            measurements.append(entry)
            print(f"코어 {entry.cores_used:.2f}")

    if not measurements:
        print("측정된 것이 없다.")
        return 1

    print(format_decode_table(measurements, total_cores))

    bench_report: Optional[Dict[str, object]] = None
    if args.bench_report.exists():
        bench_report = json.loads(args.bench_report.read_text(encoding="utf-8"))
    rows = combined_rows(measurements, bench_report, total_cores)
    print(format_combined_table(rows))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "label": args.label,
                "platform": platform.platform(),
                "total_cores": total_cores,
                "source": str(args.source),
                "duration": args.duration,
                "measurements": [asdict(m) for m in measurements],
                "combined": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n보고서: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m unittest tests.test_bench_decode -v`
Expected: PASS — `Ran 24 tests ... OK`

- [ ] **Step 5: 실제로 한 번 돌려본다 (가장 가벼운 조합)**

Run: `./venv/Scripts/python.exe bench_decode.py --clips 2mp_h264 --duration 10`
Expected: 클립이 만들어지고, 디코딩 표에 `2mp_h264` 줄이 나오고, 하드웨어 디코딩 켜짐/꺼짐이 찍힌다

- [ ] **Step 6: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: `Ran 255 tests ... OK`

- [ ] **Step 7: Commit**

```bash
git add bench_decode.py tests/test_bench_decode.py
git commit -m "feat: bench_decode.py - 디코딩 비용 측정과 추론 합산 판정"
```

---

### Task 6: `check_edge.py` — 새 박스 검수

**Files:**
- Create: `check_edge.py`
- Test: `tests/test_check_edge.py`

**Interfaces:**
- Consumes: `seatnow_hwaccel.resolve_hwaccel`
- Produces:
  - `class Check` — `.name`, `.ok`, `.detail`, `.fix`
  - `check_python(version_info) -> Check`
  - `check_cores(count) -> Check`
  - `check_memory_gb(gigabytes) -> Check`
  - `check_disk_gb(gigabytes) -> Check`
  - `check_packages(installed: Dict[str, Optional[str]]) -> List[Check]`
  - `format_report(checks) -> str`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_check_edge.py`:

```python
"""Unit tests for the edge-box readiness checks."""

from __future__ import annotations

import unittest

from check_edge import (
    Check,
    check_cores,
    check_disk_gb,
    check_memory_gb,
    check_packages,
    check_python,
    format_report,
)


class PythonCheckTests(unittest.TestCase):
    def test_supported_version_passes(self):
        self.assertTrue(check_python((3, 11, 9)).ok)

    def test_too_old_fails_and_says_what_to_do(self):
        result = check_python((3, 8, 10))
        self.assertFalse(result.ok)
        self.assertIn("3.9", result.fix)


class CoreCheckTests(unittest.TestCase):
    def test_four_cores_pass(self):
        self.assertTrue(check_cores(4).ok)

    def test_two_cores_fail(self):
        self.assertFalse(check_cores(2).ok)

    def test_unknown_core_count_is_a_failure_not_a_pass(self):
        self.assertFalse(check_cores(0).ok)


class MemoryCheckTests(unittest.TestCase):
    def test_eight_gigabytes_pass(self):
        self.assertTrue(check_memory_gb(8.0).ok)

    def test_four_gigabytes_pass(self):
        self.assertTrue(check_memory_gb(4.0).ok)

    def test_two_gigabytes_fail(self):
        self.assertFalse(check_memory_gb(2.0).ok)


class DiskCheckTests(unittest.TestCase):
    def test_plenty_of_room_passes(self):
        self.assertTrue(check_disk_gb(50.0).ok)

    def test_no_room_for_the_models_fails(self):
        self.assertFalse(check_disk_gb(2.0).ok)


class PackageCheckTests(unittest.TestCase):
    def test_all_present_passes(self):
        checks = check_packages(
            {"numpy": "1.26.4", "cv2": "4.10.0", "torch": "2.2.2",
             "ultralytics": "8.4.82"}
        )
        self.assertTrue(all(check.ok for check in checks))

    def test_missing_package_names_the_install_command(self):
        checks = check_packages(
            {"numpy": "1.26.4", "cv2": "4.10.0", "torch": None,
             "ultralytics": None}
        )
        failed = [check for check in checks if not check.ok]
        self.assertEqual(len(failed), 2)
        self.assertIn("pip install", failed[0].fix)

    def test_numpy_2_is_rejected(self):
        checks = check_packages(
            {"numpy": "2.1.0", "cv2": "4.10.0", "torch": "2.2.2",
             "ultralytics": "8.4.82"}
        )
        numpy_check = [c for c in checks if c.name.startswith("numpy")][0]
        self.assertFalse(numpy_check.ok)
        self.assertIn("numpy<2", numpy_check.fix)


class ReportTests(unittest.TestCase):
    def test_report_marks_each_line_and_summarises(self):
        report = format_report(
            [
                Check("ffmpeg", True, "8.1.2", ""),
                Check("코어 수", False, "2개", "코어 4개 이상인 박스로 바꾼다"),
            ]
        )
        self.assertIn("ffmpeg", report)
        self.assertIn("코어 4개 이상인 박스로 바꾼다", report)
        self.assertIn("1개 불합격", report)

    def test_all_pass_says_so(self):
        report = format_report([Check("ffmpeg", True, "8.1.2", "")])
        self.assertIn("전부 합격", report)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m unittest tests.test_check_edge -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'check_edge'`

- [ ] **Step 3: Write the implementation**

`check_edge.py`:

```python
"""Check whether a box can run SeatNow, in words a non-developer can act on.

Run this first on a newly bought mini-PC.  Every failing line says what to do
about it, because the person holding the box is not necessarily the person
who wrote the pipeline.

    python check_edge.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parent

MIN_PYTHON = (3, 9)
MIN_CORES = 4
MIN_MEMORY_GB = 4.0
MIN_DISK_GB = 10.0
REQUIRED_PACKAGES = ("numpy", "cv2", "torch", "ultralytics")
PIP_NAMES = {"cv2": "opencv-python"}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str


def check_python(version_info: Sequence[int]) -> Check:
    version = ".".join(str(part) for part in version_info[:3])
    ok = tuple(version_info[:2]) >= MIN_PYTHON
    return Check(
        name="Python 버전",
        ok=ok,
        detail=version,
        fix="" if ok else "Python 3.9 이상을 설치한다 (권장 3.11).",
    )


def check_cores(count: int) -> Check:
    ok = count >= MIN_CORES
    return Check(
        name="CPU 코어 수",
        ok=ok,
        detail=f"{count}개" if count else "알 수 없음",
        fix=""
        if ok
        else f"코어 {MIN_CORES}개 이상인 박스로 바꾼다. "
        "코어가 모자라면 영상 푸는 것과 판정이 서로 자리를 뺏는다.",
    )


def check_memory_gb(gigabytes: float) -> Check:
    ok = gigabytes >= MIN_MEMORY_GB
    return Check(
        name="메모리",
        ok=ok,
        detail=f"{gigabytes:.1f}GB" if gigabytes else "알 수 없음",
        fix="" if ok else f"RAM을 {MIN_MEMORY_GB:.0f}GB 이상으로 늘린다.",
    )


def check_disk_gb(gigabytes: float) -> Check:
    ok = gigabytes >= MIN_DISK_GB
    return Check(
        name="디스크 여유",
        ok=ok,
        detail=f"{gigabytes:.1f}GB",
        fix=""
        if ok
        else f"{MIN_DISK_GB:.0f}GB 이상 비운다. 모델 파일만 2GB 가까이 된다.",
    )


def check_packages(installed: Dict[str, Optional[str]]) -> List[Check]:
    checks: List[Check] = []
    for name in REQUIRED_PACKAGES:
        version = installed.get(name)
        pip_name = PIP_NAMES.get(name, name)
        if version is None:
            checks.append(
                Check(
                    name=f"{name} 설치",
                    ok=False,
                    detail="없음",
                    fix=f"pip install {pip_name}",
                )
            )
            continue
        if name == "numpy" and version.split(".")[0] == "2":
            checks.append(
                Check(
                    name="numpy 버전",
                    ok=False,
                    detail=version,
                    fix='pip install "numpy<2" — numpy 2.x는 PyTorch와 충돌한다.',
                )
            )
            continue
        checks.append(Check(name=f"{name} 설치", ok=True, detail=version, fix=""))
    return checks


def format_report(checks: Sequence[Check]) -> str:
    lines = ["", "## 엣지 박스 검수 결과", ""]
    for check in checks:
        mark = "합격" if check.ok else "불합격"
        lines.append(f"[{mark}] {check.name}: {check.detail}")
        if not check.ok and check.fix:
            lines.append(f"         → {check.fix}")
    failed = [check for check in checks if not check.ok]
    lines.append("")
    if failed:
        lines.append(f"{len(failed)}개 불합격. 위의 → 줄대로 처리하고 다시 돌린다.")
    else:
        lines.append("전부 합격. `python bench.py` 와 `python bench_decode.py` 를 돌린다.")
    return "\n".join(lines)


def _installed_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in REQUIRED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "설치됨")
        except Exception:
            versions[name] = None
    return versions


def _memory_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
        return 0.0
    if sys.platform == "win32":
        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1024.0 ** 3)
    return 0.0


def _ffmpeg_check() -> Check:
    binary = shutil.which("ffmpeg")
    if not binary:
        return Check(
            name="ffmpeg 설치",
            ok=False,
            detail="없음",
            fix="docs/edge-setup.md 의 ffmpeg 설치 절차를 따른다.",
        )
    try:
        completed = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=15.0
        )
        version = completed.stdout.splitlines()[0] if completed.stdout else "설치됨"
    except Exception:
        version = "설치됨"
    return Check(name="ffmpeg 설치", ok=True, detail=version, fix="")


def _hwaccel_check(sample: Optional[Path]) -> Check:
    from seatnow_hwaccel import resolve_hwaccel

    if sample is None or not sample.exists():
        return Check(
            name="하드웨어 디코딩",
            ok=False,
            detail="시험할 영상이 없어 확인 못 함",
            fix="--sample 로 영상 파일을 하나 지정해서 다시 돌린다.",
        )
    choice = resolve_hwaccel("auto", sample)
    return Check(
        name="하드웨어 디코딩",
        ok=choice.enabled,
        detail=choice.describe(),
        fix=""
        if choice.enabled
        else "그래픽 드라이버를 설치한다. Linux는 intel-media-va-driver, "
        "Windows는 인텔 그래픽 드라이버. 없으면 영상 푸는 데만 CPU를 크게 쓴다.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="이 박스가 SeatNow를 돌릴 수 있는지 항목별로 확인한다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=PROJECT_DIR / "sample_raw" / "cafe_sample_angle1.mov",
        help="하드웨어 디코딩을 시험해볼 영상",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"장비: {platform.node()}  ({platform.platform()})")
    print(f"CPU: {platform.processor() or '알 수 없음'}")

    checks: List[Check] = [
        check_python(sys.version_info),
        check_cores(os.cpu_count() or 0),
        check_memory_gb(_memory_gb()),
        check_disk_gb(shutil.disk_usage(PROJECT_DIR).free / (1024.0 ** 3)),
        _ffmpeg_check(),
    ]
    checks.extend(check_packages(_installed_versions()))
    checks.append(_hwaccel_check(args.sample))

    print(format_report(checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m unittest tests.test_check_edge -v`
Expected: PASS — `Ran 15 tests ... OK`

- [ ] **Step 5: 이 노트북에서 실제로 돌려본다**

Run: `./venv/Scripts/python.exe check_edge.py`
Expected: 항목별 합격/불합격이 한국어로 나오고, 마지막 줄에 요약이 찍힌다

- [ ] **Step 6: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: `Ran 270 tests ... OK`

- [ ] **Step 7: Commit**

```bash
git add check_edge.py tests/test_check_edge.py
git commit -m "feat: check_edge.py - 새 박스 검수, 불합격마다 할 일을 같이 낸다"
```

---

### Task 7: `docs/edge-setup.md` — 설치 절차

**Files:**
- Create: `docs/edge-setup.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 4~6의 명령들
- Produces: 없음 (문서)

- [ ] **Step 1: 문서 작성**

`docs/edge-setup.md`를 만든다. 반드시 담을 것:

1. **이 문서로 뭘 하는지** 한 문단 — "새로 산 미니PC에서 '이 박스로 어떤 카메라를 살 수 있나'를 재는 데까지 간다"
2. **OS 선택** 절 — 측정 단계는 깔려 온 OS 그대로, 24/7 운영은 Linux 권장. 근거는 성능이 아니라 Windows 강제 재부팅 위험 (스펙 §3-C 내용을 옮긴다)
3. **Windows 길** — Python 3.11 설치, ffmpeg 설치(gyan.dev 빌드 압축 해제 + PATH 등록), 인텔 그래픽 드라이버, 저장소 복사, `venv` 생성, `pip install -r requirements.txt`
4. **Linux 길** — Ubuntu Server 설치 USB 만들기(Rufus/balenaEtcher), `apt install python3-venv ffmpeg intel-media-va-driver-non-free`, 나머지는 동일
5. **검수 두 줄**:
   ```
   python check_edge.py
   python bench.py --frames sample_raw/cafe_sample_angle1.mov --label edge-box
   python bench_decode.py --source sample_raw/cafe_sample_angle1.mov
   ```
6. **결과 읽는 법** — 디코딩 표와 합산 표에서 어떤 카메라가 살아남는지 판단하는 법. 합산 표가 근사치라는 경고 포함
7. **자주 막히는 곳** — ffmpeg가 PATH에 없을 때, 하드웨어 디코딩이 "꺼짐"으로 나올 때(드라이버), 클립 생성이 느릴 때(`--duration 10`)

`CLAUDE.md`의 "설명은 비개발자 기준" 규칙을 따른다. 명령어는 전부 복붙 가능한 형태로.

- [ ] **Step 2: README에 연결**

`README.md`에 엣지 박스 절을 추가하고 `docs/edge-setup.md`로 연결한다.

- [ ] **Step 3: 문서의 명령이 실제로 도는지 확인**

Run: `./venv/Scripts/python.exe check_edge.py && ./venv/Scripts/python.exe bench_decode.py --clips 2mp_h264 --duration 10`
Expected: 둘 다 에러 없이 표를 출력

- [ ] **Step 4: Commit**

```bash
git add docs/edge-setup.md README.md
git commit -m "docs: 엣지 박스 설치·검수 절차 (Windows/Linux)"
```

---

### Task 8: T14 — 노트북에서 실영상 첫 실행

**Files:**
- Modify: `requirements.txt` (설치된 실제 버전과 어긋나면)
- Create: `layouts/cafe_angle1.json` (사람이 그린 결과)

**Interfaces:**
- Consumes: 없음
- Produces: `layouts/cafe_angle1.json`

⚠️ **이 작업의 마지막 단계는 사람이 마우스로 그리는 작업이다.** 자동화 대상이 아니다.

- [ ] **Step 1: 모델 의존성이 설치됐는지 확인**

Run: `./venv/Scripts/python.exe -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"`
Expected: 두 버전이 찍힌다. 실패하면 `./venv/Scripts/python.exe -m pip install "torch==2.2.2" "ultralytics==8.4.82"` (핀이 꼬이면 핀을 빼고 재설치, `numpy<2`는 유지)

- [ ] **Step 2: 회귀 확인 — 테스트가 여전히 통과하는가**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: `OK`. torch/ultralytics 설치가 기존 테스트를 깨면 **여기서 멈추고 원인을 찾는다**

- [ ] **Step 3: 설치된 버전이 requirements.txt와 다르면 갱신**

`requirements.txt`의 주석 "(Python 3.9, macOS Intel 기준 — 2026-07-12)"에 Windows/Python 3.11에서 검증된 조합을 한 줄 덧붙인다.

- [ ] **Step 4: angle1에 캘리브레이션 실행**

Run:
```
./venv/Scripts/python.exe calibrate.py sample_raw/cafe_sample_angle1.mov --output layouts/cafe_angle1.json
```
Expected: 첫 실행 시 `yolov8x.pt`(약 130MB)를 내려받고, 창이 뜨며 `_preseed`가 자동으로 잡은 테이블·의자가 그려진다

- [ ] **Step 5: 화면에서 4가지를 확인한다 (사람 작업)**

1. 일반 테이블을 몇 개나 잡았는가 — 눈으로 센 좌석 수와 비교
2. 일자형/벽 책상이 안 잡혔는가 — 안 잡혔으면 `[z]`(바 구역) + `[x]`(자리 칸)로 직접 그린다
3. 엉뚱한 것(선반·카운터)을 테이블로 잡았는가 — 지운다
4. 의자가 테이블에 연결되는가

저장 시 출력되는 `테이블 N개, 의자 M개, 바 구역 K개(S석)`가 눈으로 센 것과 맞아야 완료다.

- [ ] **Step 6: 그 레이아웃으로 실제 분석을 돌려 사유 코드 분포를 본다**

Run:
```
./venv/Scripts/python.exe seatnow.py sample_raw/cafe_sample_angle1.mov \
  --layout layouts/cafe_angle1.json --sample-seconds 5 --log-detections --no-video
./venv/Scripts/python.exe verify_seatnow.py sample_results/cafe_sample_angle1_seatnow.jsonl
```
Expected: UNKNOWN 사유 코드가 그룹별(`install`/`geometry`/`model`/`time`/`settled`)로 분해되어 나온다. **`model` 그룹 비중이 파인튜닝 판단의 근거다.**

- [ ] **Step 7: Commit**

```bash
git add layouts/cafe_angle1.json requirements.txt
git commit -m "feat: angle1 좌석 레이아웃 (T14 실영상 첫 실행)"
```

---

### Task 9: `plan.md` 갱신

**Files:**
- Modify: `plan.md`

- [ ] **Step 1: 진행 상황 표 갱신**

`plan.md` §0-a의 표에 다음을 반영한다.

- `T9 Quick Sync` → ✅ 완료 (Task 1~3). "엣지 도착 전에 끝냄 — 디코딩 측정이 의미를 가지려면 선행되어야 했다"
- `T14 실영상 첫 실행` → 결과에 따라 ✅ 또는 진행 중
- 신규 행 `T17 디코딩 벤치` → ✅ 완료. `bench_decode.py`, `check_edge.py`, `docs/edge-setup.md`

- [ ] **Step 2: 틀린 전제를 정정으로 남긴다**

`plan.md` §2의 "카메라 즉시 구매 — ❌ 엣지 벤치 후" 항목 아래에 정정을 덧붙인다.

```markdown
> **2026-08-31 정정**: "추론 벤치(`bench.py`)로 해상도를 고른다"는 틀렸다. 추론은
> 프레임을 `imgsz`로 줄여 넣으므로 **카메라 해상도와 추론 시간이 거의 무관하다.**
> 해상도가 실제로 잡아먹는 것은 24시간 도는 **디코딩**이고, 그것을 재는 코드도
> 하드웨어 디코딩 옵션도 없었다. → `bench_decode.py`(T17)와 T9로 해결.
> 카메라 선정 근거는 이제 `bench_decode.py`의 합산 표다.
```

- [ ] **Step 3: §4 다음 액션 갱신**

T8·T10을 "카메라 확정 후"로 명시하고, 다음 액션을 `엣지 박스 도착 → check_edge.py → bench.py → bench_decode.py → 카메라 확정 → T8/T10`으로 다시 그린다.

- [ ] **Step 4: Commit**

```bash
git add plan.md
git commit -m "docs: plan.md 갱신 - T9/T17 완료, 카메라 선정 근거 정정"
```

---

## Self-Review

**스펙 커버리지**

| 스펙 절 | 담당 태스크 |
|---|---|
| §3-A-1 가짜 카메라 영상 생성 | Task 4 (`build_clip_command`, `CLIP_SPECS`), Task 5 (`ensure_clips`) |
| §3-A-2 측정 | Task 4 (`parse_benchmark`, `build_decode_command`), Task 5 (`measure_clip`) |
| §3-A-3 판정 | Task 4 (`grade_decode`) |
| §3-A-4 합산 판정 | Task 5 (`combined_rows`, `format_combined_table`) |
| §3-B 하드웨어 디코딩 | Task 1 (선택), Task 2 (배선), Task 3 (CLI) |
| §3-B 조용히 넘어가지 않는다 | Task 1 (`probe_hwaccel`이 프레임 유무로 판정, `describe()`), Task 3 (화면 출력) |
| §3-B 인코딩은 손대지 않는다 | Task 2가 `FFmpegVideoWriter`를 건드리지 않음 |
| §3-C `check_edge.py` | Task 6 |
| §3-C `docs/edge-setup.md`, OS 결정 기록 | Task 7 |
| §3-D T14 | Task 8 |
| §4 하지 않는 것 | 어느 태스크에도 T8·T10·하드웨어 인코딩·psutil 없음 |
| §5 제약 | Global Constraints |
| §6 파인튜닝 근거 | Task 8 Step 6 (사유 코드 분포) |
| §7 완료 기준 1~6 | Task 6 Step 5, Task 5 Step 5, Task 1·3, Task 7 Step 3, 각 태스크 전체 테스트, Task 8 |

빠진 것 없음.

**타입 일관성**

- `hwaccel_args`는 Task 1이 `Tuple[str, ...]`로 만들고, Task 2가 `Sequence[str]`로 받아 `tuple()`로 저장하고, Task 3이 `hwaccel.args`로 넘긴다 — 일치
- `HwaccelChoice.args`는 `hwaccel_input_args`의 반환값 그대로 — 일치
- `DecodeMeasurement.cores_used`(필드)와 `cores_used()`(함수)가 이름이 겹치지만 다른 이름공간이고, Task 4·5 코드에서 실제로 그렇게 쓰인다 — 의도적
- `grade_decode(cores, total_cores)`의 인자 순서가 Task 4 테스트와 Task 5 `format_decode_table` 양쪽에서 동일 — 일치
- `combined_rows`가 읽는 `tick_utilization`·`profile`·`backend`는 `bench.py:196-210`의 `tick_budget()`이 실제로 내는 키 — 확인함

**플레이스홀더**: 없음. 모든 코드 단계에 실제 코드가 들어 있다.
