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


PROJECT_DIR = Path(__file__).resolve().parents[1]

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
        "코어가 모자라면 영상 푸는 일과 판정하는 일이 서로 자리를 뺏는다.",
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
        lines.append(
            "전부 합격. `python bench.py` 와 `python bench_decode.py` 를 돌린다."
        )
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
            return status.ullTotalPhys / (1024.0**3)
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
    from engine.seatnow_hwaccel import resolve_hwaccel

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
        "Windows는 인텔 그래픽 드라이버. 없으면 영상 푸는 데만 CPU를 3~6배 더 쓴다.",
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
        check_disk_gb(shutil.disk_usage(PROJECT_DIR).free / (1024.0**3)),
        _ffmpeg_check(),
    ]
    checks.extend(check_packages(_installed_versions()))
    checks.append(_hwaccel_check(args.sample))

    print(format_report(checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
