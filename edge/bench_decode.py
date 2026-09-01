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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from engine.seatnow_hwaccel import HWACCEL_AUTO, HWACCEL_CHOICES, resolve_hwaccel


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLIP_DIR = PROJECT_DIR / "results" / "edge" / "clips"

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
        "'점유 코어' = 영상 1초를 푸는 데 드는 CPU 초. 이 박스의 코어 수는 "
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
            "\n합산 판정: results/edge/bench_report.json 이 없어 건너뛴다. "
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
    ffmpeg: str,
    source: Path,
    specs: Sequence[ClipSpec],
    duration: float,
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
    from engine.seatnow_core import probe_video

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
        realtime_factor=round(content_seconds / rtime if rtime > 0 else 0.0, 2),
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
    parser.add_argument("--duration", type=float, default=30.0, help="클립 길이(초)")
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
        default=PROJECT_DIR / "results" / "edge" / "bench_report.json",
        help="bench.py가 남긴 추론 결과 (있으면 합산 판정에 쓴다)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "results" / "edge" / "decode_report.json",
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
