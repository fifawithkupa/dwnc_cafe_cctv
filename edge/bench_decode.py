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


from edge import tolerant_stdout

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


# 클립을 "만드는" 비용은 측정 대상이 아니다 -- 재는 것은 "푸는" 비용이다.
# 그런데 만드는 쪽(소프트웨어 인코딩)이 메모리를 훨씬 많이 먹고, 4GB 박스는
# 8MP 클립을 만들다 메모리가 모자랐다 (docs/edge-setup.md 9단계).  이 표는
# 그 박스가 만들 수 있었던 것과 없었던 것을 적은 문턱이지 정밀한 실측이
# 아니다 -- 노트북(8스레드) 최고점은 8mp_h265 1.8GB, 8mp_h264 2.3GB,
# 4mp_h265 1.0GB 였다 (2026-09-05).  박스는 어차피 클립을 만들면 안 되는
# 장비라(수십 분), 막는 쪽으로 틀리는 편이 낫다.  0.0 은 "특별히 요구하는
# 것이 없다"는 뜻이다.
ENCODE_MEMORY_GB: Dict[str, float] = {
    "4mp_h265": 1.5,
    "8mp_h264": 1.5,
    "8mp_h265": 4.0,
}


@dataclass(frozen=True)
class SkippedClip:
    """A clip this machine did not build, and the reason to show the user."""

    name: str
    reason: str


def encode_memory_gb(spec: ClipSpec) -> float:
    """Peak memory software encoding this clip wants; 0.0 = unremarkable."""
    return ENCODE_MEMORY_GB.get(spec.name, 0.0)


def can_software_encode(spec: ClipSpec, available_gb: float) -> bool:
    """Whether this machine should build the clip itself.

    An unreadable memory figure answers yes.  A wrong "no" would leave the
    user with no clips at all, while an encode that fails is recoverable and
    says so on screen.
    """
    if available_gb <= 0:
        return True
    return encode_memory_gb(spec) <= available_gb


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
            "`python -m edge.bench` 를 먼저 돌리면 추론까지 합친 표가 나온다."
        )
    lines = [
        "",
        "## 합산 판정 (디코딩 + 추론)",
        "",
        # 배포는 OpenVINO로 하므로 어느 백엔드의 줄인지가 판정을 바꾼다.
        # 이 칸이 없으면 pt 줄과 ov 줄이 똑같이 생겨서 표를 읽을 수 없다.
        "| 클립 | 디코딩 방식 | 프로파일 | 백엔드 | 디코딩 | 추론 | 합계 | 판정 |",
        "|---|---|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['clip']} | {row['hwaccel']} | {row['profile']} | "
            f"{row.get('backend') or '?'} | "
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


# 카메라를 고를 때 보는 줄은 하나다: 실제로 배포하는 설정.  프로파일은
# accuracy_default (72칸 정답지 70/72 를 낸 설정), 백엔드는 OpenVINO.
DEPLOY_PROFILE = "accuracy_default"
DEPLOY_BACKENDS = ("ov-fp32", "pt")  # 앞의 것이 있으면 그것만, 없으면 다음 것


def decision_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """The combined rows a buyer actually reads: deployed profile, deployed backend."""
    present = {row.get("backend") for row in rows}
    backend = next((name for name in DEPLOY_BACKENDS if name in present), None)
    if backend is None:
        return []
    return [
        row
        for row in rows
        if row.get("profile") == DEPLOY_PROFILE and row.get("backend") == backend
    ]


def format_decision_table(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return (
            "\n결정 표: 합산 판정이 없어 못 만든다. `python -m edge.bench` 를 먼저 돌린다."
        )
    backend = str(rows[0].get("backend"))
    lines = [
        "",
        f"## 결정 표 — 배포 설정만 ({DEPLOY_PROFILE} · {backend})",
        "",
        "| 클립 | 디코딩 방식 | 디코딩 | 추론 | 합계 | 판정 |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['clip']} | {row['hwaccel']} | "
            f"{float(row['decode_share']) * 100:.0f}% | "
            f"{float(row['inference_share']) * 100:.0f}% | "
            f"{float(row['total_share']) * 100:.0f}% | {row['grade']} |"
        )
    lines.append("")
    if backend != DEPLOY_BACKENDS[0]:
        lines.append(
            f"⚠️ {DEPLOY_BACKENDS[0]} 줄이 없어 {backend} 로 대신 보였다. 배포는 OpenVINO 라 "
            "이 표로는 카메라를 정하지 못한다 — 5단계 변환 후 `edge.bench` 부터 다시."
        )
    else:
        lines.append(
            "이 표에서 PASS 인 것 중 해상도가 가장 높은 것을 고른다. "
            "디코딩 방식은 박스가 실제로 쓸 하드웨어 줄(none 이 아닌 쪽)을 본다."
        )
    return "\n".join(lines)


def format_missing_clips(skipped: Sequence[SkippedClip]) -> str:
    """Tell the user how to get the clips this box refused to build.

    The clips are input to the measurement, not the measurement, so building
    them somewhere roomier and copying them over changes nothing about the
    numbers.  The receiving folder is created first because scp does not
    create it (docs/edge-setup.md 3단계).
    """
    if not skipped:
        return ""
    names = " ".join(entry.name for entry in skipped)
    lines = [
        "",
        "## 이 박스에서 못 만든 클립",
        "",
    ]
    for entry in skipped:
        lines.append(f"- **{entry.name}** — {entry.reason}")
    lines += [
        "",
        "클립은 **재료**일 뿐이라 어디서 만들든 측정값은 같다.",
        "노트북에서 만들어 박스로 복사하면 된다.",
        "",
        "```bash",
        "# ① 노트북에서 — 만들기만 하고 측정은 안 한다",
        f"venv/Scripts/python.exe -m edge.bench_decode --build-only --clips {names}",
        "",
        "# ② 박스에서 — 받을 폴더를 먼저 만든다 (복사는 폴더를 안 만들어준다)",
        "mkdir -p ~/seatnow/results/edge/clips",
        "",
        "# ③ 노트북에서 — 복사한다",
        f"scp results/edge/clips/{skipped[0].name}.mp4 <사용자>@<박스IP>:~/seatnow/results/edge/clips/",
        "",
        "# ④ 박스에서 — 만들지 말고 재기만 한다",
        "./venv/bin/python -m edge.bench_decode --no-build",
        "```",
    ]
    return "\n".join(lines)


def ensure_clips(
    ffmpeg: str,
    source: Path,
    specs: Sequence[ClipSpec],
    duration: float,
    rebuild: bool,
    available_gb: float = 0.0,
    build: bool = True,
) -> Tuple[Dict[str, Path], List[SkippedClip]]:
    """Create the camera-shaped clips once and reuse them afterwards.

    Returns the clips that are ready and the ones this machine declined to
    build, so the caller can tell the user where to get them instead.
    """
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clips: Dict[str, Path] = {}
    skipped: List[SkippedClip] = []
    for spec in specs:
        destination = CLIP_DIR / spec.filename
        if destination.exists() and not rebuild:
            print(f"  재사용: {destination.name}")
            clips[spec.name] = destination
            continue
        if not build:
            print(f"  건너뜀: {destination.name} (--no-build)")
            skipped.append(SkippedClip(spec.name, "이 박스에 파일이 없다"))
            continue
        if not can_software_encode(spec, available_gb):
            need = encode_memory_gb(spec)
            print(
                f"  건너뜀: {destination.name} — 메모리 {available_gb:.1f}GB 로는 "
                f"못 만든다 (인코더가 {need:.1f}GB 쯤 쓴다)"
            )
            skipped.append(
                SkippedClip(
                    spec.name,
                    f"메모리 {available_gb:.1f}GB 로는 못 만든다 "
                    f"(인코더가 {need:.1f}GB 쯤 쓴다)",
                )
            )
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
            skipped.append(SkippedClip(spec.name, "만들다 실패했다"))
            continue
        clips[spec.name] = destination
    return clips, skipped


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
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "--build-only",
        action="store_true",
        help="클립만 만들고 측정은 하지 않는다 (넉넉한 노트북에서 만들어 박스로 보낼 때)",
    )
    build_group.add_argument(
        "--no-build",
        action="store_true",
        help="클립을 만들지 않고 이미 있는 것만 잰다 (박스에서 쓴다)",
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
    tolerant_stdout()
    args = build_parser().parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg를 찾지 못했다. docs/edge-setup.md 의 설치 절차를 볼 것.")
        return 1
    if not args.source.exists():
        print(f"원본 영상이 없다: {args.source}")
        return 1

    from edge.check_edge import _memory_gb

    total_cores = os.cpu_count() or 0
    memory_gb = _memory_gb()
    print(f"장비: {args.label}  ({platform.platform()})")
    print(f"코어 수: {total_cores if total_cores else '알 수 없음'}")
    print(f"메모리: {memory_gb:.1f}GB" if memory_gb > 0 else "메모리: 알 수 없음")
    print(f"원본: {args.source}\n")

    specs = [spec for spec in CLIP_SPECS if spec.name in set(args.clips)]
    print("### 벤치 클립 준비")
    clips, skipped = ensure_clips(
        ffmpeg,
        args.source,
        specs,
        args.duration,
        args.rebuild,
        available_gb=memory_gb,
        build=not args.no_build,
    )
    if args.build_only:
        for name in clips:
            print(f"  준비됨: {CLIP_DIR / (name + '.mp4')}")
        if skipped:
            print(format_missing_clips(skipped))
        return 0 if clips else 1
    if not clips:
        print("잴 수 있는 클립이 없다.")
        if skipped:
            print(format_missing_clips(skipped))
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
    decision = decision_rows(rows)
    print(format_decision_table(decision))

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
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n보고서: {args.report}")
    if skipped:
        print(format_missing_clips(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
