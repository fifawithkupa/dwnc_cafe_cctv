"""Sweep SeatNow's cadence/resolution parameters for accuracy vs. tick cost.

plan.md T7: run the ``median_frames x sample_seconds x imgsz x table_crops``
grid over the labelled evaluation set, then pick the point that is the most
accurate among those that still fit inside a tick.

    python bench_sweep.py sample_raw/*.mp4 --expectations tests/fixtures
    python bench_sweep.py sample_raw/cafe.mp4 --imgsz 640 960 --dry-run

Prerequisites, in this order — skipping any of them makes the output
meaningless rather than merely noisy:

  T1  the chair-link regression is fixed, so there is a real baseline
  T3  the videos are labelled, so accuracy can be measured at all
  T6  ``bench.py`` has run on the edge box, so ``--sample-seconds`` is a real
      budget rather than a guess

Each grid point is a full ``seatnow.py`` run per video, so a 12-point grid over
two one-hour videos is an overnight job.  Start with ``--dry-run``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


from edge import tolerant_stdout

PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GridPoint:
    imgsz: int
    pose_imgsz: int
    median_frames: int
    sample_seconds: float
    table_crops: bool

    @property
    def name(self) -> str:
        crops = "crop" if self.table_crops else "nocrop"
        return (
            f"i{self.imgsz}_p{self.pose_imgsz}_m{self.median_frames}"
            f"_s{self.sample_seconds:g}_{crops}"
        )


@dataclass
class PointResult:
    point: Dict[str, object]
    name: str
    runtime_seconds: float
    frames_total: int
    frames_exact: int
    frame_accuracy: float
    ignore_ratio: float
    tick_seconds: float
    tick_utilization: float
    fits_tick: bool
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None


def expand_grid(
    imgsz: Sequence[int],
    pose_imgsz: Sequence[int],
    median_frames: Sequence[int],
    sample_seconds: Sequence[float],
    table_crops: Sequence[bool],
) -> List[GridPoint]:
    return [
        GridPoint(*combination)
        for combination in itertools.product(
            imgsz, pose_imgsz, median_frames, sample_seconds, table_crops
        )
    ]


def build_run_command(
    python: str,
    video: Path,
    point: GridPoint,
    log_path: Path,
    device: str,
    extra: Sequence[str] = (),
) -> List[str]:
    """Compose the seatnow.py invocation for one grid point.

    ``--no-video`` is not an optimisation here: writing an annotated MP4 for
    every grid point would dominate the runtime being measured, and plan.md
    T10 makes it the deployment default anyway.
    """
    command = [
        python,
        "-m",
        "engine.seatnow",
        str(video),
        "--no-video",
        "--log",
        str(log_path),
        "--imgsz",
        str(point.imgsz),
        "--pose-imgsz",
        str(point.pose_imgsz),
        "--median-frames",
        str(point.median_frames),
        "--sample-seconds",
        str(point.sample_seconds),
        "--device",
        device,
    ]
    command.append("--table-crops" if point.table_crops else "--no-table-crops")
    command.extend(extra)
    return command


def tick_utilization(point: GridPoint, runtime_seconds: float, samples: int) -> float:
    """Wall time per tick as a fraction of the tick length.

    Measured, not modelled: unlike ``bench.py``'s per-inference budget this
    includes decoding, tracking, and logging, which is what actually has to
    fit between two ticks on the edge box.
    """
    if samples <= 0 or point.sample_seconds <= 0:
        return 0.0
    return (runtime_seconds / samples) / point.sample_seconds


def run_point(
    point: GridPoint,
    videos: Sequence[Path],
    fixtures: Sequence[Dict[str, object]],
    work_dir: Path,
    python: str,
    device: str,
    extra: Sequence[str],
) -> PointResult:
    from checks.verify_seatnow import verify_suite

    logs: List[Path] = []
    started = time.perf_counter()
    for video in videos:
        log_path = work_dir / f"{video.stem}__{point.name}.jsonl"
        command = build_run_command(python, video, point, log_path, device, extra)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            return PointResult(
                point=asdict(point),
                name=point.name,
                runtime_seconds=time.perf_counter() - started,
                frames_total=0,
                frames_exact=0,
                frame_accuracy=0.0,
                ignore_ratio=0.0,
                tick_seconds=0.0,
                tick_utilization=0.0,
                fits_tick=False,
                logs=[str(path) for path in logs],
                error=(completed.stderr or completed.stdout)[-800:],
            )
        logs.append(log_path)
    runtime = time.perf_counter() - started

    suite = verify_suite(logs, fixtures)
    samples = suite["frames_total"]
    utilization = tick_utilization(point, runtime, samples)
    return PointResult(
        point=asdict(point),
        name=point.name,
        runtime_seconds=round(runtime, 2),
        frames_total=samples,
        frames_exact=suite["frames_exact"],
        frame_accuracy=round(suite["frame_accuracy"], 4),
        ignore_ratio=suite["ignore_ratio"],
        tick_seconds=round(runtime / samples, 2) if samples else 0.0,
        tick_utilization=round(utilization, 3),
        fits_tick=utilization <= 1.0,
        logs=[str(path) for path in logs],
    )


def rank(results: Sequence[PointResult]) -> List[PointResult]:
    """Most accurate first, but a point that misses its tick can never win."""
    return sorted(
        results,
        key=lambda result: (
            result.error is None,
            result.fits_tick,
            result.frame_accuracy,
            -result.tick_utilization,
        ),
        reverse=True,
    )


def print_table(results: Sequence[PointResult]) -> None:
    print("\n## 스윕 결과 (정확도 x tick 비용)\n")
    print("| 프로파일 | 정확도 | tick 소요 | tick 사용률 | ignore | tick 내 |")
    print("|---|---:|---:|---:|---:|:---:|")
    for result in results:
        if result.error:
            print(f"| {result.name} | ERROR | - | - | - | - |")
            continue
        print(
            f"| {result.name} | {result.frame_accuracy * 100:.1f}% "
            f"({result.frames_exact}/{result.frames_total}) "
            f"| {result.tick_seconds:.1f}s "
            f"| {result.tick_utilization * 100:.0f}% "
            f"| {result.ignore_ratio * 100:.1f}% "
            f"| {'예' if result.fits_tick else '아니오'} |"
        )
    winners = [r for r in rank(results) if r.error is None and r.fits_tick]
    if winners:
        best = winners[0]
        print(
            f"\n권고 배포 프로파일: **{best.name}** — "
            f"정확도 {best.frame_accuracy * 100:.1f}%, "
            f"tick의 {best.tick_utilization * 100:.0f}% 사용"
        )
        if best.tick_utilization > 0.5:
            print(
                "  ⚠️ tick의 50%를 넘는다. plan.md T6 기준으로는 CONDITIONAL이며, "
                "24/7 운영 시 RTSP 재연결·디코더 리셋 몫이 부족하다."
            )
    else:
        print("\ntick 안에 들어오는 조합이 없다. 해상도나 크롭을 더 내려야 한다.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("videos", type=Path, nargs="+", help="Labelled evaluation videos")
    parser.add_argument(
        "--expectations",
        type=Path,
        nargs="+",
        default=[PROJECT_DIR / "tests" / "fixtures"],
        help="Fixture file(s) or directory (see make_labels.py)",
    )
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument("--pose-imgsz", type=int, nargs="+", default=[640, 960])
    parser.add_argument("--median-frames", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--sample-seconds", type=float, nargs="+", default=[15.0])
    parser.add_argument(
        "--table-crops",
        nargs="+",
        default=["on", "off"],
        choices=["on", "off"],
        help="Whether to run the high-resolution crop pass",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", default=sys.executable, help="Interpreter for seatnow.py")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "edge" / "sweep",
        help="Where per-point JSONL logs are written",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra seatnow.py flags appended to every run (e.g. --layout ...)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "results" / "edge" / "sweep_report.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the grid and exit"
    )
    parser.add_argument(
        "--keep-logs", action="store_true", help="Do not delete per-point logs"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    tolerant_stdout()
    args = build_parser().parse_args(argv)
    from checks.verify_seatnow import discover_expectations

    grid = expand_grid(
        args.imgsz,
        args.pose_imgsz,
        args.median_frames,
        args.sample_seconds,
        [value == "on" for value in args.table_crops],
    )
    total_runs = len(grid) * len(args.videos)
    print(f"Grid: {len(grid)} points x {len(args.videos)} videos = {total_runs} runs")
    if args.dry_run:
        for point in grid:
            print(f"  {point.name}")
        return 0

    for video in args.videos:
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
    fixtures = discover_expectations(args.expectations)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    results: List[PointResult] = []
    for index, point in enumerate(grid, start=1):
        print(f"[{index}/{len(grid)}] {point.name}", flush=True)
        result = run_point(
            point,
            args.videos,
            fixtures,
            args.work_dir,
            args.python,
            args.device,
            args.extra,
        )
        if result.error:
            print(f"    ERROR: {result.error.splitlines()[-1] if result.error else ''}")
        else:
            print(
                f"    accuracy {result.frame_accuracy * 100:5.1f}%   "
                f"tick {result.tick_seconds:.1f}s "
                f"({result.tick_utilization * 100:.0f}%)"
            )
        results.append(result)

    print_table(rank(results))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nReport: {args.report}")
    if not args.keep_logs:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
