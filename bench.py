"""Measure SeatNow's per-inference latency and turn it into a tick budget.

The pilot's hard constraint is the 15-second sample interval: one tick must
finish every inference it needs before the next tick starts.  This script
measures each ``(task, imgsz, backend)`` combination once and then reports what
the measured numbers mean for a whole tick, so the same command produces a
comparable answer on the MacBook baseline and on the edge box.

    python bench.py                                   # every discovered backend
    python bench.py --frames sample_raw/cafe.mp4      # real frames, not noise
    python bench.py --backends pt --imgsz 640 1280    # quick pass

Grading follows plan.md T6: a profile that fits in half the tick passes, one
that fits in the whole tick is conditional, and anything above the tick fails.
The 50% margin exists because a 24/7 box also has to absorb RTSP reconnects,
decoder resets, and log rotation.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_SUFFIXES = {
    "pt": "",
    "ov-fp32": "_openvino_model",
    "ov-int8": "_int8_openvino_model",
}


@dataclass
class Measurement:
    task: str
    imgsz: int
    backend: str
    model: str
    median_ms: float
    p95_ms: float
    minimum_ms: float
    iterations: int


def resolve_model(stem_weights: Path, backend: str) -> Optional[Path]:
    """Return the weights path for one backend, or None if it was not exported."""
    if backend == "pt":
        return stem_weights if stem_weights.exists() else None
    candidate = stem_weights.with_name(f"{stem_weights.stem}{BACKEND_SUFFIXES[backend]}")
    return candidate if candidate.exists() else None


def load_frames(source: Optional[Path], count: int) -> List[np.ndarray]:
    """Take frames from a video/image, or fall back to synthetic noise.

    Content barely moves convolution cost, but it does move NMS cost, so a real
    cafe frame is preferred whenever one is available.
    """
    if source is None:
        rng = np.random.default_rng(0)
        return [
            rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
            for _ in range(count)
        ]

    import cv2

    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
        return [frame] * count

    from seatnow_core import FFmpegBurstReader, probe_video

    info = probe_video(source)
    reader = FFmpegBurstReader(source, info)
    step = max(info.duration / (count + 1), 0.1)
    frames: List[np.ndarray] = []
    for index in range(count):
        _, burst = reader.read_burst(step * (index + 1), 0)
        if burst:
            frames.append(burst[0][1])
    if not frames:
        raise RuntimeError(f"No frames decoded from {source}")
    return [frames[index % len(frames)] for index in range(count)]


def measure(
    model,
    frames: Sequence[np.ndarray],
    imgsz: int,
    device: str,
    warmup: int,
    iterations: int,
) -> List[float]:
    for index in range(warmup):
        model.predict(
            source=frames[index % len(frames)],
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    samples: List[float] = []
    for index in range(iterations):
        frame = frames[index % len(frames)]
        started = time.perf_counter()
        model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def run_grid(args: argparse.Namespace) -> List[Measurement]:
    from seatnow_core import load_model

    frames = load_frames(args.frames, args.iterations)
    weights = {
        "detect": Path(args.det_model),
        "pose": Path(args.pose_model),
    }
    results: List[Measurement] = []
    for task in args.tasks:
        stem = weights[task]
        stem = stem if stem.is_absolute() else PROJECT_DIR / stem
        for backend in args.backends:
            path = resolve_model(stem, backend)
            if path is None:
                print(f"  skip {task:6s} {backend:8s} (not exported)")
                continue
            model = load_model(path, task)
            for imgsz in args.imgsz:
                samples = measure(
                    model, frames, imgsz, args.device, args.warmup, args.iterations
                )
                ordered = sorted(samples)
                measurement = Measurement(
                    task=task,
                    imgsz=imgsz,
                    backend=backend,
                    model=path.name,
                    median_ms=round(statistics.median(samples), 1),
                    p95_ms=round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
                    minimum_ms=round(min(samples), 1),
                    iterations=len(samples),
                )
                results.append(measurement)
                print(
                    f"  {task:6s} {backend:8s} @{imgsz:<5d} "
                    f"median {measurement.median_ms:8.1f} ms   "
                    f"p95 {measurement.p95_ms:8.1f} ms"
                )
    return results


def lookup(
    results: Sequence[Measurement], task: str, imgsz: int, backend: str
) -> Optional[Measurement]:
    for measurement in results:
        if (
            measurement.task == task
            and measurement.imgsz == imgsz
            and measurement.backend == backend
        ):
            return measurement
    return None


def tick_budget(
    results: Sequence[Measurement],
    backend: str,
    imgsz: int,
    pose_imgsz: int,
    crop_imgsz: int,
    max_crops: int,
    median_frames: int,
    sample_seconds: float,
) -> Optional[Dict[str, object]]:
    """Cost one full tick from measured latencies, and grade it.

    A tick analyzes ``2 * median_frames + 1`` native frames, and each of those
    runs detect once, pose once, and up to ``max_crops`` crop detections.
    """
    detect = lookup(results, "detect", imgsz, backend)
    pose = lookup(results, "pose", pose_imgsz, backend)
    crop = lookup(results, "detect", crop_imgsz, backend) if max_crops else None
    if detect is None or pose is None or (max_crops and crop is None):
        return None

    frames_per_tick = 2 * median_frames + 1
    crop_ms = (crop.median_ms * max_crops) if crop else 0.0
    per_frame_ms = detect.median_ms + pose.median_ms + crop_ms
    tick_ms = per_frame_ms * frames_per_tick
    utilization = tick_ms / (sample_seconds * 1000.0)
    if utilization > 1.0:
        grade = "FAIL"
    elif utilization > 0.5:
        grade = "CONDITIONAL"
    else:
        grade = "PASS"
    return {
        "backend": backend,
        "imgsz": imgsz,
        "pose_imgsz": pose_imgsz,
        "crop_imgsz": crop_imgsz,
        "max_crops": max_crops,
        "median_frames": median_frames,
        "frames_per_tick": frames_per_tick,
        "inferences_per_tick": frames_per_tick * (2 + max_crops),
        "per_frame_ms": round(per_frame_ms, 1),
        "tick_seconds": round(tick_ms / 1000.0, 2),
        "sample_seconds": sample_seconds,
        "tick_utilization": round(utilization, 3),
        "grade": grade,
    }


PROFILES = {
    # (imgsz, pose_imgsz, crop_imgsz, max_crops, median_frames)
    "accuracy_default": (1280, 960, 960, 4, 2),
    "balanced": (960, 640, 640, 2, 2),
    "fast": (640, 640, 640, 0, 2),
    "minimum": (640, 640, 640, 0, 0),
}


def print_latency_table(results: Sequence[Measurement]) -> None:
    print("\n## 추론 latency (median ms)\n")
    backends = sorted({m.backend for m in results})
    sizes = sorted({m.imgsz for m in results})
    header = "| task | imgsz | " + " | ".join(backends) + " |"
    print(header)
    print("|---|---:|" + "---:|" * len(backends))
    for task in ("detect", "pose"):
        for imgsz in sizes:
            cells = []
            for backend in backends:
                found = lookup(results, task, imgsz, backend)
                cells.append(f"{found.median_ms:.0f}" if found else "-")
            if all(cell == "-" for cell in cells):
                continue
            print(f"| {task} | {imgsz} | " + " | ".join(cells) + " |")


def print_budget_table(budgets: Sequence[Dict[str, object]]) -> None:
    print("\n## tick 예산 (plan.md T6 합격 기준)\n")
    print("| profile | backend | 추론/tick | tick 소요 | tick 사용률 | 판정 |")
    print("|---|---|---:|---:|---:|:---:|")
    for budget in budgets:
        print(
            f"| {budget['profile']} | {budget['backend']} | "
            f"{budget['inferences_per_tick']} | {budget['tick_seconds']:.1f}s | "
            f"{budget['tick_utilization'] * 100:.0f}% | {budget['grade']} |"
        )
    print(
        "\nPASS = tick의 50% 이내 / CONDITIONAL = 50~100% / FAIL = 초과. "
        "여유 50%는 RTSP 재연결·디코더 리셋·로그 회전 몫이다."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SeatNow inference latency and derive a tick budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--det-model", default="yolov8n.pt")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKEND_SUFFIXES),
        choices=list(BACKEND_SUFFIXES),
        help="Backends to measure; missing exports are skipped, not an error",
    )
    parser.add_argument("--tasks", nargs="+", default=["detect", "pose"], choices=["detect", "pose"])
    parser.add_argument("--device", default="cpu", help="Ultralytics device")
    parser.add_argument("--frames", type=Path, help="Video or image to draw benchmark frames from")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--sample-seconds", type=float, default=15.0, help="Tick length the budget is graded against")
    parser.add_argument(
        "--label",
        default=platform.node(),
        help="Machine label recorded in the report (e.g. macbook, edge-box)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "bench" / "bench_report.json",
        help="Where to write the machine-readable report",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Machine: {args.label}  ({platform.platform()}, {platform.processor()})")
    print(f"Frames: {args.frames or 'synthetic noise'}  device={args.device}\n")

    results = run_grid(args)
    if not results:
        print("No models measured. Run export.py first, or pass --backends pt.")
        return 1
    print_latency_table(results)

    budgets: List[Dict[str, object]] = []
    for name, (imgsz, pose_imgsz, crop_imgsz, max_crops, median_frames) in PROFILES.items():
        for backend in sorted({m.backend for m in results}):
            budget = tick_budget(
                results,
                backend,
                imgsz,
                pose_imgsz,
                crop_imgsz,
                max_crops,
                median_frames,
                args.sample_seconds,
            )
            if budget is not None:
                budgets.append(dict(budget, profile=name))
    if budgets:
        print_budget_table(budgets)
    else:
        print("\ntick 예산: 필요한 imgsz 조합이 측정되지 않았다 (--imgsz 확인).")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "label": args.label,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "device": args.device,
                "frames": str(args.frames) if args.frames else None,
                "sample_seconds": args.sample_seconds,
                "measurements": [asdict(m) for m in results],
                "tick_budgets": budgets,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
