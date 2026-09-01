"""Export SeatNow's ultralytics weights to an edge-deployable format.

The pilot runs on a used mini-PC with an Intel iGPU, so OpenVINO is the target
runtime: it is the only backend that reaches both the CPU cores and the iGPU
without a CUDA stack.  Every export lands next to the source weights as a
directory ultralytics can load back (``yolov8n_openvino_model/``), which
``seatnow.py --det-model`` accepts unchanged.

    python export.py                        # detect + pose, FP32 and INT8
    python export.py --precision fp32       # skip the slow INT8 calibration
    python export.py --imgsz 640            # export for a 640 deployment

INT8 needs calibration frames.  Without ``--data`` ultralytics falls back to
its bundled COCO128 download; pass a dataset YAML of frames from the actual
camera when one exists, because post-training quantization inherits whatever
distribution it is calibrated on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = (("yolov8n.pt", "detect"), ("yolov8n-pose.pt", "pose"))


def _directory_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (
        1024 * 1024
    )


def export_openvino(
    weights: Path,
    task: str,
    imgsz: int,
    int8: bool,
    data: Optional[str],
    overwrite: bool,
) -> Dict[str, object]:
    """Export one checkpoint and report where it landed and how long it took."""
    from ultralytics import YOLO

    suffix = "_int8_openvino_model" if int8 else "_openvino_model"
    destination = weights.with_name(f"{weights.stem}{suffix}")
    if destination.exists():
        if not overwrite:
            print(f"  skip (exists): {destination.name}")
            return {
                "weights": str(weights),
                "task": task,
                "imgsz": imgsz,
                "precision": "int8" if int8 else "fp32",
                "output": str(destination),
                "size_mb": round(_directory_size_mb(destination), 2),
                "export_seconds": None,
                "status": "skipped",
            }
        shutil.rmtree(destination)

    model = YOLO(str(weights))
    started = time.perf_counter()
    kwargs: Dict[str, object] = {
        "format": "openvino",
        "imgsz": imgsz,
        "int8": int8,
        "half": False,
        "dynamic": False,
    }
    if int8 and data:
        kwargs["data"] = data
    produced = Path(model.export(**kwargs))
    elapsed = time.perf_counter() - started

    # Ultralytics names the INT8 output the same as FP32; move it aside so both
    # precisions can coexist and be benchmarked against each other.
    if produced != destination:
        if destination.exists():
            shutil.rmtree(destination)
        produced.rename(destination)

    print(f"  -> {destination.name}  ({_directory_size_mb(destination):.1f} MB, {elapsed:.1f}s)")
    return {
        "weights": str(weights),
        "task": task,
        "imgsz": imgsz,
        "precision": "int8" if int8 else "fp32",
        "output": str(destination),
        "size_mb": round(_directory_size_mb(destination), 2),
        "export_seconds": round(elapsed, 2),
        "status": "exported",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SeatNow weights to OpenVINO IR for the edge box.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--det-model", default="yolov8n.pt", help="Detection weights to export"
    )
    parser.add_argument(
        "--pose-model", default="yolov8n-pose.pt", help="Pose weights to export"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640, 960, 1280],
        help="Static input sizes to export (one export per size)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "int8", "both"],
        default="both",
        help="Which precisions to produce",
    )
    parser.add_argument(
        "--data",
        help="Dataset YAML used to calibrate INT8; omit to use ultralytics' default",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "pose", "both"],
        default="both",
        help="Which model(s) to export",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-export over existing output dirs"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "results" / "edge" / "export_report.json",
        help="Where to write the export manifest",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    targets = []
    if args.task in ("detect", "both"):
        targets.append((Path(args.det_model), "detect"))
    if args.task in ("pose", "both"):
        targets.append((Path(args.pose_model), "pose"))

    precisions = (
        [False, True]
        if args.precision == "both"
        else [args.precision == "int8"]
    )

    results: List[Dict[str, object]] = []
    for weights, task in targets:
        resolved = weights if weights.is_absolute() else PROJECT_DIR / weights
        if not resolved.exists():
            raise FileNotFoundError(
                f"{task} weights not found: {resolved}\n"
                "Download them first, e.g. "
                f"python -c \"from ultralytics import YOLO; YOLO('{weights.name}')\""
            )
        for imgsz in args.imgsz:
            for int8 in precisions:
                label = "int8" if int8 else "fp32"
                print(f"{task} {resolved.name} @{imgsz} {label}")
                results.append(
                    export_openvino(
                        resolved, task, imgsz, int8, args.data, args.overwrite
                    )
                )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nManifest: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
