"""SeatNow image/video occupancy command line application.

Examples:
  ./venv/bin/python seatnow.py cafe.jpg
  ./venv/bin/python seatnow.py cafe.mp4 --sample-seconds 2 --debug
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

from seatnow_core import (
    AdaptiveCadenceController,
    AnalyzerConfig,
    FFmpegBurstReader,
    FFmpegSampleReader,
    FFmpegVideoWriter,
    OccupancyState,
    SeatNowAnalyzer,
    TableTracker,
    aggregate_burst_observations,
    frame_log_record,
    is_scene_change,
    probe_video,
    render_frame,
)
from seatnow_layout import load_layout


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_DIR = Path(__file__).resolve().parent


def _model_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        local = PROJECT_DIR / candidate
        if local.exists():
            return local
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect per-table occupancy in an image or sampled video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input image or video")
    parser.add_argument("--output", type=Path, help="Annotated JPG/MP4 output path")
    parser.add_argument("--log", type=Path, help="JSONL output path (video only)")
    parser.add_argument("--det-model", default="yolov8n.pt", help="Ultralytics detect weights")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Ultralytics pose weights")
    parser.add_argument("--sample-seconds", type=float, default=15.0, help="Base seconds between analyzed samples (1차 판단 주기)")
    parser.add_argument("--fast-sample-seconds", type=float, default=5.0, help="Accelerated cadence while an empty seat shows occupied evidence pending confirmation (2차 판단 주기)")
    parser.add_argument("--median-frames", type=int, default=2, help="Analyze ±N consecutive native-fps frames per sample and majority-vote per-seat states (0 = single frame)")
    parser.add_argument("--no-adaptive", action="store_true", help="Pin the cadence to --sample-seconds instead of accelerating on occupancy evidence")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start analysis at this media timestamp")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--pose-imgsz", type=int, default=960, help="Pose-model inference image size")
    parser.add_argument("--table-conf", type=float, default=0.20, help="Table confidence threshold")
    parser.add_argument("--max-table-area", type=float, default=0.06, help="Soft cap: table boxes above this frame-area fraction need high confidence or supporting chairs")
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument("--table-crops", dest="table_crops", action="store_true", help="Run a slower high-resolution object pass around bare tables")
    crop_group.add_argument("--no-table-crops", dest="table_crops", action="store_false", help="Disable the high-resolution table crop pass")
    parser.set_defaults(table_crops=True)
    parser.add_argument("--crop-imgsz", type=int, default=960, help="Inference size for --table-crops")
    parser.add_argument("--crop-conf", type=float, default=0.12, help="Object confidence for --table-crops")
    parser.add_argument("--max-crops", type=int, default=4, help="Maximum table crops analyzed per sampled frame")
    parser.add_argument("--object-conf", type=float, default=0.15, help="Customer-object confidence threshold")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Pose person confidence threshold")
    parser.add_argument("--keypoint-conf", type=float, default=0.30, help="Pose keypoint confidence threshold")
    parser.add_argument("--angle", type=float, default=110.0, help="Sitting angle threshold in degrees")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--occupy-confirm", type=int, default=2, help="Samples required to confirm an occupied state change")
    parser.add_argument("--empty-confirm", type=int, default=3, help="Samples required to confirm an empty state change")
    parser.add_argument("--track-ttl", type=int, default=3, help="Missed samples retained without becoming empty")
    parser.add_argument("--no-inferred-seats", action="store_true", help="Disable occupied fallback for a seated person whose table is occluded")
    parser.add_argument("--no-video", action="store_true", help="Write the JSONL log but skip annotated MP4")
    parser.add_argument("--debug", action="store_true", help="Draw pose and tabletop diagnostics")
    parser.add_argument("--no-scene-reset", action="store_true", help="Disable automatic scene-cut reset")
    parser.add_argument("--max-samples", type=int, help="Stop after N sampled frames (smoke tests)")
    parser.add_argument("--layout", type=Path, help="Manual seat layout JSON (calibrate.py output); zones become ground truth")
    parser.add_argument("--no-layout-track", action="store_true", help="Keep layout zones fixed instead of drifting toward matching detections")
    return parser


def _default_output(input_path: Path, is_video: bool) -> Path:
    suffix = ".mp4" if is_video else ".jpg"
    if input_path.parent.name == "sample_raw":
        result_dir = input_path.parent.parent / "sample_results"
        return result_dir / f"{input_path.stem}_seatnow{suffix}"
    return input_path.with_name(f"{input_path.stem}_seatnow{suffix}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be positive")
    if args.fast_sample_seconds <= 0:
        raise ValueError("--fast-sample-seconds must be positive")
    if not args.no_adaptive and args.fast_sample_seconds > args.sample_seconds:
        raise ValueError("--fast-sample-seconds cannot exceed --sample-seconds")
    if args.median_frames < 0:
        raise ValueError("--median-frames cannot be negative")
    if args.start_seconds < 0:
        raise ValueError("--start-seconds cannot be negative")
    if args.imgsz < 320 or args.pose_imgsz < 320 or args.crop_imgsz < 320:
        raise ValueError("--imgsz, --pose-imgsz and --crop-imgsz must be at least 320")
    if args.max_crops < 0:
        raise ValueError("--max-crops cannot be negative")
    if args.occupy_confirm < 1 or args.empty_confirm < 1:
        raise ValueError("--occupy-confirm and --empty-confirm must be at least 1")
    if args.track_ttl < 0:
        raise ValueError("--track-ttl cannot be negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.layout is not None and not args.layout.exists():
        raise FileNotFoundError(f"Layout not found: {args.layout}")
    for label, value in (
        ("--table-conf", args.table_conf),
        ("--object-conf", args.object_conf),
        ("--pose-conf", args.pose_conf),
        ("--keypoint-conf", args.keypoint_conf),
        ("--max-table-area", args.max_table_area),
        ("--crop-conf", args.crop_conf),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1")


def _make_analyzer(args: argparse.Namespace, layout=None) -> SeatNowAnalyzer:
    det_path = _model_path(args.det_model)
    pose_path = _model_path(args.pose_model)
    if not det_path.exists():
        raise FileNotFoundError(f"Detection model not found: {det_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose model not found: {pose_path}")
    config = AnalyzerConfig(
        imgsz=args.imgsz,
        pose_imgsz=args.pose_imgsz,
        table_confidence=args.table_conf,
        object_confidence=args.object_conf,
        pose_confidence=args.pose_conf,
        keypoint_confidence=args.keypoint_conf,
        sitting_angle=args.angle,
        maximum_table_area_fraction=args.max_table_area,
        table_crop_objects=args.table_crops,
        table_crop_imgsz=args.crop_imgsz,
        table_crop_confidence=args.crop_conf,
        maximum_table_crops=args.max_crops,
        infer_occluded_tables=not args.no_inferred_seats,
        layout_tracking=not args.no_layout_track,
        device=args.device,
    )
    print(f"Loading detector: {det_path.name}", flush=True)
    print(f"Loading pose model: {pose_path.name}", flush=True)
    return SeatNowAnalyzer(det_path, pose_path, config, layout=layout)


def process_image(args: argparse.Namespace, analyzer: SeatNowAnalyzer) -> int:
    frame = cv2.imread(str(args.input))
    if frame is None:
        raise RuntimeError(f"OpenCV could not read image: {args.input}")
    if analyzer.layout is not None:
        height, width = frame.shape[:2]
        analyzer.layout = analyzer.layout.scaled_to(width, height)
    analysis = analyzer.analyze(frame, timestamp=0.0)
    tracker = TableTracker(occupy_confirmations=1, empty_confirmations=1, max_missed=0)
    update = tracker.update(analysis.tables, 0.0, frame.shape[:2])
    rendered = render_frame(frame, analysis, update, debug=args.debug)
    output = args.output or _default_output(args.input, is_video=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"Failed to write image: {output}")
    record = frame_log_record(0, analysis, update)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Annotated image: {output}")
    return 0


def process_video(args: argparse.Namespace, analyzer: SeatNowAnalyzer) -> int:
    info = probe_video(args.input)
    if analyzer.layout is not None:
        analyzer.layout = analyzer.layout.scaled_to(info.width, info.height)
    output = args.output or _default_output(args.input, is_video=True)
    log_path = args.log or output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Video: {info.width}x{info.height}, {info.fps:.3f} fps, "
        f"{info.duration:.3f}s, codec={info.codec}",
        flush=True,
    )
    adaptive = not args.no_adaptive
    # --median-frames 0 --no-adaptive reproduces the original fixed-interval
    # single-frame pipeline exactly (streaming reader, no vote fields).
    legacy_mode = args.median_frames == 0 and not adaptive
    run_context = {
        "profile": (
            "accuracy_default"
            if (
                args.imgsz == 1280
                and args.pose_imgsz == 960
                and args.table_conf == 0.20
                and args.object_conf == 0.15
                and args.table_crops
                and args.crop_imgsz == 960
                and args.crop_conf == 0.12
                and args.max_crops == 4
                and args.pose_conf == 0.20
                and args.keypoint_conf == 0.30
                and args.angle == 110.0
                and args.max_table_area == 0.06
                and not args.no_inferred_seats
                and not args.no_scene_reset
                and args.median_frames == 2
                and args.fast_sample_seconds == 5.0
                and not args.no_adaptive
            )
            else ("fast" if not args.table_crops and args.imgsz <= 960 else "custom")
        ),
        "input": {
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "duration": info.duration,
            "codec": info.codec,
        },
        "models": {
            "detector": str(analyzer.det_model_path),
            "detector_sha256": _sha256(analyzer.det_model_path),
            "pose": str(analyzer.pose_model_path),
            "pose_sha256": _sha256(analyzer.pose_model_path),
        },
        "config": {
            "sample_seconds": args.sample_seconds,
            "fast_sample_seconds": args.fast_sample_seconds,
            "median_frames": args.median_frames,
            "adaptive_cadence": adaptive,
            "start_seconds": args.start_seconds,
            "imgsz": args.imgsz,
            "pose_imgsz": args.pose_imgsz,
            "table_confidence": args.table_conf,
            "object_confidence": args.object_conf,
            "pose_confidence": args.pose_conf,
            "keypoint_confidence": args.keypoint_conf,
            "sitting_angle": args.angle,
            "maximum_table_area_fraction": args.max_table_area,
            "large_table_confidence": analyzer.config.large_table_confidence,
            "hard_table_area_fraction": analyzer.config.hard_table_area_fraction,
            "table_rescue_confidence": analyzer.config.table_rescue_confidence,
            "table_crop_objects": args.table_crops,
            "table_crop_imgsz": args.crop_imgsz,
            "table_crop_confidence": args.crop_conf,
            "maximum_table_crops": args.max_crops,
            "infer_occluded_tables": not args.no_inferred_seats,
            "occupy_confirmations": args.occupy_confirm,
            "empty_confirmations": args.empty_confirm,
            "track_ttl": args.track_ttl,
            "scene_reset": not args.no_scene_reset,
            "max_samples": args.max_samples,
            "device": args.device,
            "layout": (
                {"path": str(args.layout), "sha256": _sha256(args.layout)}
                if args.layout
                else None
            ),
            "layout_tracking": bool(args.layout) and not args.no_layout_track,
        },
    }
    if legacy_mode:
        print(
            f"Analyzing every {args.sample_seconds:g}s "
            f"(result video {1.0 / args.sample_seconds:.3f} fps)",
            flush=True,
        )
    else:
        cadence_note = (
            f", {args.fast_sample_seconds:g}s while confirming occupancy"
            if adaptive
            else ""
        )
        print(
            f"Analyzing every {args.sample_seconds:g}s "
            f"(±{args.median_frames} frame majority vote{cadence_note})",
            flush=True,
        )

    def new_tracker() -> TableTracker:
        return TableTracker(
            occupy_confirmations=args.occupy_confirm,
            empty_confirmations=args.empty_confirm,
            max_missed=args.track_ttl,
        )

    tracker = new_tracker()
    # Burst mode writes exactly one rendered frame per judgment — fast 5s
    # rechecks appear as their own frames (tagged in the header) instead of
    # being outweighed by duplicated base-cadence frames.
    writer_fps = (1.0 / args.sample_seconds) if legacy_mode else 1.0
    writer: Optional[FFmpegVideoWriter] = None
    if not args.no_video:
        writer = FFmpegVideoWriter(
            output,
            width=info.width,
            height=info.height,
            fps=writer_fps,
        )

    started = time.perf_counter()
    processed = 0
    scene_id = 1
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            if legacy_mode:
                reader = FFmpegSampleReader(
                    args.input,
                    args.sample_seconds,
                    info,
                    start_seconds=args.start_seconds,
                )
                previous_sample = None
                for frame_index, timestamp, frame in reader:
                    scene_changed = False
                    scene_metrics = {}
                    if previous_sample is not None and not args.no_scene_reset:
                        scene_changed, scene_metrics = is_scene_change(previous_sample, frame)
                    if scene_changed:
                        analyzer.reset_temporal()
                    analysis = analyzer.analyze(
                        frame,
                        timestamp,
                        global_motion_fraction=(
                            scene_metrics.get("global_dx_fraction", 0.0),
                            scene_metrics.get("global_dy_fraction", 0.0),
                        ),
                    )
                    analysis.scene_change = scene_changed
                    analysis.scene_metrics = scene_metrics
                    if scene_changed:
                        next_track_id = tracker.next_id
                        tracker = new_tracker()
                        tracker.next_id = next_track_id
                        scene_id += 1
                        for observation in analysis.tables:
                            observation.raw_state = OccupancyState.IGNORE
                            observation.reason = "scene_transition"
                    update = tracker.update(analysis.tables, timestamp, frame.shape[:2])
                    if scene_changed:
                        update.events.insert(
                            0,
                            {
                                "type": "scene_change",
                                "timestamp": timestamp,
                                "tracker_reset": True,
                            },
                        )
                    record = frame_log_record(frame_index, analysis, update)
                    record["scene_id"] = scene_id
                    for table in record["tables"]:
                        table["scene_id"] = scene_id
                    for event in record["events"]:
                        event["scene_id"] = scene_id
                    record["run"] = run_context
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    if writer is not None:
                        writer.write(render_frame(frame, analysis, update, debug=args.debug))
                    processed += 1
                    previous_sample = frame
                    summary = record["summary"]
                    print(
                        f"[{timestamp:5.1f}s] tables={summary['visible']} "
                        f"occupied={summary['occupied']} empty={summary['empty']} "
                        f"ignore={summary['ignore']} poses(seated/standing/unknown)="
                        f"{summary['seated_poses']}/{summary['standing_poses']}/{summary['unknown_poses']} "
                        f"inference={analysis.inference_ms:.0f}ms",
                        flush=True,
                    )
                    if args.max_samples is not None and processed >= args.max_samples:
                        break
            else:
                reader = FFmpegBurstReader(args.input, info)
                controller = AdaptiveCadenceController(
                    base_seconds=args.sample_seconds,
                    fast_seconds=(
                        args.fast_sample_seconds if adaptive else args.sample_seconds
                    ),
                )
                previous_center = None
                center_time = args.start_seconds
                scheduled_fast = False
                while info.duration <= 0 or center_time <= info.duration:
                    center_index, burst = reader.read_burst(
                        center_time, args.median_frames
                    )
                    if not burst:
                        break
                    center_frame = burst[center_index][1]
                    scene_changed = False
                    scene_metrics = {}
                    if previous_center is not None and not args.no_scene_reset:
                        scene_changed, scene_metrics = is_scene_change(
                            previous_center, center_frame
                        )
                    if scene_changed:
                        analyzer.reset_temporal()
                    # Side frames vote without committing temporal state; the
                    # center frame runs last so zone drift and pose history
                    # advance once per sample, against the previous sample.
                    frame_tables: list = [[] for _ in burst]
                    side_inference_ms = 0.0
                    for position, (frame_timestamp, side_frame) in enumerate(burst):
                        if position == center_index:
                            continue
                        side_analysis = analyzer.analyze(
                            side_frame,
                            frame_timestamp,
                            update_temporal=False,
                        )
                        frame_tables[position] = side_analysis.tables
                        side_inference_ms += side_analysis.inference_ms
                    analysis = analyzer.analyze(
                        center_frame,
                        center_time,
                        global_motion_fraction=(
                            scene_metrics.get("global_dx_fraction", 0.0),
                            scene_metrics.get("global_dy_fraction", 0.0),
                        ),
                    )
                    frame_tables[center_index] = analysis.tables
                    analysis.tables = aggregate_burst_observations(
                        frame_tables, center_index
                    )
                    analysis.inference_ms += side_inference_ms
                    analysis.scene_change = scene_changed
                    analysis.scene_metrics = scene_metrics
                    if scene_changed:
                        next_track_id = tracker.next_id
                        tracker = new_tracker()
                        tracker.next_id = next_track_id
                        scene_id += 1
                        for observation in analysis.tables:
                            observation.raw_state = OccupancyState.IGNORE
                            observation.reason = "scene_transition"
                    update = tracker.update(
                        analysis.tables, center_time, center_frame.shape[:2]
                    )
                    if scene_changed:
                        update.events.insert(
                            0,
                            {
                                "type": "scene_change",
                                "timestamp": center_time,
                                "tracker_reset": True,
                            },
                        )
                    fast_mode = adaptive and AdaptiveCadenceController.wants_fast(
                        update.all_tracks
                    )
                    next_interval = controller.next_interval(update.all_tracks)
                    record = frame_log_record(processed, analysis, update)
                    record["scene_id"] = scene_id
                    for table in record["tables"]:
                        table["scene_id"] = scene_id
                    for event in record["events"]:
                        event["scene_id"] = scene_id
                    record["cadence"] = {
                        "mode": "fast" if fast_mode else "base",
                        "next_interval_seconds": next_interval,
                        "burst_frames": len(burst),
                    }
                    record["run"] = run_context
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    if writer is not None:
                        writer.write(
                            render_frame(
                                center_frame,
                                analysis,
                                update,
                                debug=args.debug,
                                cadence="fast" if scheduled_fast else None,
                            )
                        )
                    processed += 1
                    previous_center = center_frame
                    summary = record["summary"]
                    print(
                        f"[{center_time:5.1f}s] tables={summary['visible']} "
                        f"occupied={summary['occupied']} empty={summary['empty']} "
                        f"ignore={summary['ignore']} poses(seated/standing/unknown)="
                        f"{summary['seated_poses']}/{summary['standing_poses']}/{summary['unknown_poses']} "
                        f"inference={analysis.inference_ms:.0f}ms "
                        f"cadence={'fast' if fast_mode else 'base'}",
                        flush=True,
                    )
                    if args.max_samples is not None and processed >= args.max_samples:
                        break
                    scheduled_fast = fast_mode
                    center_time += next_interval
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except Exception as cleanup_error:
                print(
                    f"SeatNow cleanup warning: {cleanup_error}",
                    file=sys.stderr,
                )
        raise
    else:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started
    if processed == 0:
        raise RuntimeError("No frames were decoded from the input video")
    print(f"Completed {processed} samples in {elapsed:.1f}s")
    if writer is not None:
        print(f"Annotated video: {output}")
    print(f"JSONL log: {log_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        suffix = args.input.suffix.lower()
        if suffix not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported input extension: {suffix or '(none)'}")
        layout = load_layout(args.layout) if args.layout else None
        analyzer = _make_analyzer(args, layout=layout)
        if suffix in VIDEO_SUFFIXES:
            return process_video(args, analyzer)
        return process_image(args, analyzer)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"SeatNow error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
