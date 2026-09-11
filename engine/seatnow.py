"""SeatNow image/video occupancy command line application.

Examples:
  ./venv/bin/python seatnow.py cafe.jpg
  ./venv/bin/python seatnow.py cafe.mp4 --sample-seconds 2 --debug
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

from engine.seatnow_core import (
    AnalyzerConfig,
    FFmpegBurstReader,
    FFmpegSampleReader,
    FFmpegVideoWriter,
    OccupancyState,
    SeatNowAnalyzer,
    TableTracker,
    aggregate_burst_observations,
    frame_log_record,
    model_backend,
    is_scene_change,
    probe_video,
    render_frame,
)
from engine.frame_dump import save_frame_pair
from engine.seatnow_hwaccel import HWACCEL_AUTO, HWACCEL_CHOICES, resolve_hwaccel
from engine.seatnow_layout import LayoutError, load_layout
from engine.seatnow_live import (
    FFmpegLiveReader,
    LiveLogRotator,
    TickSchedule,
    is_live_source,
    burst_period_frames,
    probe_live_hwaccel,
    redact_url,
    resolve_max_frame_age,
    should_disable_hwaccel,
    probe_stream,
    process_rss_mb,
)
from edge.publish import (
    _publish_stats_for_record,
    box_version,
    gap_payload,
    live_payload,
    publisher_from_env,
    seat_index_from_layout,
)


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _model_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        local = PROJECT_DIR / candidate
        if local.exists():
            return local
    return candidate


def _input_argument(value: str):
    """Streams stay strings — ``Path`` would fold ``rtsp://`` into ``rtsp:/``."""
    if is_live_source(value):
        return value
    return Path(value)


def live_log_default(url: str) -> Path:
    """Where a live run writes its JSONL when ``--log`` is not given."""
    return PROJECT_DIR / "results" / "live" / "log.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect per-table occupancy in an image or sampled video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=_input_argument, help="Input image, video, or rtsp:// stream")
    parser.add_argument("--output", type=Path, help="Annotated JPG/MP4 output path")
    parser.add_argument("--log", type=Path, help="JSONL output path (video only)")
    parser.add_argument("--det-model", default="yolov8n.pt", help="Ultralytics detect weights")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Ultralytics pose weights")
    parser.add_argument("--sample-seconds", type=float, default=15.0, help="Seconds between analyzed samples (판단 주기)")
    parser.add_argument("--median-frames", type=int, default=2, help="Analyze ±N consecutive native-fps frames per sample and majority-vote per-seat states (0 = single frame)")
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
    parser.add_argument("--seat-conf", type=float, default=0.20, help="Chair/couch/bench confidence threshold for chair linking")
    parser.add_argument("--strong-chair-link", type=float, default=0.75, help="Chair-to-table link score required to propagate occupancy")
    parser.add_argument("--angle", type=float, default=110.0, help="Sitting angle threshold in degrees")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--occupy-confirm", type=int, default=2, help="Samples required to confirm an occupied state change")
    parser.add_argument("--empty-confirm", type=int, default=3, help="Samples required to confirm an empty state change")
    parser.add_argument("--track-ttl", type=int, default=3, help="Missed samples retained without becoming empty")
    parser.add_argument("--table-layout-add-confirm", type=int, default=3, help="Repeated samples required before a new detected table joins the active layout")
    parser.add_argument("--table-layout-move-confirm", type=int, default=3, help="Repeated samples required before a moved table bbox is committed")
    parser.add_argument("--table-layout-remove-confirm", type=int, default=3, help="Repeated missed samples required before a table is retired from the active layout")
    parser.add_argument("--table-layout-bbox-alpha", type=float, default=0.25, help="EMA alpha used to smooth stable table layout boxes")
    parser.add_argument("--no-inferred-seats", action="store_true", help="Disable occupied fallback for a seated person whose table is occluded")
    parser.add_argument("--no-video", action="store_true", help="Write the JSONL log but skip annotated MP4")
    parser.add_argument("--debug", action="store_true", help="Draw pose and tabletop diagnostics")
    parser.add_argument("--log-detections", action="store_true", help="Add the detector's raw output and dropped-table rules to each JSONL record (diagnoses model miss vs. code rejection)")
    parser.add_argument("--no-scene-reset", action="store_true", help="Disable automatic scene-cut reset")
    parser.add_argument("--max-samples", type=int, help="Stop after N sampled frames (smoke tests)")
    parser.add_argument("--log-dir", type=Path, help="실시간 입력용: 이 폴더에 날짜별 JSONL(YYYY-MM-DD.jsonl)로 이어 쓴다. 재시작해도 그날 파일에 붙고 자정에 새 파일로 넘어간다 (--log 와 같이 못 씀)")
    parser.add_argument("--keep-days", type=int, default=14, help="--log-dir 에서 이 날짜보다 오래된 파일은 지운다 (0 = 안 지움)")
    parser.add_argument("--telemetry-dir", type=Path, default=None,
                        help="판정 개선용 기록을 이 폴더에 쌓는다 (없으면 안 쌓는다). "
                             "보내는 건 따로 돈다: python3 -m edge.telemetry_upload")
    parser.add_argument("--open-hours", type=str, default=None,
                        help='매장 영업시간 "09:00-22:00". 문 닫은 시간을 알아야 '
                             '장식과 손님 짐을 구분할 수 있다 (문서/판정개선_데이터설계.md §4-1)')
    parser.add_argument("--telemetry-control-rate", type=float, default=0.01,
                        help="잘 돌아간 틱 중 대조군으로 남길 비율 (기본 1%%). "
                             "어려운 것만 모으면 배운 규칙이 편향된다")
    parser.add_argument("--run-seconds", type=float, help="실시간(rtsp://) 입력일 때 이 시간이 지나면 멈춘다 (없으면 계속 돈다)")
    parser.add_argument("--max-frame-age-seconds", type=float, default=None, help="실시간 입력에서 이보다 오래된 화면은 없는 것으로 친다 (기본: 판단 주기와 같음, 0 = 끔). 끊긴 카메라의 옛 화면으로 판정하지 않기 위한 것")
    parser.add_argument("--live-burst-seconds", type=float, default=5.0, help="실시간 입력에서 몇 초마다 프레임 묶음 하나를 변환할지 (0 = 모든 프레임 변환; 2코어 박스에서는 5초가 맞다)")
    parser.add_argument(
        "--hwaccel",
        default=HWACCEL_AUTO,
        choices=HWACCEL_CHOICES,
        help="영상 디코딩에 쓸 하드웨어 가속기 (auto = OS에 맞는 것을 실제로 시험해보고 고름)",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        help="판정한 tick마다 사진 두 장을 이 폴더에 저장한다 "
             "(clean/ = 박스 없음, marked/ = 판정 그려짐). --no-video와 같이 쓴다",
    )
    parser.add_argument("--layout", type=Path, help="Manual seat layout JSON (calibrate.py output); zones become ground truth")
    parser.add_argument("--no-layout-track", action="store_true", help="Keep layout zones fixed instead of drifting toward matching detections")
    return parser


def _require_complete_layout(layout) -> None:
    """Refuse to judge with a bar zone nobody sliced into seat slots.

    Capacity is the number of seat slots, so such a zone contributes no
    judgement unit: the bar would not be reported as occupied, empty, or even
    unknown -- it would simply be absent, and the app would show a cafe with
    fewer seats than it has.  Saving that state is fine (the install may be
    half done); running on it is the failure worth stopping for.
    """
    if layout is None:
        return None
    incomplete = layout.incomplete_zones()
    if incomplete:
        raise LayoutError(
            f"자리 칸이 없는 바 구역이 있어 실행할 수 없습니다: "
            f"{', '.join(incomplete)} — calibrate.py 에서 그 구역을 선택하고 "
            f"seat[x] 로 자리마다 칸을 그으세요. 칸 개수가 곧 자리 수입니다. "
            f"(칸이 없으면 그 구역의 좌석이 아무 집계에도 안 잡힙니다)"
        )
    return None


def _default_output(input_path: Path, is_video: bool) -> Path:
    suffix = ".mp4" if is_video else ".jpg"
    if input_path.parent.name == "sample_raw":
        result_dir = input_path.parent.parent / "results" / input_path.stem
        return result_dir / f"annotated{suffix}"
    return input_path.with_name(f"{input_path.stem}_seatnow{suffix}")


def _sha256(path: Path) -> str:
    """Digest a weights file, or a whole exported-model directory.

    OpenVINO/NCNN exports are directories, so a run's model provenance can no
    longer assume a single file.
    """
    digest = hashlib.sha256()
    if path.is_dir():
        for member in sorted(path.rglob("*")):
            if member.is_file():
                digest.update(member.relative_to(path).as_posix().encode("utf-8"))
                digest.update(_sha256(member).encode("ascii"))
        return digest.hexdigest()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    live = is_live_source(args.input)
    if not live and not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.run_seconds is not None:
        if not live:
            raise ValueError("--run-seconds 는 실시간(rtsp://) 입력에만 쓴다")
        if args.run_seconds <= 0:
            raise ValueError("--run-seconds must be positive")
    if getattr(args, "live_burst_seconds", 0.0) is not None and args.live_burst_seconds < 0:
        raise ValueError("--live-burst-seconds cannot be negative")
    if getattr(args, "max_frame_age_seconds", None) is not None and args.max_frame_age_seconds < 0:
        raise ValueError("--max-frame-age-seconds cannot be negative")
    if getattr(args, "log_dir", None) is not None:
        if not live:
            raise ValueError("--log-dir 는 실시간(rtsp://) 입력에만 쓴다")
        if args.log is not None:
            raise ValueError("--log-dir 와 --log 는 같이 쓸 수 없다")
    if getattr(args, "keep_days", 0) is not None and args.keep_days < 0:
        raise ValueError("--keep-days cannot be negative")
    if args.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be positive")
    if args.median_frames < 0:
        raise ValueError("--median-frames cannot be negative")
    if not 0.0 <= args.telemetry_control_rate <= 1.0:
        raise ValueError("--telemetry-control-rate must be between 0 and 1")
    if args.open_hours is not None:
        from edge.telemetry import parse_open_window

        if parse_open_window(args.open_hours) is None:
            raise ValueError('--open-hours must look like "09:00-22:00"')
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
    if (
        args.table_layout_add_confirm < 1
        or args.table_layout_move_confirm < 1
        or args.table_layout_remove_confirm < 1
    ):
        raise ValueError("--table-layout-*-confirm values must be at least 1")
    if not 0.0 <= args.table_layout_bbox_alpha <= 1.0:
        raise ValueError("--table-layout-bbox-alpha must be between 0 and 1")
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
        ("--seat-conf", args.seat_conf),
        ("--strong-chair-link", args.strong_chair_link),
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
        seat_confidence=args.seat_conf,
        strong_chair_link=args.strong_chair_link,
        layout_tracking=not args.no_layout_track,
        device=args.device,
    )
    print(
        f"Loading detector: {det_path.name} [{model_backend(det_path)}]", flush=True
    )
    print(
        f"Loading pose model: {pose_path.name} [{model_backend(pose_path)}]", flush=True
    )
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
    record = frame_log_record(
        0, analysis, update, include_raw_detections=args.log_detections
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Annotated image: {output}")
    return 0


def _open_telemetry(args, run_context, cafe_id: str):
    """`--telemetry-dir` 를 줬을 때만 기록기를 만든다.  아니면 None.

    설계는 `문서/판정개선_데이터설계.md` §9·§10.  루프는 파일에 덧붙이기만 하고,
    보내는 건 `python3 -m edge.telemetry_upload` 가 하루 한 번 따로 한다.
    만들다 실패해도 판정은 그대로 돈다 — 기록은 판정보다 덜 중요하다.
    """
    directory = getattr(args, "telemetry_dir", None)
    if directory is None:
        return None
    try:
        from edge.telemetry_spool import TelemetryWriter

        writer = TelemetryWriter(
            directory,
            cafe_id=cafe_id or "unknown",
            run_context=run_context,
            open_hours=getattr(args, "open_hours", None),
            control_rate=getattr(args, "telemetry_control_rate", None),
        )
        print(f"판정 기록: {directory} (보내기는 edge.telemetry_upload 가 따로)", flush=True)
        return writer
    except Exception as error:  # noqa: BLE001 -- 기록 때문에 판정을 막지 않는다
        print(f"판정 기록을 못 열었다(무시하고 계속): {type(error).__name__}: {error}", flush=True)
        return None


class _TickRunner:
    """One judgement per burst of frames — the body shared by the file loop
    and the live loop.  Holds the tracker, scene counter and the previous
    centre frame so both loops behave identically after a scene cut."""

    #: 판정 개선용 기록.  `--telemetry-dir` 를 줬을 때만 채워진다.
    telemetry = None

    def __init__(self, args, analyzer, new_tracker, run_context, log_file, writer):
        self.args = args
        self.analyzer = analyzer
        self.new_tracker = new_tracker
        self.run_context = run_context
        self.log_file = log_file
        self.writer = writer
        self.tracker = new_tracker()
        self.scene_id = 1
        self.processed = 0
        self.previous_center = None

    def judge(self, burst, center_index: int, center_time: float, extra: Optional[dict] = None) -> dict:
        args = self.args
        analyzer = self.analyzer
        tick_started = time.perf_counter()
        center_frame = burst[center_index][1]
        scene_changed = False
        scene_metrics = {}
        if self.previous_center is not None and not args.no_scene_reset:
            scene_changed, scene_metrics = is_scene_change(
                self.previous_center, center_frame
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
        analysis.tables = aggregate_burst_observations(frame_tables, center_index)
        analysis.inference_ms += side_inference_ms
        analysis.scene_change = scene_changed
        analysis.scene_metrics = scene_metrics
        if scene_changed:
            next_track_id = self.tracker.next_id
            self.tracker = self.new_tracker()
            self.tracker.next_id = next_track_id
            self.scene_id += 1
            for observation in analysis.tables:
                observation.raw_state = OccupancyState.IGNORE
                observation.reason = "scene_transition"
        update = self.tracker.update(
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
        record = frame_log_record(
            self.processed,
            analysis,
            update,
            include_raw_detections=args.log_detections,
        )
        record["scene_id"] = self.scene_id
        for table in record["tables"]:
            table["scene_id"] = self.scene_id
        for event in record["events"]:
            event["scene_id"] = self.scene_id
        record["cadence"] = {
            "interval_seconds": args.sample_seconds,
            "burst_frames": len(burst),
        }
        if extra:
            if "tick" in extra:
                extra["tick"]["duration_s"] = round(time.perf_counter() - tick_started, 3)
            record.update(extra)
        record["run"] = self.run_context
        self.log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.telemetry is not None:
            # 파일에 덧붙이기만 한다.  보내는 건 밤에 따로 돈다 —
            # 7.5초 예산에 전송이 끼면 안 된다 (CLAUDE.md).
            self.telemetry.record(record)
        self.log_file.flush()
        # Rendered once even when both outputs are on: drawing the
        # same overlay twice would inflate the tick budget for no
        # gain.
        rendered = None
        if self.writer is not None or args.frame_dir is not None:
            rendered = render_frame(
                center_frame,
                analysis,
                update,
                debug=args.debug,
            )
        if self.writer is not None:
            self.writer.write(rendered)
        if args.frame_dir is not None:
            save_frame_pair(args.frame_dir, center_time, center_frame, rendered)
        self.processed += 1
        self.previous_center = center_frame
        summary = record["summary"]
        print(
            f"[{center_time:5.1f}s] tables={summary['visible']} "
            f"occupied={summary['occupied']} empty={summary['empty']} "
            f"ignore={summary['ignore']} poses(seated/standing/unknown)="
            f"{summary['seated_poses']}/{summary['standing_poses']}/{summary['unknown_poses']} "
            f"inference={analysis.inference_ms:.0f}ms",
            flush=True,
        )
        return record


def _build_run_context(args: argparse.Namespace, analyzer: SeatNowAnalyzer, input_info: dict) -> dict:
    """Provenance written into every JSONL record (file and live runs)."""
    return {
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
                and args.seat_conf == 0.20
                and args.strong_chair_link == 0.75
                and args.max_table_area == 0.06
                and not args.no_inferred_seats
                and not args.no_scene_reset
                and args.median_frames == 2
            )
            else ("fast" if not args.table_crops and args.imgsz <= 960 else "custom")
        ),
        "input": input_info,
        "models": {
            "detector": str(analyzer.det_model_path),
            "detector_sha256": _sha256(analyzer.det_model_path),
            "detector_backend": analyzer.det_backend,
            "pose": str(analyzer.pose_model_path),
            "pose_sha256": _sha256(analyzer.pose_model_path),
            "pose_backend": analyzer.pose_backend,
        },
        "config": {
            "sample_seconds": args.sample_seconds,
            "median_frames": args.median_frames,
            "start_seconds": args.start_seconds,
            "imgsz": args.imgsz,
            "pose_imgsz": args.pose_imgsz,
            "table_confidence": args.table_conf,
            "object_confidence": args.object_conf,
            "pose_confidence": args.pose_conf,
            "keypoint_confidence": args.keypoint_conf,
            "sitting_angle": args.angle,
            "seat_confidence": args.seat_conf,
            "strong_chair_link": args.strong_chair_link,
            "maximum_table_area_fraction": args.max_table_area,
            "large_table_confidence": analyzer.config.large_table_confidence,
            "hard_table_area_fraction": analyzer.config.hard_table_area_fraction,
            "table_rescue_confidence": analyzer.config.table_rescue_confidence,
            "table_crop_objects": args.table_crops,
            "table_crop_imgsz": args.crop_imgsz,
            "table_crop_confidence": args.crop_conf,
            "maximum_table_crops": args.max_crops,
            "infer_occluded_tables": not args.no_inferred_seats,
            "log_detections": args.log_detections,
            "frame_dir": str(args.frame_dir) if args.frame_dir else None,
            "occupy_confirmations": args.occupy_confirm,
            "empty_confirmations": args.empty_confirm,
            "track_ttl": args.track_ttl,
            "table_layout": {
                "add_confirmation_samples": args.table_layout_add_confirm,
                "move_confirmation_samples": args.table_layout_move_confirm,
                "remove_confirmation_samples": args.table_layout_remove_confirm,
                "bbox_ema_alpha": args.table_layout_bbox_alpha,
            },
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
    hwaccel = resolve_hwaccel(args.hwaccel, args.input)
    print(hwaccel.describe(), flush=True)
    # --median-frames 0 reproduces the original single-frame pipeline exactly
    # (streaming reader, no vote fields).
    legacy_mode = args.median_frames == 0
    run_context = _build_run_context(
        args,
        analyzer,
        input_info={
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "duration": info.duration,
            "codec": info.codec,
        },
    )
    telemetry = _open_telemetry(args, run_context, getattr(args, "cafe_id", "") or "file")
    if legacy_mode:
        print(
            f"Analyzing every {args.sample_seconds:g}s "
            f"(result video {1.0 / args.sample_seconds:.3f} fps)",
            flush=True,
        )
    else:
        print(
            f"Analyzing every {args.sample_seconds:g}s "
            f"(±{args.median_frames} frame majority vote)",
            flush=True,
        )

    new_tracker = _tracker_factory(args)
    tracker = new_tracker()
    # Burst mode writes exactly one rendered frame per judgment, so the result
    # video plays one second per judgment regardless of the sample interval.
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
                    hwaccel_args=hwaccel.args,
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
                    record = frame_log_record(
                        frame_index,
                        analysis,
                        update,
                        include_raw_detections=args.log_detections,
                    )
                    record["scene_id"] = scene_id
                    for table in record["tables"]:
                        table["scene_id"] = scene_id
                    for event in record["events"]:
                        event["scene_id"] = scene_id
                    record["run"] = run_context
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    if telemetry is not None:
                        # 파일에 덧붙이기만 한다.  보내는 건 밤에 따로 돈다 —
                        # 7.5초 예산에 전송이 끼면 안 된다 (CLAUDE.md).
                        telemetry.record(record)
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
                reader = FFmpegBurstReader(args.input, info, hwaccel_args=hwaccel.args)
                runner = _TickRunner(args, analyzer, new_tracker, run_context, log_file, writer)
                runner.telemetry = telemetry
                center_time = args.start_seconds
                while info.duration <= 0 or center_time <= info.duration:
                    center_index, burst = reader.read_burst(
                        center_time, args.median_frames
                    )
                    if not burst:
                        break
                    runner.judge(burst, center_index, center_time)
                    if args.max_samples is not None and runner.processed >= args.max_samples:
                        break
                    center_time += args.sample_seconds
                processed = runner.processed
                scene_id = runner.scene_id
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except Exception as cleanup_error:
                print(
                    f"SeatNow cleanup warning: {cleanup_error}",
                    file=sys.stderr,
                )
        # 도중에 죽어도 그때까지의 요약은 남긴다.  안 그러면 그날 통계가 통째로 사라진다.
        if telemetry is not None:
            telemetry.close()
        raise
    else:
        if writer is not None:
            writer.close()
        if telemetry is not None:
            telemetry.close()

    elapsed = time.perf_counter() - started
    if processed == 0:
        raise RuntimeError("No frames were decoded from the input video")
    print(f"Completed {processed} samples in {elapsed:.1f}s")
    if writer is not None:
        print(f"Annotated video: {output}")
    print(f"JSONL log: {log_path}")
    return 0


def _tracker_factory(args: argparse.Namespace):
    def new_tracker() -> TableTracker:
        return TableTracker(
            occupy_confirmations=args.occupy_confirm,
            empty_confirmations=args.empty_confirm,
            max_missed=args.track_ttl,
            layout_add_confirmations=args.table_layout_add_confirm,
            layout_move_confirmations=args.table_layout_move_confirm,
            layout_remove_confirmations=args.table_layout_remove_confirm,
            layout_bbox_alpha=args.table_layout_bbox_alpha,
        )

    return new_tracker


def _percentile(values, fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _publish_warn(error: BaseException) -> None:
    """Say a transport failed, and survive a console that cannot print it.

    A Windows terminal in cp949 raises on the warning sign, and a crash in
    the warning would be a crash in the judging loop.
    """
    message = f"Supabase 전송 준비 실패: {type(error).__name__}: {error}"
    try:
        print(f"⚠️  {message}", flush=True)
    except Exception:  # noqa: BLE001
        try:
            print(message.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:  # noqa: BLE001
            pass


def _publish_tick(publisher, record: dict, cafe_id: str, version: str) -> None:
    """Hand one judged tick to the publisher.  Nothing here may raise.

    A transport problem must never stop the judging loop: the box keeps
    its own JSONL either way, and the app falls back to "확인 중" through
    the 45s staleness rule (docs/앱연동.md).
    """
    if publisher is None:
        return
    try:
        publisher.publish_live(live_payload(record, cafe_id, version))
        record["publish"] = _publish_stats_for_record(publisher)
    except Exception as error:  # noqa: BLE001 -- 전송 문제는 판정을 멈추지 않는다
        _publish_warn(error)


def _publish_gap(publisher, seat_index: list, cafe_id: str, version: str, wall_clock: str) -> None:
    """A hole in the log goes out as "every seat unknown", never as old values."""
    if publisher is None:
        return
    try:
        publisher.publish_live(gap_payload(seat_index, cafe_id, version, wall_clock))
    except Exception as error:  # noqa: BLE001
        _publish_warn(error)


def process_live(args: argparse.Namespace, analyzer: SeatNowAnalyzer) -> int:
    """Judge a camera stream on a wall-clock schedule until told to stop.

    This is the deployment loop.  It never writes video (문서/plan.md T10), it
    judges the newest frames on every tick even while the stream is
    reconnecting, and every JSONL record carries what the 24/7 box is judged
    on: how late the tick started, how long it took, memory, decoder health.
    """
    url = str(args.input)
    shown_url = redact_url(url)
    print(f"스트림 확인 중: {shown_url}", flush=True)
    info = probe_stream(url)
    if analyzer.layout is not None:
        analyzer.layout = analyzer.layout.scaled_to(info.width, info.height)
    rotator: Optional[LiveLogRotator] = None
    if args.log_dir is not None:
        rotator = LiveLogRotator(args.log_dir, keep_days=args.keep_days)
        removed = rotator.prune()
        if removed:
            print(f"오래된 로그 {len(removed)}개 삭제 (--keep-days {args.keep_days})", flush=True)
        log_path = None
        summary_path = Path(args.log_dir) / "last_run_summary.json"
    else:
        log_path = args.log or live_log_default(url)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path = log_path.with_name(log_path.stem + "_summary.json")
    print(
        f"Live: {shown_url} — {info.width}x{info.height}, {info.fps:.3f} fps, codec={info.codec}",
        flush=True,
    )
    hwaccel = resolve_hwaccel(args.hwaccel, url, prober=probe_live_hwaccel)
    print(hwaccel.describe(), flush=True)

    publisher, publish_state = publisher_from_env(os.environ)
    version = box_version()
    cafe_id = publisher.cafe_id if publisher is not None else ""
    seat_index = seat_index_from_layout(analyzer.layout) if analyzer.layout is not None else []
    print(f"Supabase 전송: {publish_state}", flush=True)
    if publisher is not None:
        publisher.start()
    if hwaccel.fallback:
        print(
            "⚠️  하드웨어 디코딩이 잡히지 않았습니다 — 소프트웨어로 돕니다. "
            "24시간 운영에서 15초 주기를 못 맞출 수 있습니다 (docs/edge-setup.md 4단계 '막혔을 때').",
            flush=True,
        )
    run_context = _build_run_context(
        args,
        analyzer,
        input_info={
            "url": shown_url,
            "live": True,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "codec": info.codec,
        },
    )
    run_context["decode"] = {
        "hwaccel": hwaccel.name,
        "requested": hwaccel.requested,
        "fallback": hwaccel.fallback,
        "tried": list(hwaccel.tried),
    }
    burst_frames = 2 * args.median_frames + 1
    burst_period = burst_period_frames(info.fps, args.live_burst_seconds, burst_frames)
    run_context["decode"]["burst_every_frames"] = burst_period
    run_context["decode"]["burst_frames"] = burst_frames
    telemetry = _open_telemetry(args, run_context, cafe_id)
    print(
        f"Analyzing every {args.sample_seconds:g}s "
        f"(newest {burst_frames} frames majority vote, live"
        + (
            f"; ffmpeg converts {burst_frames} frames every {burst_period} ({args.live_burst_seconds:g}s)"
            if burst_period
            else "; ffmpeg converts every frame"
        )
        + ")",
        flush=True,
    )
    new_tracker = _tracker_factory(args)
    reader = FFmpegLiveReader(
        url,
        info.width,
        info.height,
        hwaccel_args=hwaccel.args,
        buffer_frames=burst_frames + 3,
        burst_every_frames=burst_period,
        burst_frames=burst_frames,
    )
    max_span_s = 1.0 if burst_period else None
    # A camera that went quiet must not keep being judged from its last
    # picture: frames older than one interval count as "no frames".
    max_age_s = resolve_max_frame_age(args.max_frame_age_seconds, args.sample_seconds)
    run_context["decode"]["max_frame_age_s"] = max_age_s
    frames_at_last_tick = 0
    fallback_during_run = False
    schedule = TickSchedule(args.sample_seconds)
    started = time.perf_counter()
    deadline = started + args.run_seconds if args.run_seconds else None
    no_frame_ticks = 0
    tick_durations: list = []
    inference_ms: list = []
    late_seconds: list = []
    rss_samples: list = []
    ffmpeg_rss_samples: list = []
    warned_hwaccel = False
    interrupted = False
    runner: Optional[_TickRunner] = None
    reader.start()
    try:
        if rotator is not None:
            log_file = rotator.open()
            log_path = rotator.path
        else:
            log_file = log_path.open("w", encoding="utf-8")
        try:
            runner = _TickRunner(args, analyzer, new_tracker, run_context, log_file, writer=None)
            runner.telemetry = telemetry
            while True:
                if rotator is not None:
                    rotated = rotator.maybe_rotate()
                    if rotated is not log_file:
                        log_file = rotated
                        runner.log_file = rotated
                        log_path = rotator.path
                        print(f"새 로그 파일: {log_path}", flush=True)
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                wait = schedule.wait_seconds()
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - time.perf_counter()))
                if wait > 0:
                    time.sleep(wait)
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                late = schedule.late_seconds()
                center_index, burst = reader.read_burst(
                    args.median_frames,
                    timeout=min(10.0, args.sample_seconds),
                    max_span_s=max_span_s,
                    max_age_s=max_age_s,
                )
                stats = reader.stats()
                if should_disable_hwaccel(stats, frames_at_last_tick):
                    reader.disable_hwaccel()
                    fallback_during_run = True
                    run_context["decode"]["fallback_during_run"] = True
                    print(
                        "⚠️  하드웨어 디코더가 죽어서 스트림이 끊겼습니다 — 소프트웨어 디코딩으로 갈아탑니다. "
                        "판정은 계속되지만 느려집니다 (docs/edge-setup.md 부록 B): "
                        + " | ".join(stats["hwaccel_messages"][:2]),
                        flush=True,
                    )
                if stats["hwaccel_suspect"] and not warned_hwaccel:
                    warned_hwaccel = True
                    print(
                        "⚠️  ffmpeg 가 하드웨어 디코딩에 실패해 소프트웨어로 떨어졌습니다: "
                        + " | ".join(stats["hwaccel_messages"][:2]),
                        flush=True,
                    )
                if not burst:
                    no_frame_ticks += 1
                    print(
                        f"[{time.perf_counter() - started:6.1f}s] 프레임 없음 — "
                        f"재연결 {stats['reconnects']}회, 마지막 오류: {stats['last_error']}",
                        flush=True,
                    )
                    # A visible hole in the log: consumers must show these
                    # seats as unknown, never as "still what it was".
                    gap = {
                        "gap": True,
                        "reason": "no_fresh_frames",
                        "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "elapsed_s": round(time.perf_counter() - started, 1),
                        "scheduled_s": round(schedule.scheduled - schedule.t0, 3),
                        "live": {
                            "reconnects": stats["reconnects"],
                            "connected": stats["connected"],
                            "frames_decoded": stats["frames_decoded"],
                            "newest_age_s": stats["newest_age_s"],
                            "last_error": stats["last_error"],
                            "hwaccel_suspect": stats["hwaccel_suspect"],
                            "hwaccel_disabled": stats["hwaccel_disabled"],
                        },
                    }
                    log_file.write(json.dumps(gap, ensure_ascii=False) + "\n")
                    log_file.flush()
                    _publish_gap(publisher, seat_index, cafe_id, version, gap["wall_clock"])
                    frames_at_last_tick = stats["frames_decoded"]
                    schedule.advance()
                    continue
                frames_at_last_tick = stats["frames_decoded"]
                center_time = burst[center_index][0]
                rss = process_rss_mb()
                child = reader.child_pid()
                ffmpeg_rss = process_rss_mb(child) if child else None
                extra = {
                    "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "tick": {
                        "scheduled_s": round(schedule.scheduled - schedule.t0, 3),
                        "late_s": round(late, 3),
                        "skipped_total": schedule.skipped,
                        "no_frame_ticks": no_frame_ticks,
                        "burst_age_s": round(stats["newest_age_s"] or 0.0, 3),
                    },
                    "live": {
                        "hwaccel": hwaccel.name,
                        "hwaccel_suspect": stats["hwaccel_suspect"],
                        "hwaccel_disabled": stats["hwaccel_disabled"],
                        "frames_decoded": stats["frames_decoded"],
                        "decode_fps": stats["decode_fps"],
                        "reconnects": stats["reconnects"],
                        "connected": stats["connected"],
                        "last_error": stats["last_error"],
                    },
                    "process": {
                        "rss_mb": round(rss, 1) if rss is not None else None,
                        "ffmpeg_rss_mb": round(ffmpeg_rss, 1) if ffmpeg_rss is not None else None,
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    },
                }
                if publisher is not None:
                    try:
                        extra["publish"] = _publish_stats_for_record(publisher)
                    except Exception:  # noqa: BLE001
                        pass
                record = runner.judge(burst, center_index, center_time, extra=extra)
                _publish_tick(publisher, record, cafe_id, version)
                tick_durations.append(record["tick"]["duration_s"])
                inference_ms.append(record.get("inference_ms", 0.0))
                late_seconds.append(late)
                if rss is not None:
                    rss_samples.append(rss)
                if ffmpeg_rss is not None:
                    ffmpeg_rss_samples.append(ffmpeg_rss)
                schedule.advance()
                if args.max_samples is not None and runner.processed >= args.max_samples:
                    break
        finally:
            if rotator is not None:
                rotator.close()
            else:
                log_file.close()
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted — 요약을 쓰고 끝냅니다", file=sys.stderr, flush=True)
    finally:
        reader.close()
        if publisher is not None:
            publisher.stop()
        if telemetry is not None:
            telemetry.close()

    stats = reader.stats()
    elapsed = time.perf_counter() - started
    processed = runner.processed if runner is not None else 0
    summary = {
        "url": shown_url,
        "elapsed_s": round(elapsed, 1),
        "ticks": processed,
        "no_frame_ticks": no_frame_ticks,
        "skipped_slots": schedule.skipped,
        "interval_seconds": args.sample_seconds,
        "tick_duration_s": {
            "mean": round(sum(tick_durations) / len(tick_durations), 3) if tick_durations else None,
            "p95": _percentile(tick_durations, 0.95),
            "max": max(tick_durations) if tick_durations else None,
            "first": tick_durations[0] if tick_durations else None,
        },
        "inference_ms": {
            "mean": round(sum(inference_ms) / len(inference_ms), 1) if inference_ms else None,
            "max": round(max(inference_ms), 1) if inference_ms else None,
        },
        "late_s": {
            "mean": round(sum(late_seconds) / len(late_seconds), 3) if late_seconds else None,
            "max": round(max(late_seconds), 3) if late_seconds else None,
        },
        "rss_mb": {
            "first": round(rss_samples[0], 1) if rss_samples else None,
            "last": round(rss_samples[-1], 1) if rss_samples else None,
            "max": round(max(rss_samples), 1) if rss_samples else None,
        },
        "ffmpeg_rss_mb": {
            "first": round(ffmpeg_rss_samples[0], 1) if ffmpeg_rss_samples else None,
            "last": round(ffmpeg_rss_samples[-1], 1) if ffmpeg_rss_samples else None,
            "max": round(max(ffmpeg_rss_samples), 1) if ffmpeg_rss_samples else None,
        },
        "decode": {
            "hwaccel": hwaccel.name,
            "fallback_at_start": hwaccel.fallback,
            "fallback_during_run": fallback_during_run,
            "suspect_during_run": stats["hwaccel_suspect"],
            "messages": stats["hwaccel_messages"],
            "frames_decoded": stats["frames_decoded"],
            "reconnects": stats["reconnects"],
            "last_error": stats["last_error"],
        },
        "publish": (
            {"enabled": True, **publisher.stats()} if publisher is not None else {"enabled": False}
        ),
        "interrupted": interrupted,
        "log": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    td = summary["tick_duration_s"]
    print(
        f"\nCompleted {processed} live ticks in {elapsed:.1f}s "
        f"(프레임 없음 {no_frame_ticks}, 건너뛴 슬롯 {schedule.skipped}, 재연결 {stats['reconnects']})"
    )
    if td["mean"] is not None:
        print(
            f"틱 시간: 평균 {td['mean']:.2f}s · p95 {td['p95']:.2f}s · 최대 {td['max']:.2f}s "
            f"(첫 틱 {td['first']:.2f}s) · 합격선 {args.sample_seconds / 2:.1f}s"
        )
    if summary["rss_mb"]["first"] is not None:
        print(
            f"메모리(이 프로세스): 처음 {summary['rss_mb']['first']:.0f}MB → "
            f"마지막 {summary['rss_mb']['last']:.0f}MB (최대 {summary['rss_mb']['max']:.0f}MB)"
            + (
                f" · ffmpeg {summary['ffmpeg_rss_mb']['last']:.0f}MB"
                if summary["ffmpeg_rss_mb"]["last"] is not None
                else ""
            )
        )
    print(
        f"디코딩: {hwaccel.name}"
        + (" ⚠️ 실행 중 하드웨어가 죽어 소프트웨어로 갈아탐" if fallback_during_run else "")
        + (" ⚠️ 실행 중 하드웨어 실패 흔적 있음" if stats["hwaccel_suspect"] and not fallback_during_run else "")
    )
    if publisher is not None:
        ps = summary["publish"]
        print(
            f"Supabase 전송: 성공 {ps['sent']}회 · 실패 {ps['failed']}회"
            + (f" · 마지막 오류: {ps['last_error']}" if ps["last_error"] else "")
        )
    print(f"JSONL log: {log_path}")
    print(f"Summary: {summary_path}")
    if processed == 0 and not interrupted:
        raise RuntimeError("실시간 스트림에서 판정한 틱이 하나도 없습니다 (프레임을 못 받았습니다)")
    return 130 if interrupted else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        if is_live_source(args.input):
            layout = load_layout(args.layout) if args.layout else None
            _require_complete_layout(layout)
            analyzer = _make_analyzer(args, layout=layout)
            return process_live(args, analyzer)
        suffix = args.input.suffix.lower()
        if suffix not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported input extension: {suffix or '(none)'}")
        layout = load_layout(args.layout) if args.layout else None
        _require_complete_layout(layout)
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
