"""Verify a SeatNow JSONL run against a manually annotated fixture."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPECTATIONS = PROJECT_DIR / "tests" / "fixtures" / "test_video_expectations.json"


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"No frame records found in {path}")
    return records


def expected_interval(expectations: Dict[str, object], timestamp: float) -> Dict[str, object]:
    timeline = expectations["timeline"]
    for index, interval in enumerate(timeline):
        start = float(interval["start_seconds"])
        end = float(interval["end_seconds"])
        is_last = index == len(timeline) - 1
        if start <= timestamp < end or (is_last and abs(timestamp - end) <= 1e-6):
            return interval
    raise ValueError(f"Timestamp {timestamp} is outside the expectation timeline")


def verify_records(
    records: Sequence[Dict[str, object]], expectations: Dict[str, object]
) -> Dict[str, object]:
    expected_video = expectations["video"]
    run = records[0].get("run") or {}
    actual_input = run.get("input") or {}
    metadata_checks = {
        "sha256": actual_input.get("sha256") == expected_video.get("sha256"),
        "width": actual_input.get("width") == expected_video.get("width_pixels"),
        "height": actual_input.get("height") == expected_video.get("height_pixels"),
        "duration": abs(
            float(actual_input.get("duration") or 0.0)
            - float(expected_video.get("duration_seconds") or 0.0)
        )
        <= 0.01,
    }

    frames = []
    exact_frames = 0
    for record in records:
        timestamp = float(record["timestamp"])
        interval = expected_interval(expectations, timestamp)
        expected = interval["expected"]
        summary = record["summary"]
        expected_occupied = len(expected["occupied"])
        expected_empty = len(expected["empty"])
        actual_occupied = int(summary["occupied"])
        actual_empty = int(summary["empty"])
        exact = (
            actual_occupied == expected_occupied
            and actual_empty == expected_empty
        )
        exact_frames += int(exact)
        frames.append(
            {
                "timestamp": timestamp,
                "expected_occupied": expected_occupied,
                "actual_occupied": actual_occupied,
                "expected_empty": expected_empty,
                "actual_empty": actual_empty,
                "expected_ids": expected,
                "exact_counts": exact,
            }
        )

    state_changes = []
    scene_changes = []
    for record in records:
        for event in record.get("events") or []:
            if event.get("type") == "state_change":
                state_changes.append(event)
            elif event.get("type") == "scene_change":
                scene_changes.append(event)

    expected_transition_count = int(
        expectations["expected_events"]["expected_customer_occupancy_transition_count"]
    )
    run_config = run.get("config") or {}
    sample_seconds = float(run_config.get("sample_seconds") or 0.0)
    start_seconds = float(run_config.get("start_seconds") or 0.0)
    duration = float(expected_video["duration_seconds"])
    expected_frame_count = (
        int(math.ceil(max(0.0, duration - start_seconds) / sample_seconds))
        if sample_seconds > 0
        else 0
    )
    timestamps_match = sample_seconds > 0 and all(
        abs(float(record["timestamp"]) - (start_seconds + index * sample_seconds))
        <= 1e-4
        for index, record in enumerate(records)
    )
    checks = {
        "metadata": all(metadata_checks.values()),
        "full_timeline_coverage": (
            len(records) == expected_frame_count and timestamps_match
        ),
        "all_sampled_counts_exact": exact_frames == len(frames),
        "occupancy_transition_count": len(state_changes) == expected_transition_count,
        "no_unexpected_scene_change": not scene_changes,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metadata_checks": metadata_checks,
        "profile": run.get("profile"),
        "frames_total": len(frames),
        "frames_exact": exact_frames,
        "frame_accuracy": exact_frames / len(frames),
        "frames": frames,
        "state_changes": state_changes,
        "scene_changes": scene_changes,
        "limitations": [
            "Aggregate counts cannot prove semantic A-G identity by themselves.",
            "Staff/reflection exclusion still requires visual review of the annotated MP4.",
        ],
    }


def print_report(report: Dict[str, object]) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(f"SeatNow fixture verification: {status}")
    print(
        f"Exact sampled counts: {report['frames_exact']}/{report['frames_total']} "
        f"({report['frame_accuracy'] * 100:.1f}%)"
    )
    print(f"Profile: {report.get('profile')}")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    mismatches = [frame for frame in report["frames"] if not frame["exact_counts"]]
    if mismatches:
        print("Mismatched timestamps:")
        for frame in mismatches:
            print(
                f"  t={frame['timestamp']:5.1f}s  "
                f"occupied {frame['actual_occupied']}/{frame['expected_occupied']}  "
                f"empty {frame['actual_empty']}/{frame['expected_empty']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="SeatNow JSONL result")
    parser.add_argument(
        "--expectations",
        type=Path,
        default=DEFAULT_EXPECTATIONS,
        help="Manual expectation fixture",
    )
    parser.add_argument("--json-report", type=Path, help="Optional machine-readable report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_jsonl(args.log)
        with args.expectations.open("r", encoding="utf-8") as source:
            expectations = json.load(source)
        report = verify_records(records, expectations)
        print_report(report)
        if args.json_report:
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0 if report["passed"] else 1
    except Exception as exc:
        print(f"Verification error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
