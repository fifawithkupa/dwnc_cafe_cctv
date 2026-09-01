"""Verify SeatNow JSONL runs against manually annotated fixtures.

One 20-second fixture cannot separate a real parameter improvement from noise,
so this scores a *suite*: any number of (log, expectations) pairs, matched by
the input video's sha256, with the per-video and pooled accuracy reported
side by side.

    python verify_seatnow.py run.jsonl
    python verify_seatnow.py runs/*.jsonl --expectations tests/fixtures
    python verify_seatnow.py runs/*.jsonl --json-report report.json

Beyond accuracy it reports *coverage*: how much of the labelled time SeatNow
spends on tables it refuses to score (``raw_state=ignore``) and how often a
table goes missing.  That is the number that decides the one-camera question
in plan.md §2 — a seat the camera cannot see is not a model problem.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from engine.seatnow_report import ACTIONABLE_GROUPS, REASON_GROUPS


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTATIONS = PROJECT_DIR / "tests" / "fixtures" / "test_video_expectations.json"
FIXTURE_GLOB = "*_expectations.json"

# A fixture may waive checks that only make sense for the original short clip.
# A one-hour recording scored under the adaptive cadence, for example, has no
# fixed sample grid to compare timestamps against.
DEFAULT_SCORING = {
    "require_full_timeline": True,
    "require_transition_count": True,
    "require_no_scene_change": True,
}


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


def load_expectations(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def discover_expectations(paths: Sequence[Path]) -> List[Dict[str, object]]:
    """Load every fixture named, expanding directories."""
    fixtures: List[Dict[str, object]] = []
    for path in paths:
        if path.is_dir():
            members = sorted(path.glob(FIXTURE_GLOB))
            if not members:
                raise ValueError(f"No {FIXTURE_GLOB} fixtures in {path}")
            fixtures.extend(load_expectations(member) for member in members)
        else:
            fixtures.append(load_expectations(path))
    return fixtures


def scoring_policy(expectations: Dict[str, object]) -> Dict[str, bool]:
    policy = dict(DEFAULT_SCORING)
    policy.update(expectations.get("scoring") or {})
    return policy


def fixture_label(expectations: Dict[str, object]) -> str:
    return str(
        expectations.get("fixture_id")
        or (expectations.get("video") or {}).get("filename")
        or "unnamed_fixture"
    )


def match_fixture(
    records: Sequence[Dict[str, object]], fixtures: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Pick the fixture whose video sha256 matches this log.

    Matching on content, not filename, is what makes ``*.jsonl`` globs safe:
    scoring a run against the wrong video would silently report nonsense.
    """
    run = records[0].get("run") or {}
    digest = (run.get("input") or {}).get("sha256")
    if len(fixtures) == 1 and digest is None:
        return fixtures[0]
    for expectations in fixtures:
        if (expectations.get("video") or {}).get("sha256") == digest:
            return expectations
    known = ", ".join(fixture_label(fixture) for fixture in fixtures)
    raise ValueError(
        f"No fixture matches input sha256 {digest!r}. Loaded fixtures: {known}"
    )


def coverage_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Quantify what the camera and the tracker never got to score.

    ``ignore_ratio`` is the share of table observations SeatNow declined to
    judge (border-cropped seats, mostly).  ``missing_count`` is how many
    consecutive samples a tracked table went unseen before this record.
    """
    total = 0
    by_state: Dict[str, int] = {}
    missing_histogram: Dict[str, int] = {}
    per_table: Dict[str, Dict[str, int]] = {}
    for record in records:
        for table in record.get("tables") or []:
            total += 1
            state = str(table.get("raw_state"))
            by_state[state] = by_state.get(state, 0) + 1
            missed = int(table.get("missing_count") or 0)
            key = str(missed)
            missing_histogram[key] = missing_histogram.get(key, 0) + 1
            label = str(table.get("layout_name") or table.get("label") or "?")
            bucket = per_table.setdefault(label, {"observations": 0, "ignored": 0})
            bucket["observations"] += 1
            bucket["ignored"] += int(state == "ignore")
    ignored = by_state.get("ignore", 0)
    return {
        "table_observations": total,
        "observations_by_raw_state": dict(sorted(by_state.items())),
        "ignore_ratio": round(ignored / total, 4) if total else 0.0,
        "missing_count_histogram": dict(
            sorted(missing_histogram.items(), key=lambda item: int(item[0]))
        ),
        "any_missed_ratio": (
            round(
                sum(count for key, count in missing_histogram.items() if int(key) > 0)
                / total,
                4,
            )
            if total
            else 0.0
        ),
        "per_table": {
            label: dict(
                bucket,
                ignore_ratio=round(bucket["ignored"] / bucket["observations"], 4),
            )
            for label, bucket in sorted(per_table.items())
        },
    }


def summarize_unknown_reasons(
    records: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Break the UNKNOWN rate down by what would actually fix it.

    "UNKNOWN 34% = geometry 22% + model 8% + time 4%" turns the next
    engineering decision into a table lookup instead of an argument.  The
    ``time`` group (waiting for confirmation) is deliberately excluded from
    ``actionable_unknown_ticks``: it is normal operation, not a defect.
    """
    group_of = {
        code.value: group
        for group, codes in REASON_GROUPS.items()
        for code in codes
    }
    by_group: Dict[str, int] = {group: 0 for group in REASON_GROUPS}
    by_code: Dict[str, int] = {}
    total = 0
    unknown = 0

    def record_code(code: str, count: int) -> None:
        by_code[code] = by_code.get(code, 0) + count
        group = group_of.get(code)
        if group:
            by_group[group] = by_group.get(group, 0) + count

    for record in records:
        report = record.get("seat_report") or {}
        for seat in report.get("seats") or []:
            if seat.get("kind") == "counted_zone":
                total += int(seat.get("capacity") or 0)
                unknown += int(seat.get("unknown") or 0)
                for code, count in (seat.get("reason_codes") or {}).items():
                    record_code(str(code), int(count))
                continue
            total += 1
            if seat.get("state") != "unknown":
                continue
            unknown += 1
            record_code(str(seat.get("reason_code") or ""), 1)

    actionable = sum(by_group.get(group, 0) for group in ACTIONABLE_GROUPS)
    return {
        "total_seat_ticks": total,
        "unknown_seat_ticks": unknown,
        "unknown_rate": round(unknown / total, 4) if total else 0.0,
        "by_group": by_group,
        "by_code": dict(sorted(by_code.items())),
        "actionable_unknown_ticks": actionable,
    }


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
    policy = scoring_policy(expectations)
    required = ["metadata", "all_sampled_counts_exact"]
    if policy["require_full_timeline"]:
        required.append("full_timeline_coverage")
    if policy["require_transition_count"]:
        required.append("occupancy_transition_count")
    if policy["require_no_scene_change"]:
        required.append("no_unexpected_scene_change")
    return {
        "fixture_id": fixture_label(expectations),
        "passed": all(checks[name] for name in required),
        "checks": checks,
        "required_checks": required,
        "coverage": coverage_metrics(records),
        "unknown_reasons": summarize_unknown_reasons(records),
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


def verify_suite(
    logs: Sequence[Path], fixtures: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Score every log against its matching fixture and pool the results."""
    reports = []
    for log in logs:
        records = load_jsonl(log)
        expectations = match_fixture(records, fixtures)
        report = verify_records(records, expectations)
        report["log"] = str(log)
        reports.append(report)

    frames_total = sum(report["frames_total"] for report in reports)
    frames_exact = sum(report["frames_exact"] for report in reports)
    observations = sum(
        report["coverage"]["table_observations"] for report in reports
    )
    ignored = sum(
        report["coverage"]["observations_by_raw_state"].get("ignore", 0)
        for report in reports
    )
    return {
        "passed": all(report["passed"] for report in reports),
        "videos": len(reports),
        "frames_total": frames_total,
        "frames_exact": frames_exact,
        "frame_accuracy": (frames_exact / frames_total) if frames_total else 0.0,
        "table_observations": observations,
        "ignore_ratio": round(ignored / observations, 4) if observations else 0.0,
        "reports": reports,
    }


def print_coverage(coverage: Dict[str, object], indent: str = "  ") -> None:
    print(f"{indent}Coverage: {coverage['table_observations']} table observations")
    print(
        f"{indent}  ignore ratio      {coverage['ignore_ratio'] * 100:5.1f}%"
        "   (화각이 못 보는 좌석)"
    )
    print(f"{indent}  any-missed ratio  {coverage['any_missed_ratio'] * 100:5.1f}%")
    by_state = ", ".join(
        f"{state}={count}" for state, count in coverage["observations_by_raw_state"].items()
    )
    if by_state:
        print(f"{indent}  raw states        {by_state}")
    worst = sorted(
        coverage["per_table"].items(),
        key=lambda item: item[1]["ignore_ratio"],
        reverse=True,
    )[:3]
    for label, bucket in worst:
        if bucket["ignore_ratio"] > 0:
            print(
                f"{indent}  {label:<10s} ignored "
                f"{bucket['ignored']}/{bucket['observations']}"
            )


GROUP_NOTES = {
    "install": "카메라 재설치",
    "geometry": "가림·구제 로직",
    "model": "모델(파인튜닝·해상도)",
    "time": "확정 대기 — 고칠 것 없음",
}


def print_unknown_reasons(breakdown: Dict[str, object], indent: str = "  ") -> None:
    total = breakdown["total_seat_ticks"]
    if not total:
        return
    print(
        f"{indent}UNKNOWN: {breakdown['unknown_seat_ticks']}/{total} "
        f"({breakdown['unknown_rate'] * 100:.1f}%)  "
        f"개선 대상 {breakdown['actionable_unknown_ticks']}건"
    )
    for group, count in breakdown["by_group"].items():
        if not count or group == "settled":
            continue
        share = count / total * 100
        print(f"{indent}  {group:<9s} {count:4d} ({share:4.1f}%)  {GROUP_NOTES[group]}")
    for code, count in breakdown["by_code"].items():
        print(f"{indent}    - {code:<24s} {count}")


def print_suite_report(suite: Dict[str, object]) -> None:
    for report in suite["reports"]:
        print_report(report)
        print()
    if suite["videos"] > 1:
        status = "PASS" if suite["passed"] else "FAIL"
        print(f"=== Suite: {status} over {suite['videos']} videos ===")
        print(
            f"Pooled exact sampled counts: {suite['frames_exact']}/"
            f"{suite['frames_total']} ({suite['frame_accuracy'] * 100:.1f}%)"
        )
        print(f"Pooled ignore ratio: {suite['ignore_ratio'] * 100:.1f}%")


def print_report(report: Dict[str, object]) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(f"SeatNow fixture verification [{report.get('fixture_id')}]: {status}")
    print(
        f"Exact sampled counts: {report['frames_exact']}/{report['frames_total']} "
        f"({report['frame_accuracy'] * 100:.1f}%)"
    )
    print(f"Profile: {report.get('profile')}")
    required = set(report.get("required_checks") or report["checks"])
    for name, passed in report["checks"].items():
        mark = "PASS" if passed else "FAIL"
        suffix = "" if name in required else "  (not required by this fixture)"
        print(f"  {mark}  {name}{suffix}")
    if report.get("coverage"):
        print_coverage(report["coverage"])
    if report.get("unknown_reasons"):
        print_unknown_reasons(report["unknown_reasons"])
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log", type=Path, nargs="+", help="SeatNow JSONL result(s)")
    parser.add_argument(
        "--expectations",
        type=Path,
        nargs="+",
        default=[DEFAULT_EXPECTATIONS],
        help=f"Fixture file(s), or a directory scanned for {FIXTURE_GLOB}",
    )
    parser.add_argument("--json-report", type=Path, help="Optional machine-readable report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fixtures = discover_expectations(args.expectations)
        suite = verify_suite(args.log, fixtures)
        print_suite_report(suite)
        if args.json_report:
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(
                json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0 if suite["passed"] else 1
    except Exception as exc:
        print(f"Verification error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
