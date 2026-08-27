"""Generate — and validate — a SeatNow labelling fixture for one video.

plan.md T3: the 1-hour recording and the 10-minute edge cases cannot be scored
because they have no labels.  Labelling is manual work that only a person can
do, so this script does everything around it: it probes the video, extracts a
contact sheet to label from, and writes a skeleton fixture with every interval
pre-created and every occupancy field left blank.

    # 1. contact sheet + skeleton (one interval per 30 s)
    python make_labels.py sample_raw/cafe_1h.mp4 --interval 30 \
        --contact-sheet labels/cafe_1h --layout layouts/cafe.json

    # 2. open labels/cafe_1h/*.jpg, fill in the "expected" blocks by hand

    # 3. check the fixture is complete before spending a run on it
    python make_labels.py sample_raw/cafe_1h.mp4 --validate \
        tests/fixtures/cafe_1h_expectations.json

Every interval needs each seat listed in exactly one of ``occupied``,
``empty``, or ``ignore``.  ``ignore`` is not a cop-out: it is the label for a
seat the camera genuinely cannot judge, and its share is the coverage number
that decides the one-camera question in plan.md §2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
TODO = "TODO"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seat_names_from_layout(layout) -> List[str]:
    """Label targets are judgement units: a bar zone contributes one per seat.

    Labelling a whole bar as a single row would make its per-seat judgements
    unscorable, so the skeleton lists BAR-1, BAR-2, … just like the analyzer.
    """
    return [unit.name for unit in layout.judgement_units()]


def seat_ids(layout_path: Optional[Path], count: int) -> List[str]:
    """Seat labels for the timeline: from a calibrated layout, or A, B, C…"""
    if layout_path is not None:
        from seatnow_layout import load_layout

        return seat_names_from_layout(load_layout(layout_path))
    return [chr(ord("A") + index) for index in range(count)]


def extract_contact_sheet(
    source: Path, destination: Path, interval: float, duration: float
) -> List[Path]:
    """Write one JPEG per labelling interval, named by its start second."""
    from seatnow_core import FFmpegBurstReader, probe_video

    import cv2

    destination.mkdir(parents=True, exist_ok=True)
    reader = FFmpegBurstReader(source, probe_video(source))
    written: List[Path] = []
    timestamp = 0.0
    while timestamp < duration:
        _, burst = reader.read_burst(timestamp, 0)
        if burst:
            path = destination / f"{int(timestamp):06d}s.jpg"
            cv2.imwrite(str(path), burst[0][1])
            written.append(path)
        timestamp += interval
    return written


def build_skeleton(
    source: Path,
    interval: float,
    seats: Sequence[str],
    notes: Sequence[str] = (),
) -> Dict[str, object]:
    from seatnow_core import probe_video

    info = probe_video(source)
    duration = info.duration
    timeline = []
    start = 0.0
    while start < duration:
        end = min(start + interval, duration)
        timeline.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "expected": {"occupied": TODO, "empty": TODO, "ignore": TODO},
                "notes": "",
            }
        )
        start = end
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": f"seatnow_{source.stem}",
        "active_ground_truth_profile": "project_literal_or_v1",
        "annotation": {
            "source": "manual_visual_observation",
            "annotated_on": TODO,
            "model_inference_used_as_ground_truth": False,
            "sampling_interval_seconds_for_contact_sheet": interval,
            "time_boundary_uncertainty_seconds": interval / 2.0,
            "notes": list(notes),
        },
        "video": {
            "filename": source.name,
            "repository_relative_path": str(source),
            "sha256": sha256_file(source),
            "width_pixels": info.width,
            "height_pixels": info.height,
            "duration_seconds": duration,
            "frame_rate": {"frames_per_second": info.fps},
            "video_codec": info.codec,
        },
        # A long recording scored under the adaptive cadence has no fixed
        # sample grid, so the strict timeline check is off by default here.
        "scoring": {
            "require_full_timeline": False,
            "require_transition_count": False,
            "require_no_scene_change": True,
        },
        "semantic_tables": {
            seat: {
                "name": TODO,
                "description": TODO,
                "manual_confidence": TODO,
            }
            for seat in seats
        },
        "timeline": timeline,
        "expected_events": {
            "expected_customer_occupancy_transition_count": TODO,
        },
    }


def validate(fixture: Dict[str, object]) -> List[str]:
    """Return every reason this fixture is not ready to score against."""
    problems: List[str] = []
    seats = set(fixture.get("semantic_tables") or {})
    if not seats:
        problems.append("semantic_tables is empty")
    video = fixture.get("video") or {}
    if not video.get("sha256"):
        problems.append("video.sha256 is missing")

    timeline = fixture.get("timeline") or []
    if not timeline:
        problems.append("timeline is empty")
    previous_end: Optional[float] = None
    for index, interval in enumerate(timeline):
        where = f"timeline[{index}] ({interval.get('start_seconds')}s)"
        expected = interval.get("expected") or {}
        listed: List[str] = []
        for key in ("occupied", "empty", "ignore"):
            value = expected.get(key)
            if value == TODO or value is None:
                problems.append(f"{where}: '{key}' is still {TODO}")
                continue
            if not isinstance(value, list):
                problems.append(f"{where}: '{key}' must be a list")
                continue
            listed.extend(str(seat) for seat in value)
        if not listed:
            continue
        duplicates = {seat for seat in listed if listed.count(seat) > 1}
        if duplicates:
            problems.append(f"{where}: seat(s) listed twice: {sorted(duplicates)}")
        unknown = sorted(set(listed) - seats)
        if unknown:
            problems.append(f"{where}: unknown seat(s): {unknown}")
        missing = sorted(seats - set(listed))
        if missing:
            problems.append(
                f"{where}: seat(s) unlabelled: {missing} "
                "(every seat needs occupied/empty/ignore)"
            )
        start = float(interval.get("start_seconds", 0.0))
        if previous_end is not None and abs(start - previous_end) > 1e-6:
            problems.append(f"{where}: gap or overlap after {previous_end}s")
        previous_end = float(interval.get("end_seconds", start))

    events = fixture.get("expected_events") or {}
    if events.get("expected_customer_occupancy_transition_count") == TODO:
        problems.append("expected_events.expected_customer_occupancy_transition_count is still TODO")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Video to label")
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds per labelling interval (also the contact-sheet spacing)",
    )
    parser.add_argument(
        "--seats",
        type=int,
        default=6,
        help="How many seats to pre-create when no --layout is given",
    )
    parser.add_argument(
        "--layout", type=Path, help="calibrate.py layout JSON; seat names come from it"
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="Directory to write one labelling frame per interval into",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the skeleton fixture "
        "(default tests/fixtures/<stem>_expectations.json)",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        help="Check an existing fixture instead of generating one",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing fixture"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.validate is not None:
        fixture = json.loads(args.validate.read_text(encoding="utf-8"))
        problems = validate(fixture)
        if problems:
            print(f"{args.validate}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        intervals = len(fixture.get("timeline") or [])
        seats = len(fixture.get("semantic_tables") or {})
        print(f"{args.validate}: OK — {intervals} intervals x {seats} seats")
        return 0

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    output = args.output or (
        PROJECT_DIR / "tests" / "fixtures" / f"{args.video.stem}_expectations.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    seats = seat_ids(args.layout, args.seats)
    skeleton = build_skeleton(args.video, args.interval, seats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    intervals = len(skeleton["timeline"])
    print(f"Skeleton fixture: {output}  ({intervals} intervals x {len(seats)} seats)")

    if args.contact_sheet is not None:
        frames = extract_contact_sheet(
            args.video,
            args.contact_sheet,
            args.interval,
            float(skeleton["video"]["duration_seconds"]),
        )
        print(f"Contact sheet: {args.contact_sheet}  ({len(frames)} frames)")

    print(
        "\n다음 단계: 대조표 이미지를 보면서 각 interval의 "
        "occupied / empty / ignore 를 채운 뒤\n"
        f"  python make_labels.py {args.video} --validate {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
