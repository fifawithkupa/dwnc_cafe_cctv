"""Put the three layers of one judgement on one line, side by side.

"Does YOLO work here" cannot be answered, because a person whose lower body
is under a table is not a bug.  "Where does it break first" can be, and that
is what this table is for: raw detection, then pose, then the seat verdict,
in that order, against how many people were actually in the picture.

    python inspect_run.py sample_results/angle1.jsonl --judge frames/angle1/judge

Read it downward, not across.  If the detection column is already wrong the
other two columns are consequences, not findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from frame_dump import frame_stem
from judge_frames import Judgement
from seatnow_report import REASON_GROUPS
from verify_seatnow import load_jsonl


PROJECT_DIR = Path(__file__).resolve().parent
LAYERS = ("detector", "pose")

# COCO names the detector emits for the two furniture classes SeatNow uses.
CHAIR_CLASSES = ("chair", "couch", "bench")
TABLE_CLASSES = ("dining table",)


@dataclass
class Row:
    """One judged tick: what was seen, what was inferred, what was decided."""

    stem: str
    timestamp: float
    det_person: int
    det_chair: int
    det_table: int
    pose_seated: int
    pose_standing: int
    pose_unknown: int
    seat_occupied: int
    seat_empty: int
    seat_unknown: int
    seat_ignore: int
    truth: Optional[int] = None
    truth_tables_visible: Optional[int] = None
    truth_tables_in_use: Optional[int] = None
    truth_tables_belongings_only: Optional[int] = None
    excluded_people: Optional[str] = None
    excluded_tables: Optional[str] = None

    @property
    def pose_total(self) -> int:
        return self.pose_seated + self.pose_standing + self.pose_unknown

    @property
    def detector_gap(self) -> Optional[int]:
        return None if self.truth is None else self.det_person - self.truth

    @property
    def pose_gap(self) -> Optional[int]:
        return None if self.truth is None else self.pose_total - self.truth

    @property
    def seat_gap(self) -> Optional[int]:
        """Seats we call taken, minus seats a person judged actually in use.

        Positive means the app tells a customer "no room" when there was
        room -- the direction that turns people away from the door.
        """
        if self.truth_tables_in_use is None:
            return None
        return self.seat_occupied - self.truth_tables_in_use

    def found(self, layer: str) -> int:
        if layer == "detector":
            return self.det_person
        if layer == "pose":
            return self.pose_total
        raise ValueError(f"Unknown layer: {layer}")


def _counts(record: Dict[str, object]) -> Dict[str, int]:
    raw = record.get("raw_detections") or {}
    return dict(raw.get("counts") or {})  # type: ignore[union-attr]


def build_rows(
    records: List[Dict[str, object]], judgements: Dict[str, Judgement]
) -> List[Row]:
    """Join our own log with the blind counts, matched by frame stem."""
    rows: List[Row] = []
    for record in records:
        timestamp = float(record["timestamp"])  # type: ignore[arg-type]
        stem = frame_stem(timestamp)
        counts = _counts(record)
        summary = record.get("summary") or {}
        judgement = judgements.get(stem)

        truth: Optional[int] = None
        tables_visible: Optional[int] = None
        tables_in_use: Optional[int] = None
        tables_bags_only: Optional[int] = None
        excluded_people: Optional[str] = None
        excluded_tables: Optional[str] = None
        if judgement is not None:
            # People and belongings are excluded independently.  One
            # ambiguous grey object -- customer bag or cafe supply? -- must
            # not throw away a people count that was never in doubt.
            if judgement.error is not None:
                excluded_people = excluded_tables = "error"
            else:
                if judgement.uncertain_people:
                    excluded_people = "uncertain"
                else:
                    truth = judgement.people_total
                if judgement.uncertain_tables:
                    excluded_tables = "uncertain"
                else:
                    tables_visible = judgement.tables_visible
                    tables_in_use = judgement.tables_in_use
                    tables_bags_only = judgement.tables_belongings_only

        rows.append(
            Row(
                stem=stem,
                timestamp=timestamp,
                det_person=int(counts.get("person", 0)),
                det_chair=sum(int(counts.get(name, 0)) for name in CHAIR_CLASSES),
                det_table=sum(int(counts.get(name, 0)) for name in TABLE_CLASSES),
                pose_seated=int(summary.get("seated_poses", 0)),  # type: ignore[union-attr]
                pose_standing=int(summary.get("standing_poses", 0)),  # type: ignore[union-attr]
                pose_unknown=int(summary.get("unknown_poses", 0)),  # type: ignore[union-attr]
                seat_occupied=int(summary.get("occupied", 0)),  # type: ignore[union-attr]
                seat_empty=int(summary.get("empty", 0)),  # type: ignore[union-attr]
                seat_unknown=int(summary.get("unknown", 0)),  # type: ignore[union-attr]
                seat_ignore=int(summary.get("ignore", 0)),  # type: ignore[union-attr]
                truth=truth,
                truth_tables_visible=tables_visible,
                truth_tables_in_use=tables_in_use,
                truth_tables_belongings_only=tables_bags_only,
                excluded_people=excluded_people,
                excluded_tables=excluded_tables,
            )
        )
    return rows


@dataclass
class Recall:
    """How much of what was there got found, at one layer."""

    layer: str
    found: int
    truth: int
    value: Optional[float]
    scored_frames: int
    excluded_frames: int


def recall(rows: List[Row], layer: str) -> Recall:
    """Pooled recall over the frames that have usable ground truth.

    Per-frame found counts are capped at that frame's truth: recall answers
    "how many of the people present were found", so extra boxes in one frame
    must not pay for people missed in another.
    """
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer: {layer}. Expected one of {LAYERS}")
    found = 0
    total = 0
    scored = 0
    excluded = 0
    for row in rows:
        if row.excluded_people is not None:
            excluded += 1
            continue
        if row.truth is None:
            continue
        scored += 1
        total += row.truth
        found += min(row.found(layer), row.truth)
    value = (found / total) if total else None
    return Recall(
        layer=layer,
        found=found,
        truth=total,
        value=value,
        scored_frames=scored,
        excluded_frames=excluded,
    )


def load_judgements(judge_dir: Optional[Path]) -> Dict[str, Judgement]:
    """Read every written judgement.  A missing directory means none yet."""
    if judge_dir is None or not Path(judge_dir).is_dir():
        return {}
    judgements: Dict[str, Judgement] = {}
    for path in sorted(Path(judge_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("stem", path.stem)
        judgements[payload["stem"]] = Judgement(**payload)
    return judgements


def reason_distribution(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count reason codes by the group that fixes them.

    The grouping is the point: "geometry 22% / model 8%" says whether the
    next move is a rescue path or fine-tuning, which one flat list does not.

    Two shapes have to be read, because ``build_seat_report`` emits two: a
    plain table carries one ``reason_code``, while a bar zone carries a
    ``reason_codes`` tally over its seats (``seatnow_report.py:150-182``).
    Reading only the first shape drops every bar seat without a trace.
    """
    lookup = {
        code.value: group
        for group, codes in REASON_GROUPS.items()
        for code in codes
    }
    distribution: Dict[str, int] = {group: 0 for group in REASON_GROUPS}
    distribution["other"] = 0

    def add(code: str, count: int = 1) -> None:
        distribution[lookup.get(code, "other")] += count

    for record in records:
        report = record.get("seat_report") or {}
        for seat in report.get("seats") or []:  # type: ignore[union-attr]
            if seat.get("kind") == "counted_zone":
                for code, count in (seat.get("reason_codes") or {}).items():
                    add(str(code), int(count))
                continue
            code = seat.get("reason_code")
            if code is not None:
                add(str(code))
    return distribution


def disagreements(rows: List[Row]) -> List[Row]:
    """Rows where we and the blind count differ -- the diagnosis shortlist."""
    return [
        row
        for row in rows
        if row.excluded_people is None and row.truth is not None and row.detector_gap != 0
    ]


def _truth_cell(row: Row) -> str:
    if row.excluded_people is not None:
        return row.excluded_people
    if row.truth is None:
        return "___"
    return str(row.truth)


def _gap_cell(row: Row) -> str:
    if row.excluded_people is not None or row.detector_gap is None:
        return ""
    if row.detector_gap == 0:
        return "0"
    return f"{row.detector_gap:+d} !!"


def _table_truth_cell(row: Row) -> str:
    if row.excluded_tables is not None:
        return row.excluded_tables
    if row.truth_tables_in_use is None:
        return "___ / ___ / ___ (짐만)"
    return (
        f"{row.truth_tables_visible}/{row.truth_tables_in_use}/"
        f"{row.truth_tables_belongings_only} (짐만)"
    )


def _seat_gap_cell(row: Row) -> str:
    if row.excluded_tables is not None or row.seat_gap is None:
        return ""
    if row.seat_gap == 0:
        return "0"
    return f"{row.seat_gap:+d} !!"


def render_table(rows: List[Row]) -> str:
    """The reading table: three layers across, one tick per line."""
    lines = [
        "| 사진 | 검출(사람/의자/책상) | 포즈(앉음/섬/모름) | 좌석(점유/빈/모름/무시) "
        "| 실제 사람 | 차이 | 실제 테이블(보임/사용중/짐만) | 점유 차이 |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.stem} "
            f"| {row.det_person}/{row.det_chair}/{row.det_table} "
            f"| {row.pose_seated}/{row.pose_standing}/{row.pose_unknown} "
            f"| {row.seat_occupied}/{row.seat_empty}/{row.seat_unknown}/{row.seat_ignore} "
            f"| {_truth_cell(row)} "
            f"| {_gap_cell(row)} "
            f"| {_table_truth_cell(row)} "
            f"| {_seat_gap_cell(row)} |"
        )
    return "\n".join(lines)


def _recall_line(label: str, result: Recall) -> str:
    if result.value is None:
        # The excluded count belongs here too: "no score" because nobody
        # counted and "no score" because every count was thrown out are
        # different findings, and only the second one is about the harness.
        return (
            f"- **{label} 재현율**: 정답 없음 "
            f"(채점한 사진 {result.scored_frames}장, 제외 {result.excluded_frames}장)"
        )
    return (
        f"- **{label} 재현율**: {result.value:.2f} "
        f"({result.found}/{result.truth}명, 사진 {result.scored_frames}장, "
        f"제외 {result.excluded_frames}장)"
    )


def render_summary(rows: List[Row], records: List[Dict[str, object]]) -> str:
    """The two numbers and the backlog table, under one heading."""
    detector = recall(rows, "detector")
    pose = recall(rows, "pose")
    distribution = reason_distribution(records)
    total_reasons = sum(distribution.values()) or 1

    lines = [
        "## 요약",
        "",
        f"- 판정한 tick: {len(rows)}회",
        _recall_line("검출", detector),
        _recall_line("포즈", pose),
        _over_detection_line("검출", over_detection(rows, "detector")),
        _over_detection_line("포즈", over_detection(rows, "pose")),
        _seat_inflation_line(seat_inflation(rows)),
        "",
        "> 재현율만 보면 안 된다. 재현율은 \"있는 사람을 몇 % 찾았나\"라서",
        "> 없는 사람을 만들어내는 것을 아예 못 본다 — 2명짜리 장면에서 7명을",
        "> 잡아도 재현율은 1.00이다. 두 줄을 같이 읽어야 한다.",
        "",
        "### 사유 코드 분포 (무엇이 고치는가로 묶음)",
        "",
        "| 그룹 | 건수 | 비율 | 고치는 방법 |",
        "|---|---:|---:|---|",
    ]
    how = {
        "install": "카메라를 다시 단다 (코드로 풀지 않는다)",
        "geometry": "구제 경로·좌석 칸 다시 긋기",
        "model": "파인튜닝 또는 imgsz 상향 (plan.md T11)",
        "time": "기다리면 된다 — 할 일 없음",
        "settled": "판정이 끝난 것 — 문제 아님",
        "other": "어휘에 없는 코드 — 확인 필요",
    }
    for group, count in distribution.items():
        lines.append(
            f"| {group} | {count} | {count / total_reasons:.0%} | {how.get(group, '')} |"
        )

    shortlist = disagreements(rows)
    lines += [
        "",
        f"### 진단 대상 {len(shortlist)}장 (3단계에서 이것만 본다)",
        "",
    ]
    if not shortlist:
        lines.append("없음 — 우리 숫자와 실제가 전부 일치한다.")
    else:
        for row in shortlist:
            lines.append(
                f"- `marked/{row.stem}.jpg` — 우리: 사람 {row.det_person}명, "
                f"점유 {row.seat_occupied}석, 모름 {row.seat_unknown}석 "
                f"/ 실제: {row.truth}명 ({row.detector_gap:+d})"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="seatnow.py 가 쓴 JSONL")
    parser.add_argument("--judge", type=Path, help="judge_frames.py 결과 폴더")
    parser.add_argument("--output", type=Path, help="마크다운 저장 경로 (기본: 화면 출력)")
    parser.add_argument("--title", default=None, help="보고서 제목")
    args = parser.parse_args(argv)

    records = load_jsonl(args.log)
    judgements = load_judgements(args.judge)
    rows = build_rows(records, judgements)

    title = args.title or f"# 검출 판독표 — {args.log.name}"
    report = "\n\n".join(
        [title, render_summary(rows, records), "## 판독표", render_table(rows)]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"판독표: {args.output}")
    else:
        print(report)
    return 0



@dataclass
class OverDetection:
    """The failure recall cannot see: boxes that no person stands behind.

    Recall caps found at truth, so seven boxes over two people still scores
    1.00.  In this café the misses are rare and the inventions are not, which
    makes this the number that actually describes the system.
    """

    layer: str
    scored_frames: int
    frames_over: int
    frames_under: int
    frames_exact: int
    extra_total: int
    worst_gap: int


def over_detection(rows: List[Row], layer: str) -> OverDetection:
    """Count invented people, in frames and in heads."""
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer: {layer}. Expected one of {LAYERS}")
    scored = over = under = exact = extra = 0
    worst = 0
    for row in rows:
        if row.excluded_people is not None or row.truth is None:
            continue
        scored += 1
        gap = row.found(layer) - row.truth
        if gap > 0:
            over += 1
            extra += gap
            worst = max(worst, gap)
        elif gap < 0:
            under += 1
        else:
            exact += 1
    return OverDetection(
        layer=layer,
        scored_frames=scored,
        frames_over=over,
        frames_under=under,
        frames_exact=exact,
        extra_total=extra,
        worst_gap=worst,
    )


def _over_detection_line(label: str, result: OverDetection) -> str:
    if result.scored_frames == 0:
        return f"- **{label} 과탐**: 정답 없음"
    share = result.frames_over / result.scored_frames
    return (
        f"- **{label} 과탐**: {result.frames_over}/{result.scored_frames}장 "
        f"({share:.0%}), 없는 사람 총 {result.extra_total}명, "
        f"최악 한 장 +{result.worst_gap}명 "
        f"| 놓침 {result.frames_under}장 | 정확 {result.frames_exact}장"
    )

def seat_inflation(rows: List[Row]) -> OverDetection:
    """How far our "occupied" count runs ahead of the tables really in use.

    This is the number that maps onto what a customer sees.  Over-reporting
    puts "no room" in the app while seats sit empty; under-reporting sends
    someone to a cafe that is full.  Both are here, counted separately,
    because they are not equally bad and must not average out.
    """
    scored = over = under = exact = extra = 0
    worst = 0
    for row in rows:
        if row.excluded_tables is not None or row.seat_gap is None:
            continue
        scored += 1
        gap = row.seat_gap
        if gap > 0:
            over += 1
            extra += gap
            worst = max(worst, gap)
        elif gap < 0:
            under += 1
        else:
            exact += 1
    return OverDetection(
        layer="seat",
        scored_frames=scored,
        frames_over=over,
        frames_under=under,
        frames_exact=exact,
        extra_total=extra,
        worst_gap=worst,
    )


def _seat_inflation_line(result: OverDetection) -> str:
    if result.scored_frames == 0:
        return "- **자리 없음 부풀림**: 정답 없음"
    share = result.frames_over / result.scored_frames
    return (
        f"- **자리 없음 부풀림**: {result.frames_over}/{result.scored_frames}장 "
        f"({share:.0%}), 없는 점유 총 {result.extra_total}석, "
        f"최악 한 장 +{result.worst_gap}석 "
        f"| 적게 셈 {result.frames_under}장 | 정확 {result.frames_exact}장"
    )


if __name__ == "__main__":
    sys.exit(main())
