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
    excluded: Optional[str] = None

    @property
    def pose_total(self) -> int:
        return self.pose_seated + self.pose_standing + self.pose_unknown

    @property
    def detector_gap(self) -> Optional[int]:
        return None if self.truth is None else self.det_person - self.truth

    @property
    def pose_gap(self) -> Optional[int]:
        return None if self.truth is None else self.pose_total - self.truth

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
        excluded: Optional[str] = None
        if judgement is not None:
            if judgement.error is not None:
                excluded = "error"
            elif judgement.uncertain:
                excluded = "uncertain"
            else:
                truth = judgement.people_total

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
                excluded=excluded,
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
        if row.excluded is not None:
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
