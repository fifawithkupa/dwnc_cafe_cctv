"""사람이 고치기 쉬운 "구간 표"로 15초 정답지를 만든다.

정답지(15초 칸 × 좌석)를 사람이 직접 채우면 칸이 수천 개다.  대신 좌석마다
"몇 초부터 몇 초까지 사용중" 구간만 적고, 칸은 기계가 채운다.

    venv/Scripts/python.exe -m checks.segments_to_labels results/moonq_golden/2026-11-15/segments.json

구간 표(segments.json) 모양:
    {"video": "sample_raw/....mp4", "layout": "layouts/moonq.json", "interval": 15,
     "edge_margin": 8,
     "occupied": {"T1": [[365, 605]], ...},     # 사람 눈으로 사람·짐이 있던 구간
     "ignore":   {"BAR6-3": [[1450, 1560]]},    # 사람 눈으로도 애매한 구간
     "events":   [[2115, 2257, "조명 꺼짐"]]}

칸 규칙: 구간에 칸 전체가 들어가면 occupied, 전혀 안 겹치면 empty, 경계에
걸치면 ignore (사람이 움직이는 중이라 정답이 없다).  구간 끝 ± edge_margin 초도
경계로 본다 — 영상을 15초 간격으로 보고 정한 시각이라 그만큼 흔들린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from checks.make_labels import build_skeleton, seat_ids

PROJECT_DIR = Path(__file__).resolve().parents[1]


def label_cell(start: float, end: float, occ: Sequence[Sequence[float]],
               ign: Sequence[Sequence[float]], margin: float) -> str:
    for a, b in ign:
        if start < b and end > a:
            return "ignore"
    for a, b in occ:
        if a + margin <= start and end <= b - margin:
            return "occupied"
    for a, b in occ:
        if start < b + margin and end > a - margin:
            return "ignore"
    return "empty"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("segments", type=Path)
    p.add_argument("--out", type=Path, help="기본: 구간 표 옆 expectations.json")
    args = p.parse_args(argv)

    spec = json.loads(args.segments.read_text(encoding="utf-8"))
    video = PROJECT_DIR / spec["video"]
    seats = seat_ids(PROJECT_DIR / spec["layout"], 0)
    unknown = (set(spec.get("occupied", {})) | set(spec.get("ignore", {}))) - set(seats)
    if unknown:
        raise SystemExit(f"좌석 이름이 배치도에 없다: {sorted(unknown)}")

    fixture = build_skeleton(video, float(spec["interval"]), seats,
                             notes=spec.get("notes", []))
    fixture["video"]["repository_relative_path"] = spec["video"]
    fixture["annotation"]["annotated_on"] = spec.get("annotated_on", "")
    fixture["annotation"]["segments_file"] = str(args.segments).replace("\\", "/")
    margin = float(spec.get("edge_margin", 8))
    events = spec.get("events", [])
    counts: Dict[str, int] = {"occupied": 0, "empty": 0, "ignore": 0}
    for cell in fixture["timeline"]:
        s, e = cell["start_seconds"], cell["end_seconds"]
        groups: Dict[str, List[str]] = {"occupied": [], "empty": [], "ignore": []}
        for seat in seats:
            state = label_cell(s, e, spec.get("occupied", {}).get(seat, []),
                               spec.get("ignore", {}).get(seat, []), margin)
            groups[state].append(seat)
            counts[state] += 1
        cell["expected"] = groups
        cell["notes"] = "; ".join(n for a, b, n in events if s < b and e > a)
    for seat in seats:
        fixture["semantic_tables"][seat] = {
            "name": seat, "description": "", "manual_confidence": "medium"}
    fixture["expected_events"]["expected_customer_occupancy_transition_count"] = sum(
        len(v) for v in spec.get("occupied", {}).values()) * 2

    out = args.out or args.segments.with_name("expectations.json")
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(counts.values())
    print(f"칸 {len(fixture['timeline'])}개 × 좌석 {len(seats)} = {total}")
    for k, v in counts.items():
        print(f"  {k:8s} {v:5d}  ({v / total:.0%})")
    print(f"정답지: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
