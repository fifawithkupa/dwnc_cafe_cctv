"""앱 코드에 넣은 이름표 목록이 박스 좌석 파일과 같은지 대조한다 (문서/앱팀할일.md 3단계).

    python -m install.seat_check --layout layouts/<카페>.json --app-ids <앱이 보낸 목록.txt>

앱이 보낸 목록은 한 줄에 이름표 하나. 빈 줄과 `#` 로 시작하는 줄은 무시한다.
통과하면 0, 아니면 1 로 끝난다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from edge.publish import seat_index_from_layout
from engine.seatnow_layout import load_layout


def parse_app_ids(lines: Iterable[str]) -> List[str]:
    ids: List[str] = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        ids.append(text)
    return ids


def compare(box_ids: Sequence[str], app_ids: Sequence[str]) -> List[str]:
    """사람이 읽을 문제 목록. 비어 있으면 통과."""
    problems: List[str] = []
    seen = set()
    for seat_id in app_ids:
        if seat_id in seen:
            problems.append(f"앱 목록에 `{seat_id}` 가 두 번 있음")
        seen.add(seat_id)
    box_set, app_set = set(box_ids), set(app_ids)
    for seat_id in box_ids:
        if seat_id not in app_set:
            problems.append(f"`{seat_id}` 이(가) 앱에 없음")
    for seat_id in app_ids:
        if seat_id not in box_set:
            problems.append(f"`{seat_id}` 이(가) 박스에 없음")
    for seat_id in app_ids:
        if seat_id not in box_set:
            for candidate in box_ids:
                if candidate.lower() == seat_id.lower() and candidate != seat_id:
                    problems.append(f"`{seat_id}` 는 대소문자가 다름 — 박스는 `{candidate}`")
    if len(box_ids) != len(app_ids):
        problems.append(f"개수 다름 — 박스 {len(box_ids)}개, 앱 {len(app_ids)}개")
    return problems


def report(box_ids: Sequence[str], app_ids: Sequence[str]) -> str:
    problems = compare(box_ids, app_ids)
    if not problems:
        return f"통과 — {len(box_ids)}개, 이름 전부 일치"
    return "실패\n" + "\n".join(f"- {line}" for line in problems)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, required=True, help="박스 좌석 파일 (layouts/<카페>.json)")
    parser.add_argument("--app-ids", type=Path, required=True, help="앱이 보낸 이름표 목록 (한 줄 하나)")
    args = parser.parse_args(argv)
    layout = load_layout(args.layout)
    box_ids = [entry["seat_id"] for entry in seat_index_from_layout(layout)]
    app_ids = parse_app_ids(args.app_ids.read_text(encoding="utf-8").splitlines())
    text = report(box_ids, app_ids)
    print(text)
    return 0 if text.startswith("통과") else 1


if __name__ == "__main__":
    sys.exit(main())
