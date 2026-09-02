"""사람이 손으로 쓴 정답지를 읽어 채점할 수 있는 형태로 바꾼다.

정답지는 개발자가 아니라 **사장님이 사진을 보면서 쓴다.**  그래서 형식을
엄격하게 잡지 않는다 — 대소문자, 공백, 한글/영어 어휘를 다 받아준다.
대신 **읽을 수 없는 칸은 조용히 넘기지 않고 오류로 세운다.**  마음대로
읽으면 채점표가 사장님이 쓴 것과 다른 것을 채점하게 된다.

## 표 형식 (실제로 쓰는 것)

    | 시간(s) | T1    | T2 | T3    | ... | 비고     |
    | ----- | ----- | -- | ----- | --- | ------ |
    | 0     | X (t) | O  | O     | ... |        |
    | 30    | X (t) | O  | X (s) | ... | 테이블 옮김 |

* `X` = 사용중, `O` = 비어있음, `?` = 모름
* 괄호 = **그 자리에 지금 무엇이 있는가** — `t` 테이블에 물건 / `s` 사람이
  앉음 / `c` 의자에 짐.  `O (c)` 는 짐은 있지만 쓸 수 있는 자리다 (바 규칙)

괄호까지 정답으로 받는 것이 이 파서의 값이다.  상태만 맞으면 "맞음"으로
넘어가는 칸에도 **근거를 놓친 것**이 숨어 있고, 그게 곧 다음 숙제다.

## 줄 형식 (짧게 고쳐 쓸 때)

    1번 사진 (0초)
    - T1 : 사용중
    - BAR7-4 : 모름 ← 사람이 완전히 가림
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


# 상태 어휘.  세 상태를 그대로 내보내는 것이 이 제품의 원칙이라
# (CLAUDE.md), 애매한 것을 '빈자리'로 반올림하는 어휘는 넣지 않는다.
STATE_WORDS = {
    "x": "occupied",
    "사용중": "occupied",
    "사용": "occupied",
    "점유": "occupied",
    "occupied": "occupied",
    "o": "empty",
    "0": "empty",
    "빈자리": "empty",
    "빈": "empty",
    "빔": "empty",
    "비어있음": "empty",
    "비었음": "empty",
    "empty": "empty",
    "?": "unknown",
    "모름": "unknown",
    "알수없음": "unknown",
    "unknown": "unknown",
    # IGNORE = 화각이 아예 못 보는 자리.  설치 결함이지 판정 오답이 아니라서
    # (CLAUDE.md) 채점 분모에서 뺀다.
    "무시": "ignore",
    "안보임": "ignore",
    "ignore": "ignore",
}

# 근거 글자 — engine.seatnow_core 의 EVIDENCE_* 와 같은 어휘를 쓴다.
EVIDENCE_LETTERS = frozenset({"s", "t", "c"})

# "1번 사진 (0초)" / "## 3번 사진" / "4번 사진 - 45초"
_HEADING = re.compile(r"^\s*#*\s*(\d+)\s*번\s*사진(.*)$")
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)\s*초")
# "- T1 : 사용중 ← 이유" / "* bar7-4: 모름" / "T2 : 빔  # 이유"
_SEAT_LINE = re.compile(r"^\s*[-*+]?\s*([A-Za-z][A-Za-z0-9\s\-_]*?)\s*[:：]\s*(.+?)\s*$")
# 이유를 여는 글자들.  화살표든 주석 기호든 다 받는다.
_NOTE_SPLIT = re.compile(r"\s*(?:←|<-|<=|—|--|#|//)\s*")
# 표의 구분줄: | --- | :--- | ---: |
_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
# 한 칸: "X (t,c)" / "O ( c)" / "?"
_CELL = re.compile(r"^\s*([^\s(]+)\s*(?:\(([^)]*)\))?\s*$")
# 표 머리의 시간 칸과 비고 칸
_TIME_HEADER = re.compile(r"시간|초|second|time", re.IGNORECASE)
_NOTE_HEADER = re.compile(r"비고|메모|note|remark", re.IGNORECASE)


def normalize_seat(name: str) -> str:
    """`bar7 - 4` 도 `BAR7-4` 로 읽는다.  사람이 쓴 이름은 고르지 않다."""
    return re.sub(r"\s+", "", str(name)).upper()


def normalize_state(word: str) -> Optional[str]:
    """상태 낱말 하나를 표준 상태로.  못 읽으면 None (호출부가 오류를 낸다)."""
    cleaned = re.sub(r"\s+", "", str(word)).lower()
    return STATE_WORDS.get(cleaned)


def parse_evidence(text: Optional[str]) -> Set[str]:
    """괄호 안의 근거 글자들.  `t,c` `t c` `t、c` 를 모두 같은 것으로 읽는다."""
    if not text:
        return set()
    letters = {piece.strip().lower() for piece in re.split(r"[,\s/·、+]+", text)}
    return {letter for letter in letters if letter in EVIDENCE_LETTERS}


@dataclass
class Answer:
    """정답지 한 칸 — 이 시점의 이 자리는 무엇이어야 하는가."""

    timestamp: Optional[float]
    photo_index: int
    seat: str
    state: str
    note: str = ""
    # 그 자리에 지금 무엇이 있는가 (s/t/c).  상태와 따로 채점한다.
    evidence: Set[str] = field(default_factory=set)

    @property
    def evidence_code(self) -> str:
        """`occupancy_evidence_code` 와 같은 순서로 — 사람 > 책상 > 의자."""
        return "".join(letter for letter in "stc" if letter in self.evidence)


@dataclass
class AnswerKey:
    answers: List[Answer] = field(default_factory=list)

    def timestamps(self) -> List[float]:
        seen: List[float] = []
        for answer in self.answers:
            if answer.timestamp is not None and answer.timestamp not in seen:
                seen.append(answer.timestamp)
        return sorted(seen)

    def lookup(self, timestamp: float, seat: str) -> Optional[Answer]:
        wanted = normalize_seat(seat)
        for answer in self.answers:
            if answer.seat == wanted and _same_time(answer.timestamp, timestamp):
                return answer
        return None

    def resolve_photo_numbers(self, timestamps: Sequence[float]) -> None:
        """초를 안 적은 정답지를 실행의 시점 목록에 맞춰 채운다.

        "3번 사진"만 적혀 있으면 그것이 몇 초인지는 정답지 혼자서는 모른다.
        실행 로그의 시점을 순서대로 대응시킨다 — 사진 수가 안 맞으면
        ``validate`` 가 빠진 시점으로 잡아낸다.
        """
        ordered = sorted(timestamps)
        for answer in self.answers:
            if answer.timestamp is not None:
                continue
            index = answer.photo_index - 1
            if 0 <= index < len(ordered):
                answer.timestamp = ordered[index]


def _same_time(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def _split_note(body: str) -> tuple:
    parts = _NOTE_SPLIT.split(body, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), ""


def _table_cells(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def parse_table(text: str) -> AnswerKey:
    """머리줄에 좌석 이름이 있는 표를 읽는다.  없으면 빈 정답지를 준다."""
    key = AnswerKey()
    lines = text.splitlines()
    header: Optional[List[str]] = None
    time_column = 0
    note_columns: Set[int] = set()
    photo_index = 0

    for number, line in enumerate(lines, start=1):
        cells = _table_cells(line)
        if not cells:
            continue
        if _TABLE_RULE.match(line):
            continue
        if header is None:
            # 좌석 이름이 두 개 이상 보이는 줄만 머리줄로 받는다.
            seats = [
                index
                for index, cell in enumerate(cells)
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9\s\-_]*", cell)
                and not _TIME_HEADER.search(cell)
                and not _NOTE_HEADER.search(cell)
            ]
            if len(seats) < 2:
                continue
            header = cells
            time_column = next(
                (index for index, cell in enumerate(cells) if _TIME_HEADER.search(cell)),
                0,
            )
            note_columns = {
                index for index, cell in enumerate(cells) if _NOTE_HEADER.search(cell)
            }
            continue

        raw_time = cells[time_column] if time_column < len(cells) else ""
        seconds = re.search(r"(\d+(?:\.\d+)?)", raw_time)
        if not seconds:
            continue
        timestamp = float(seconds.group(1))
        photo_index += 1
        row_note = " ".join(
            cells[index] for index in sorted(note_columns) if index < len(cells)
        ).strip()

        for index, cell in enumerate(cells):
            if index == time_column or index in note_columns or index >= len(header):
                continue
            seat = normalize_seat(header[index])
            if not re.fullmatch(r"[A-Z][A-Z0-9\-_]*", seat):
                continue
            if not cell:
                raise ValueError(
                    f"{number}줄: {timestamp:g}초의 '{seat}' 칸이 비어 있다 — "
                    "빈 칸을 '맞음'으로 가정하면 회귀를 못 잡는다"
                )
            match = _CELL.match(cell)
            state = normalize_state(match.group(1)) if match else None
            if state is None:
                raise ValueError(
                    f"{number}줄: {timestamp:g}초의 '{seat}' 를 못 읽었다 — '{cell}'.\n"
                    "  쓸 수 있는 것: X(사용중) / O(비어있음) / ?(모름)"
                )
            key.answers.append(
                Answer(
                    timestamp=timestamp,
                    photo_index=photo_index,
                    seat=seat,
                    state=state,
                    note=row_note,
                    evidence=parse_evidence(match.group(2)),
                )
            )
    return key


def parse_lines(text: str) -> AnswerKey:
    """`1번 사진 (0초)` + `- T1 : 사용중` 형식을 읽는다.

    산문(제목, 총평)은 그냥 넘어간다.  하지만 자리 줄의 모양은 갖췄는데
    **상태를 못 읽으면 멈춘다.**  넘겨버리면 그 칸이 채점에서 조용히
    사라지고, 사장님은 자기가 답한 줄 안다.
    """
    key = AnswerKey()
    photo_index: Optional[int] = None
    timestamp: Optional[float] = None

    for number, line in enumerate(text.splitlines(), start=1):
        heading = _HEADING.match(line)
        if heading:
            photo_index = int(heading.group(1))
            seconds = _SECONDS.search(heading.group(2))
            timestamp = float(seconds.group(1)) if seconds else None
            continue

        match = _SEAT_LINE.match(line)
        if not match:
            continue
        seat = normalize_seat(match.group(1))
        # 자리 이름 모양이 아닌 줄("총평 : 좋다")은 산문이다.
        if not re.fullmatch(r"[A-Z][A-Z0-9\-_]*", seat):
            continue

        body, note = _split_note(match.group(2))
        cell = _CELL.match(body)
        state = normalize_state(cell.group(1)) if cell else normalize_state(body)
        if state is None:
            raise ValueError(
                f"{number}줄: '{seat}' 의 상태를 못 읽었다 — '{body}'.\n"
                f"  쓸 수 있는 낱말: {', '.join(sorted(set(STATE_WORDS)))}"
            )
        if photo_index is None:
            raise ValueError(
                f"{number}줄: '{seat}' 가 어느 사진인지 모른다 — "
                "위에 '1번 사진 (0초)' 같은 줄이 있어야 한다"
            )
        key.answers.append(
            Answer(
                timestamp=timestamp,
                photo_index=photo_index,
                seat=seat,
                state=state,
                note=note,
                evidence=parse_evidence(cell.group(2)) if cell else set(),
            )
        )
    return key


def parse_answer_key(text: str) -> AnswerKey:
    """표든 줄이든 알아서 읽는다.  표가 있으면 표가 이긴다."""
    table = parse_table(text)
    if table.answers:
        return table
    return parse_lines(text)


def validate(
    key: AnswerKey, seats: Sequence[str], timestamps: Sequence[float]
) -> List[str]:
    """채점하기 전에, 이 정답지가 통째로 쓸 수 있는지 본다.

    빠진 칸을 '맞음'으로 가정하면 **회귀를 못 잡는다** — 지금 맞게 보고
    있는 자리가 수정 중에 깨져도 드러나지 않는다.  그래서 모든 시점 ×
    모든 자리가 차 있어야 채점을 시작한다.
    """
    problems: List[str] = []
    wanted_seats = [normalize_seat(seat) for seat in seats]
    known = set(wanted_seats)

    by_time: Dict[float, List[Answer]] = {}
    for answer in key.answers:
        if answer.timestamp is None:
            problems.append(
                f"{answer.photo_index}번 사진의 {answer.seat}: 몇 초인지 모른다 "
                "('N번 사진 (30초)' 처럼 적거나, 실행 시점 수와 사진 수를 맞춰라)"
            )
            continue
        by_time.setdefault(answer.timestamp, []).append(answer)

    for stamp in sorted(timestamps):
        entries = by_time.get(stamp)
        if not entries:
            problems.append(f"{stamp:g}초 사진의 정답이 통째로 없다")
            continue
        listed = [entry.seat for entry in entries]
        for seat in sorted(set(listed)):
            if listed.count(seat) > 1:
                problems.append(f"{stamp:g}초: {seat} 가 두 번 적혀 있다")
        unknown = sorted(set(listed) - known)
        if unknown:
            problems.append(
                f"{stamp:g}초: 없는 자리 이름 {unknown} "
                f"(이 레이아웃의 자리: {wanted_seats})"
            )
        missing = [seat for seat in wanted_seats if seat not in set(listed)]
        if missing:
            problems.append(f"{stamp:g}초: 정답이 빠진 자리 {missing}")

    wanted_times = {float(stamp) for stamp in timestamps}
    for stamp in sorted(set(by_time) - wanted_times):
        problems.append(f"{stamp:g}초: 실행에 없는 시점의 정답이 적혀 있다")

    return problems
