"""사람이 한 장씩 넘겨보는 검수 폴더를 만든다.

`marked/` 의 주석 사진은 진단용이라 글자가 너무 많다 — `layout=v1 ACTIVE
conf=1.00 table=1.00 | no_customer_evidence` 같은 줄이 서로 겹쳐 사진을 덮는다.
검수는 목적이 다르다.  **자리 이름과 판정, 그리고 왜 그렇게 봤는지**만 남기고
나머지는 전부 지운다.

    python -m checks.make_review results/angle1_layout --title angle1

만들어지는 것:

    review/
      00_읽는법.md
      01_t0000s_점유3빈9모름0_사람3대2_차이+1_★_원본.jpg   ← 아무것도 안 그린 사진
      01_t0000s_점유3빈9모름0_사람3대2_차이+1_★_판정.jpg   ← 우리 판정

파일 이름이 곧 요약이다.  `_원본` 이 `_판정` 보다 먼저 정렬되는 건 일부러다 —
답을 먼저 보면 눈이 그쪽으로 끌려간다.

이 폴더는 **디스크에 이미 있는 실행 결과로부터** 다시 그린다.  영상을 다시
돌리지 않으므로 판정이 바뀌지 않는다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from checks.inspect_run import load_judgements
from checks.judge_frames import Judgement
from checks.verify_seatnow import load_jsonl
from engine.frame_dump import frame_stem
from engine.seatnow_core import evidence_code_from_log


# 판정 색 (OpenCV 는 BGR 이다).  marked/ 와 같은 색을 써서 두 폴더를 나란히
# 봐도 눈이 헷갈리지 않게 한다.
COLORS = {
    "occupied": (50, 55, 235),  # 빨강 — 사용중
    "empty": (60, 190, 70),  # 초록 — 빈자리
    "unknown": (155, 155, 155),  # 회색 — 모름
    "ignore": (80, 185, 230),  # 주황 — 화각 밖
}
PERSON_COLOR = (235, 140, 40)  # 파랑 — 우리가 사람으로 잡은 것
MARK = {"occupied": "X", "empty": "O", "unknown": "?", "ignore": "-"}

# 사용중 근거 글자를 한글로 푼 것.  `00_읽는법.md` 와 표에서 같이 쓴다.
EVIDENCE_KOREAN = {
    "s": "사람이 앉음",
    "t": "책상에 짐",
    "c": "의자에 짐",
}

STATE_KOREAN = {
    "occupied": "사용중",
    "empty": "빔",
    "unknown": "모름",
    "ignore": "화각 밖",
}

# 근거가 없을 때 사유 코드를 한글로 푼 것.  없는 코드는 코드 그대로 보여준다 —
# 조용히 "알 수 없음"으로 뭉개면 새 사유가 생겨도 아무도 모른다.
REASON_KOREAN = {
    "no_customer_evidence": "아무 근거 없음",
    "belongings_only": "짐만 있음 (바 자리라 안 셈)",
    "spans_multiple_seats": "한 사람이 두 칸에 걸침",
    "border_cropped": "화면 끝에 잘림",
}


@dataclass
class Tick:
    """검수 한 장에 필요한 것 전부."""

    index: int
    stem: str
    timestamp: float
    record: Dict[str, object]
    judgement: Optional[Judgement]

    @property
    def summary(self) -> Dict[str, int]:
        return dict(self.record.get("summary") or {})  # type: ignore[arg-type]

    @property
    def pose_total(self) -> int:
        summary = self.summary
        return (
            int(summary.get("seated_poses", 0))
            + int(summary.get("standing_poses", 0))
            + int(summary.get("unknown_poses", 0))
        )

    @property
    def our_people(self) -> int:
        """탐지기가 잡은 사람 수 — 자세 판정 이전의 날것.

        자세 판정을 통과한 수(``pose_total``)를 쓰면 헛것 하나를 잡았다가
        자세 단계에서 조용히 버린 경우가 안 보인다.  검수의 목적은 그
        어긋남을 사람 눈으로 잡는 것이므로 가장 앞 단계 수를 센다.
        """
        counts = (self.record.get("raw_detections") or {}).get("counts") or {}
        if "person" in counts:
            return int(counts["person"])
        return self.pose_total

    @property
    def truth_people(self) -> Optional[int]:
        if self.judgement is None or not self.judgement.usable_people:
            return None
        return self.judgement.people_total

    @property
    def gap(self) -> Optional[int]:
        truth = self.truth_people
        return None if truth is None else self.our_people - truth

    @property
    def filename_stem(self) -> str:
        summary = self.summary
        seats = (
            f"점유{summary.get('occupied', 0)}"
            f"빈{summary.get('empty', 0)}"
            f"모름{summary.get('unknown', 0)}"
        )
        truth = self.truth_people
        gap = self.gap
        if truth is None:
            # 전각 물음표다.  윈도우 파일 이름에 ASCII "?" 를 못 쓴다.
            people = f"사람{self.our_people}대？"
            gap_text = "차이？"
            star = "－"
        else:
            people = f"사람{self.our_people}대{truth}"
            gap_text = f"차이{gap:+d}"
            star = "★" if gap else "－"
        seconds = int(round(self.timestamp))
        return (
            f"{self.index:02d}_t{seconds:04d}s_{seats}_{people}_{gap_text}_{star}"
        )


def build_ticks(
    records: Sequence[Dict[str, object]], judgements: Dict[str, Judgement]
) -> List[Tick]:
    ticks: List[Tick] = []
    for index, record in enumerate(records, start=1):
        timestamp = float(record["timestamp"])  # type: ignore[arg-type]
        stem = frame_stem(timestamp)
        ticks.append(
            Tick(
                index=index,
                stem=stem,
                timestamp=timestamp,
                record=record,
                judgement=judgements.get(stem),
            )
        )
    return ticks


def seat_name(table: Dict[str, object]) -> str:
    """사람이 그린 이름(T1, BAR7-3)을 쓰고, 없으면 추적 번호를 쓴다."""
    name = table.get("layout_name")
    return str(name) if name else str(table.get("label", "?"))


def seat_caption(table: Dict[str, object]) -> str:
    """상자 위에 올릴 짧은 글자: `X T5 (t)` / `O T3` / `? BAR7-4`."""
    state = str(table.get("state", "unknown"))
    caption = f"{MARK.get(state, '?')} {seat_name(table)}"
    # 근거 글자는 근거를 실제로 본 자리에만 붙인다.  사용중이거나, 점유
    # 근거를 처음 봐서 모름으로 잡아둔 자리다.  빈자리 옆의 괄호는 읽는
    # 사람에게 "그럼 왜 빈자리인가"라는 되묻기를 만들 뿐이다.
    if state == "occupied" or (
        state == "unknown" and str(table.get("raw_state")) == "occupied"
    ):
        code = evidence_code_from_log(table)
        if code:
            caption += f" ({code})"
    return caption


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float,
) -> None:
    thickness = max(2, int(round(scale * 2)))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = origin
    y = max(height + baseline + 4, y)
    cv2.rectangle(
        frame,
        (x, y - height - baseline - 6),
        (x + width + 10, y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 5, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def read_image(path: Path) -> Optional[np.ndarray]:
    """cv2.imread 대신.  OpenCV 는 한글이 든 경로를 열지 못한다 (Windows)."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def write_image(path: Path, frame: np.ndarray) -> None:
    """cv2.imwrite 대신.  한글 파일 이름에 imwrite 는 조용히 실패한다 —
    사진이 안 생긴 걸 눈으로 세기 전까지 아무도 모른다."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"사진을 인코딩하지 못했다: {path}")
    path.write_bytes(buffer.tobytes())


def render_verdict(frame: np.ndarray, tick: Tick) -> np.ndarray:
    """자리 상자와 사람 상자만 그린다.  진단용 글자는 전부 뺀다."""
    output = frame.copy()
    height, width = output.shape[:2]
    scale = max(0.7, min(width, height) / 1250.0)
    header = max(60, int(height * 0.055))

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, header), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
    summary = tick.summary
    cv2.putText(
        output,
        f"t={tick.timestamp:05.1f}s   "
        f"X used={summary.get('occupied', 0)}   "
        f"O free={summary.get('empty', 0)}   "
        f"? unknown={summary.get('unknown', 0)}",
        (18, int(header * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (240, 243, 250),
        max(2, int(round(scale * 2))),
        cv2.LINE_AA,
    )

    for pose in tick.record.get("poses") or []:  # type: ignore[union-attr]
        x1, y1, x2, y2 = [int(round(float(value))) for value in pose["box"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        draw_label(output, "person", (x1, max(header + 4, y1)), PERSON_COLOR, scale * 0.7)

    for table in tick.record.get("tables") or []:  # type: ignore[union-attr]
        state = str(table.get("state", "unknown"))
        color = COLORS.get(state, COLORS["unknown"])
        x1, y1, x2, y2 = [int(round(float(value))) for value in table["box"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        draw_label(
            output, seat_caption(table), (x1, max(header + 4, y1)), color, scale * 0.8
        )
    return output


def object_basis(table: Dict[str, object]) -> str:
    """어떤 물건이 이 자리를 몇 % 덮어서 붙었는지.

    한 물건은 그것을 가장 많이 덮는 자리 하나에만 속한다.  그 근거를
    적어두지 않으면 "왜 저 짐이 옆 테이블 것이 아니라 이 테이블 것인가"를
    사람이 확인할 방법이 없다.
    """
    named = []
    for obj in table.get("objects") or []:  # type: ignore[union-attr]
        share = obj.get("share")
        if share is None:
            named.append(str(obj.get("class", "?")))
        else:
            named.append(f"{obj.get('class', '?')} {float(share) * 100:.0f}%")
    return f" — {', '.join(named)}" if named else ""


def explain_seat(table: Dict[str, object]) -> str:
    """`00_읽는법.md` 의 "왜 그렇게 봤나" 칸."""
    state = str(table.get("state", "unknown"))
    raw = str(table.get("raw_state", state))
    reason = str(table.get("reason", ""))

    if state == "occupied":
        code = evidence_code_from_log(table)
        parts = [EVIDENCE_KOREAN[letter] for letter in code if letter in EVIDENCE_KOREAN]
        if parts:
            return " + ".join(parts) + f" ({code})" + object_basis(table)
        # 근거가 이번 tick 에는 안 보이는데 사용중을 유지하는 경우.  성급히
        # 자리를 놔주지 않으려는 규칙이 일한 것이지 버그가 아니다.
        return "이번엔 근거가 안 보였지만 사용중 유지 (연속 3번이 아직 안 됨)"

    # 점유 근거를 처음 본 판단.  reason 은 사용중 쪽 근거를 담고 있으므로
    # 빈자리 사유로 읽으면 거짓말이 된다.
    if raw == "occupied" and state != "occupied":
        code = evidence_code_from_log(table)
        seen = " + ".join(
            EVIDENCE_KOREAN[letter] for letter in code if letter in EVIDENCE_KOREAN
        )
        return (
            f"**점유 근거를 처음 봤다 ({seen or '?'})** — "
            "다음 판단에서 또 보이면 사용중이 된다"
        )

    for key, korean in REASON_KOREAN.items():
        if reason.startswith(key):
            return korean
    return reason or "-"


HEADER = """# {title} — 사진 한 장씩 확인하기

사진 {count}장을 시간 순으로 놓았다. **파일 이름만 봐도 무엇을 볼지 알 수 있다.**

```
01_t0000s_점유3빈9모름0_사람3대2_차이+1_★_원본.jpg
   |  |     |              |         |
   |  |     |              |         +-- ★ = 우리와 정답지가 다름
   |  |     |              +-- 우리가 센 사람 : 정답지가 센 사람
   |  |     +-- 우리 좌석 판정
   |  +-- 영상에서 몇 초 지점인가
   +-- 순서
```

`_원본`은 아무것도 안 그린 사진, `_판정`은 우리가 판정을 그려 넣은 사진이다.
파일 순서상 원본이 먼저 온다.

### 판정 그림 보는 법

| 그림 | 뜻 |
|---|---|
| **빨간 상자 `X T1 (t)`** | 그 자리를 **사용 중**으로 봤다. 괄호는 그 이유다 |
| **초록 상자 `O T3`** | 그 자리를 **빈자리**로 봤다 |
| **회색 상자 `? T5`** | **모름** — 보이긴 하는데 판단이 안 됐다 |
| **회색 상자 `? T3 (t)`** | 모름인데 괄호가 있으면 **점유 근거를 방금 처음 본 자리**다. 다음 판단에서 또 보이면 사용중이 된다 |
| **파란 상자 `person`** | 우리가 **사람으로 잡은 것** |
| 맨 위 검은 띠 | 그 시점의 합계 |

**사용중 옆 괄호 글자 = 왜 사용중인가**

| 글자 | 뜻 |
|---|---|
| `s` | 사람이 앉아 있다 |
| `t` | 책상 위에 짐이 있다 |
| `c` | 의자에 짐이 있다 |

`stc` 처럼 붙어 나오면 근거가 여러 개라는 뜻이다.

### 특히 이걸 봐줘

- **파란 상자가 없는 사람이 있는가** — 그게 우리가 놓친 사람이다
- **사람이 없는 곳에 파란 상자가 있는가** — 헛것을 본 것이다
- **빨간 자리의 괄호 글자가 실제와 맞는가** — `(t)`인데 책상이 비었으면 헛것이다
- **초록 자리가 정말 비어 있는가** — 이게 틀리면 손님을 남의 자리로 보낸다
- **괄호가 붙은 회색 자리** — 손님이 방금 앉았거나, 헛것을 본 것이다. 어느 쪽인지 봐줘

**원본을 먼저 보고 직접 세어 본 뒤 판정을 보는 것**을 권한다.

---

"""


def render_readme(title: str, ticks: Sequence[Tick]) -> str:
    starred = [tick for tick in ticks if tick.gap]
    lines = [HEADER.format(title=title, count=len(ticks))]
    if starred:
        marks = ", ".join(
            f"{tick.timestamp:.0f}초 ({tick.gap:+d})" for tick in starred
        )
        lines.append(f"**★ 표시 {len(starred)}장**: {marks}\n")
    else:
        lines.append("**★ 없음** — 사람 수는 전부 정답지와 같았다.\n")

    for tick in ticks:
        summary = tick.summary
        star = " ★ 사람 수가 어긋남" if tick.gap else ""
        lines.append(f"## {tick.index:02d}. {tick.timestamp:.0f}초{star}\n")
        lines.append(
            f"- 우리가 잡은 사람: **{tick.our_people}명** "
            f"(자세 판정 — 앉음 {summary.get('seated_poses', 0)} / "
            f"섬 {summary.get('standing_poses', 0)} / "
            f"모름 {summary.get('unknown_poses', 0)})"
        )
        if tick.pose_total != tick.our_people:
            # 사람 찾기와 자세 판정은 서로 다른 모델이라 수가 갈릴 수 있다.
            # 갈린 것 자체가 볼거리라 숨기지 않는다.
            lines.append(
                f"  - 두 모델이 다르게 봤다: 사람 찾기 {tick.our_people}명, "
                f"자세 판정 {tick.pose_total}명"
            )
        if tick.judgement is None:
            # 판단 주기가 바뀌면 예전에 채점하지 않은 시점이 생긴다.
            lines.append("- 정답지: **이 시점은 아직 채점되지 않았다** — 직접 세어봐줘")
        elif tick.truth_people is None:
            lines.append("- 정답지: 사람 수를 채점할 수 없다고 했다")
        else:
            lines.append(f"- 정답지가 센 사람: **{tick.truth_people}명**")
        if tick.judgement is not None and tick.judgement.note:
            lines.append(f"- 정답지 메모: {tick.judgement.note}")
        lines.append(
            f"- 우리 좌석 판정: **사용중 {summary.get('occupied', 0)} / "
            f"빈자리 {summary.get('empty', 0)} / "
            f"모름 {summary.get('unknown', 0)}**\n"
        )
        lines.append("| 자리 | 판정 | 왜 그렇게 봤나 |")
        lines.append("|---|---|---|")
        for table in tick.record.get("tables") or []:  # type: ignore[union-attr]
            state = STATE_KOREAN.get(str(table.get("state")), str(table.get("state")))
            lines.append(f"| {seat_name(table)} | {state} | {explain_seat(table)} |")
        lines.append("")
    return "\n".join(lines)


def build(
    run_dir: Path, title: str, output_dir: Optional[Path] = None
) -> Path:
    log_path = run_dir / "log.jsonl"
    clean_dir = run_dir / "clean"
    if not log_path.is_file():
        raise SystemExit(f"로그가 없다: {log_path}")
    if not clean_dir.is_dir():
        raise SystemExit(
            f"원본 사진이 없다: {clean_dir}\n"
            "seatnow.py 를 --frames 와 같이 돌려야 clean/ 이 생긴다."
        )

    records = load_jsonl(log_path)
    judgements = load_judgements(run_dir / "judge")
    ticks = build_ticks(records, judgements)

    review_dir = output_dir or (run_dir / "review")
    review_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for tick in ticks:
        source = clean_dir / f"{tick.stem}.jpg"
        if not source.is_file():
            print(f"[건너뜀] 원본 사진 없음: {source}", file=sys.stderr)
            continue
        frame = read_image(source)
        if frame is None:
            print(f"[건너뜀] 사진을 읽을 수 없음: {source}", file=sys.stderr)
            continue
        shutil.copyfile(source, review_dir / f"{tick.filename_stem}_원본.jpg")
        write_image(
            review_dir / f"{tick.filename_stem}_판정.jpg",
            render_verdict(frame, tick),
        )
        written += 1

    readme = review_dir / "00_읽는법.md"
    readme.write_text(render_readme(title, ticks), encoding="utf-8")
    print(f"{review_dir} - 사진 {written}쌍 + 읽는법")
    return review_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="results/<이름> 폴더")
    parser.add_argument("--title", default=None, help="읽는법 제목 (기본: 폴더 이름)")
    parser.add_argument(
        "--output", type=Path, default=None, help="저장 위치 (기본: <run_dir>/review)"
    )
    args = parser.parse_args(argv)
    build(args.run_dir, args.title or args.run_dir.name, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
