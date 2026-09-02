"""정답지와 실행 로그를 나란히 놓고, 우리가 어느 칸에서 틀렸는지 센다.

    python -m checks.score_answers results/angle1_layout \
        --answers results/angle1_layout/angle_answer.md

**정답지는 사람의 눈이다.**  사장님은 사진을 보고 "지금 거기 사람/물건이
있나 없나"를 적었지, 우리 규칙이 무엇을 해야 하는지를 적은 게 아니다.
그래서 정답과 우리 판정이 다르다고 곧바로 오답이 아니다 — **잘 짜인
규칙이 일부러 모름을 내는 자리**가 있고, 그건 맞는 판단이다.

## 채점이 면제하는 두 가지 (2026-09-02 사장님 공지)

1. **가림 모름** — 사람이 자리를 덮어서 모름을 냈다.  사람이 비키면
   판단한다.  정답이 사용중이든 빈자리든 **지금은 모름이 맞다**
2. **전이 모름** — 사용중↔빈자리로 바뀌는 중이라 확정을 기다린다.
   성급히 놔주면 손님이 남의 자리로 간다

## 그래서 한 칸을 두 겹으로 본다

로그에는 값이 두 개 있다.

* ``state``     — 시간 로직을 통과한 최종. 손님 앱에 나가는 답
* ``raw_state`` — **이번 판단에서 실제로 본 것**

| 앱 답 | 근거 | 등급 | 고치나 |
|---|---|---|---|
| ✅ | ✅ | 맞음 | — |
| ❌ | ✅ | **지연** — 근거는 맞고 확정 대기 중 | ❌ 규칙대로다 |
| ✅ | ❌ | **유지** — 이번엔 못 봤는데 과거 근거로 버텼다 | ⚠️ 낮은 우선순위 |
| ❌ | ❌ | **오답** | ✅ 최우선 |

## 근거 글자도 채점한다

정답지의 괄호(`X (t,c)`)는 그 자리에 **무엇이** 있는지를 말한다.  상태가
맞아도 근거를 놓친 칸이 있고 — 예를 들어 우리가 `t`만 보고 `c`(의자 위
가방)를 못 본 자리 — 그게 곧 다음 숙제다.  상태 오답과 섞지 않고 따로
센다.

모델도 GPU도 안 쓴다.  로그와 글자만 읽으므로 테스트로 고정할 수 있다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from checks.answer_key import AnswerKey, normalize_seat, parse_answer_key, validate
from checks.verify_seatnow import load_jsonl


PROJECT_DIR = Path(__file__).resolve().parents[1]

# 사람 가림 때문에 나온 모름.  사람이 비키면 풀리므로 코드로 고칠 오답이
# 아니다 (engine/seatnow_report.py 의 ReasonCode 와 같은 어휘).
OCCLUSION_REASONS = (
    "occluded_by_person",
    "nearby_person_pose_unknown",
    "spans_multiple_seats",
    "occluded_lower_body",
    "temporarily_occluded",
)

# 오답의 방향.  손님에게 끼치는 해가 서로 달라 한 덩어리로 세지 않는다.
DIRECTION_HARM = {
    "놓침": "손님이 남의 자리로 간다 — 가장 나쁘다",
    "헛것": "빈자리가 있는데 없다고 한다",
    "과잉모름": "판단할 수 있었는데 모름을 냈다",
    "과소모름": "근거 없이 단정했다",
    "기타": "",
}

CATEGORY_ORDER = ("오답", "유지", "지연", "가림모름", "맞음", "화각밖", "정답없음")

# 고칠 값이 있는 등급.  '지연'과 '가림모름'은 규칙이 제대로 동작한 것이다.
FIXABLE_CATEGORIES = ("오답", "유지")

EVIDENCE_MEANING = {"s": "사람이 앉음", "t": "책상에 짐", "c": "의자에 짐"}


def review_state(table: Mapping[str, object]) -> str:
    """검수 사진에 그려진 상태.  사장님이 실제로 본 것과 같아야 한다.

    ``checks/make_review.py`` 와 같은 규칙이다 — 이번 판단에서 아무 근거도
    못 봤는데 사용중을 유지하는 자리는 사진에서 회색(모름)으로 보인다.
    """
    state = str(table.get("state", "unknown"))
    raw = str(table.get("raw_state", state))
    if state == "occupied" and raw == "empty":
        return "unknown"
    return state


def is_occlusion_unknown(table: Mapping[str, object]) -> bool:
    """이 자리의 모름이 '사람이 가려서'인가.

    가림은 코드로 푸는 UNKNOWN(엔지니어링 지표)이 아니라, 사람이 비키면
    저절로 풀리는 상태다.  사장님 공지대로 맞는 판단으로 센다.
    """
    if review_state(table) != "unknown":
        return False
    reason = str(table.get("reason", ""))
    return any(marker in reason for marker in OCCLUSION_REASONS)


def direction(truth: str, ours: str) -> Optional[str]:
    """무엇을 어느 쪽으로 틀렸는가.  같으면 방향이 없다."""
    if truth == ours:
        return None
    if truth == "occupied" and ours == "empty":
        return "놓침"
    if truth == "empty" and ours == "occupied":
        return "헛것"
    if ours == "unknown":
        return "과잉모름"
    if truth == "unknown":
        return "과소모름"
    return "기타"


@dataclass
class Verdict:
    """정답지 한 칸에 대한 채점 결과 + 진단기가 쓸 증거."""

    timestamp: float
    seat: str
    truth: str
    app_state: str
    evidence_state: str
    category: str
    direction: Optional[str]
    reason: str
    note: str = ""
    truth_evidence: str = ""
    our_evidence: str = ""
    objects: List[Mapping[str, object]] = field(default_factory=list)
    chair_objects: List[Mapping[str, object]] = field(default_factory=list)
    seat_box: List[float] = field(default_factory=list)
    connected_chairs: List[Mapping[str, object]] = field(default_factory=list)

    @property
    def app_correct(self) -> bool:
        return self.app_state == self.truth

    @property
    def evidence_correct(self) -> bool:
        return self.evidence_state == self.truth

    @property
    def is_fixable(self) -> bool:
        """기계가 손댈 값이 있는 칸인가 — 근거를 잘못 본 칸이다."""
        return self.category in FIXABLE_CATEGORIES

    @property
    def missed_evidence(self) -> str:
        """정답에는 있는데 우리가 못 본 근거 글자."""
        if self.category in ("화각밖", "정답없음"):
            return ""
        return "".join(
            letter for letter in "stc"
            if letter in self.truth_evidence and letter not in self.our_evidence
        )

    @property
    def imagined_evidence(self) -> str:
        """우리는 봤는데 정답에는 없는 근거 글자."""
        if self.category in ("화각밖", "정답없음"):
            return ""
        return "".join(
            letter for letter in "stc"
            if letter in self.our_evidence and letter not in self.truth_evidence
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "seat": self.seat,
            "truth": self.truth,
            "app_state": self.app_state,
            "evidence_state": self.evidence_state,
            "category": self.category,
            "direction": self.direction,
            "reason": self.reason,
            "note": self.note,
            "truth_evidence": self.truth_evidence,
            "our_evidence": self.our_evidence,
            "missed_evidence": self.missed_evidence,
            "imagined_evidence": self.imagined_evidence,
            "objects": list(self.objects),
            "chair_objects": list(self.chair_objects),
            "seat_box": list(self.seat_box),
            "connected_chairs": list(self.connected_chairs),
        }


def _our_evidence_code(table: Mapping[str, object]) -> str:
    """엔진이 남긴 근거 글자.  옛 로그를 위해 사유 문자열도 본다."""
    code = table.get("evidence_code")
    if code is not None:
        return str(code)
    from engine.seatnow_core import evidence_code_from_log

    return evidence_code_from_log(table)


def grade_seat(
    truth: str, table: Mapping[str, object], truth_evidence: str = ""
) -> Verdict:
    """한 칸을 채점한다 (모듈 설명의 표 + 면제 두 가지)."""
    app_state = str(table.get("state", "unknown"))
    evidence_state = str(table.get("raw_state", app_state))

    if truth == "ignore":
        # 화각이 아예 못 보는 자리는 판정 오답이 아니라 설치 지표다(CLAUDE.md).
        category = "화각밖"
    elif app_state == truth and evidence_state == truth:
        category = "맞음"
    elif is_occlusion_unknown(table):
        # 사람이 가려서 모름 — 비키면 판단한다.  맞는 판단이다.
        category = "가림모름"
    elif evidence_state == truth:
        # 근거는 맞게 봤고, 확정에 필요한 횟수를 아직 못 채웠다.
        category = "지연"
    elif app_state == truth:
        # 이번엔 근거를 못 봤는데 과거 근거로 답을 지켰다.
        category = "유지"
    else:
        category = "오답"

    return Verdict(
        timestamp=float(table.get("timestamp", 0.0) or 0.0),
        seat=str(table.get("layout_name") or table.get("label") or "?"),
        truth=truth,
        app_state=app_state,
        evidence_state=evidence_state,
        category=category,
        direction=(
            None
            if category in ("맞음", "화각밖")
            else direction(truth, evidence_state)
        ),
        reason=str(table.get("reason", "")),
        truth_evidence=truth_evidence,
        our_evidence=_our_evidence_code(table),
        objects=list(table.get("objects") or []),
        chair_objects=list(table.get("chair_objects") or []),
        seat_box=[float(value) for value in (table.get("box") or [])],
        connected_chairs=list(table.get("connected_chairs") or []),
    )


def score_run(records: Sequence[Mapping[str, object]], key: AnswerKey) -> List[Verdict]:
    """실행 로그의 모든 칸을 정답지와 맞춰본다.

    정답이 없는 칸은 '맞음'으로 넘기지 않고 ``정답없음`` 으로 남긴다.
    빠진 칸을 맞다고 가정하면 회귀가 조용히 통과한다.
    """
    verdicts: List[Verdict] = []
    for record in sorted(records, key=lambda item: float(item.get("timestamp", 0.0))):
        timestamp = float(record.get("timestamp", 0.0))
        for table in record.get("tables") or []:
            seat = str(table.get("layout_name") or table.get("label") or "?")
            answer = key.lookup(timestamp, seat)
            merged = dict(table)
            merged["timestamp"] = timestamp
            if answer is None:
                verdict = grade_seat("?", merged)
                verdict.category = "정답없음"
                verdict.direction = None
            else:
                verdict = grade_seat(answer.state, merged, answer.evidence_code)
                verdict.note = answer.note
            verdict.seat = normalize_seat(verdict.seat)
            verdicts.append(verdict)
    return verdicts


def summarize(verdicts: Sequence[Verdict]) -> Dict[str, object]:
    """칸 수를 등급별·방향별로 센다.  화각밖은 분모에서 빠진다."""
    summary: Dict[str, object] = {"total": len(verdicts)}
    for category in CATEGORY_ORDER:
        summary[category] = sum(1 for v in verdicts if v.category == category)
    summary["scored"] = len(verdicts) - int(summary["화각밖"])
    # 규칙이 제대로 동작한 칸까지 합한 것.  "맞음"만 세면 가림·확정 대기를
    # 오답처럼 보이게 해서, 그 규칙들을 없애는 쪽으로 유인이 생긴다.
    summary["accepted"] = (
        int(summary["맞음"]) + int(summary["지연"]) + int(summary["가림모름"])
    )
    summary["fixable"] = sum(1 for v in verdicts if v.is_fixable)
    summary["evidence_missed"] = sum(1 for v in verdicts if v.missed_evidence)
    summary["evidence_imagined"] = sum(1 for v in verdicts if v.imagined_evidence)
    directions: Dict[str, int] = {}
    for verdict in verdicts:
        if verdict.direction and verdict.is_fixable:
            directions[verdict.direction] = directions.get(verdict.direction, 0) + 1
    summary["directions"] = directions
    return summary


def _objects_phrase(verdict: Verdict) -> str:
    parts = []
    for obj in list(verdict.objects)[:3]:
        share = obj.get("share")
        share_text = f" {float(share) * 100:.0f}%" if share is not None else ""
        parts.append(f"{obj.get('class')}{share_text}")
    for obj in list(verdict.chair_objects)[:2]:
        parts.append(f"의자:{obj.get('class')}")
    return ", ".join(parts) if parts else "—"


def _evidence_phrase(code: str) -> str:
    if not code:
        return "—"
    return f"{code} ({', '.join(EVIDENCE_MEANING[letter] for letter in code)})"


def render_report(verdicts: Sequence[Verdict], title: str) -> str:
    """사람이 읽을 채점표.  고칠 칸부터, 심한 것부터 나온다."""
    summary = summarize(verdicts)
    scored = int(summary["scored"])
    lines = [f"# 채점표 — {title}", ""]
    lines.append(
        f"**{scored}칸 중 {summary['accepted']}칸이 옳게 나왔다** "
        f"(딱 맞음 {summary['맞음']} + 확정 대기 {summary['지연']} "
        f"+ 가림 모름 {summary['가림모름']})"
    )
    lines.append("")
    lines.append(f"**고칠 칸 {summary['fixable']}개** — 오답 {summary['오답']} · 유지 {summary['유지']}")
    if summary["정답없음"]:
        lines.append(f"⚠️ 정답이 없는 칸 {summary['정답없음']}개 — 정답지가 덜 찼다")
    if summary["화각밖"]:
        lines.append(f"화각밖(IGNORE) {summary['화각밖']}칸은 분모에서 뺐다")
    lines.append("")
    lines += [
        "> 확정 대기와 가림 모름은 **규칙이 일부러 그렇게 낸 것**이라 오답이 아니다.",
        "> 성급히 확정하면 손님이 남의 자리로 간다 (2026-09-02 공지).",
        "",
    ]

    if summary["directions"]:
        lines += [
            "## 어느 쪽으로 틀렸나",
            "",
            "| 방향 | 칸 | 손님에게 무슨 일이 |",
            "|---|---:|---|",
        ]
        for name, harm in DIRECTION_HARM.items():
            count = summary["directions"].get(name, 0)
            if count:
                lines.append(f"| {name} | {count} | {harm} |")
        lines.append("")

    lines += [
        "## 고쳐야 할 칸",
        "",
        "| 시각 | 자리 | 정답 | 앱 | 근거 | 등급 | 방향 | 우리가 본 것 | 사유 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fixable = [v for v in verdicts if v.is_fixable]
    fixable.sort(key=lambda v: (v.category != "오답", v.timestamp, v.seat))
    for verdict in fixable:
        truth = verdict.truth + (
            f" ({verdict.truth_evidence})" if verdict.truth_evidence else ""
        )
        lines.append(
            f"| {verdict.timestamp:g}초 | {verdict.seat} | {truth} "
            f"| {verdict.app_state} | {verdict.evidence_state} | {verdict.category} "
            f"| {verdict.direction or '—'} | {_objects_phrase(verdict)} "
            f"| `{verdict.reason or '—'}` |"
        )
    if not fixable:
        lines.append("| — | — | — | — | — | — | — | — | 고칠 칸이 없다 |")
    lines.append("")

    missed = [v for v in verdicts if v.missed_evidence or v.imagined_evidence]
    lines += [
        "## 근거를 놓쳤거나 헛본 칸",
        "",
        "상태는 맞았어도 **무엇 때문에 그런지**를 틀린 칸이다. 상태에 안 드러날 뿐,",
        "탐지가 무엇을 못 보는지 그대로 말해준다.",
        "",
        "| 시각 | 자리 | 상태 | 정답 근거 | 우리 근거 | 놓친 것 | 헛본 것 |",
        "|---|---|---|---|---|---|---|",
    ]
    for verdict in missed:
        lines.append(
            f"| {verdict.timestamp:g}초 | {verdict.seat} | {verdict.category} "
            f"| {_evidence_phrase(verdict.truth_evidence)} "
            f"| {_evidence_phrase(verdict.our_evidence)} "
            f"| **{verdict.missed_evidence or '—'}** | {verdict.imagined_evidence or '—'} |"
        )
    if not missed:
        lines.append("| — | — | — | — | — | — | 근거까지 다 맞았다 |")
    lines.append("")

    exempt = [v for v in verdicts if v.category in ("지연", "가림모름")]
    if exempt:
        lines += [
            "## 규칙이 일부러 모름을 낸 칸 (고치지 않는다)",
            "",
            "| 시각 | 자리 | 정답 | 등급 | 사유 |",
            "|---|---|---|---|---|",
        ]
        for verdict in exempt:
            lines.append(
                f"| {verdict.timestamp:g}초 | {verdict.seat} | {verdict.truth} "
                f"| {verdict.category} | `{verdict.reason or '—'}` |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="정답지로 실행을 채점한다")
    parser.add_argument("run_dir", type=Path, help="results/<이름> 폴더")
    parser.add_argument(
        "--answers", type=Path, help="정답지 (기본: <run_dir>/angle_answer.md)"
    )
    parser.add_argument("--log", type=Path, help="기본: <run_dir>/log.jsonl")
    parser.add_argument("--output", type=Path, help="채점표 (기본: <run_dir>/채점표.md)")
    parser.add_argument(
        "--misses",
        type=Path,
        help="진단기가 읽을 오답 JSON (기본: <run_dir>/오답.json)",
    )
    parser.add_argument("--title", default="", help="채점표 제목")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    answers_path = args.answers or run_dir / "angle_answer.md"
    log_path = args.log or run_dir / "log.jsonl"

    if not answers_path.exists():
        print(f"정답지가 없다: {answers_path}")
        return 2
    records = load_jsonl(log_path)
    if not records:
        print(f"로그가 비었다: {log_path}")
        return 2

    key = parse_answer_key(answers_path.read_text(encoding="utf-8"))
    timestamps = [float(record.get("timestamp", 0.0)) for record in records]
    key.resolve_photo_numbers(timestamps)

    seats: List[str] = []
    for table in records[0].get("tables") or []:
        name = str(table.get("layout_name") or table.get("label") or "?")
        if name not in seats:
            seats.append(name)

    problems = validate(key, seats, timestamps)
    if problems:
        print(f"{answers_path}: 정답지에 문제가 {len(problems)}개 있다")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    verdicts = score_run(records, key)
    summary = summarize(verdicts)

    output = args.output or run_dir / "채점표.md"
    misses = args.misses or run_dir / "오답.json"
    # 한글이 든 경로에는 Path 로 쓴다 (plan.md §7 윈도우 함정).
    output.write_text(render_report(verdicts, args.title or run_dir.name), encoding="utf-8")
    misses.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "log": str(log_path),
                "summary": summary,
                "misses": [v.as_dict() for v in verdicts if v.is_fixable],
                "evidence_gaps": [
                    v.as_dict()
                    for v in verdicts
                    if (v.missed_evidence or v.imagined_evidence) and not v.is_fixable
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"채점표: {output}")
    print(f"오답 목록: {misses}  ({summary['fixable']} cells)")
    print(
        f"scored={summary['scored']} accepted={summary['accepted']} "
        f"(exact={summary['맞음']} delayed={summary['지연']} occluded={summary['가림모름']}) "
        f"| wrong={summary['오답']} held={summary['유지']} "
        f"| evidence missed={summary['evidence_missed']} imagined={summary['evidence_imagined']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
