"""JSONL 한 줄 → 텔레메트리 표의 줄들.

설계 근거는 `문서/판정개선_데이터설계.md` §10.  여기서 만드는 것은 "판정을 계속
좋게 만들기 위한 기록"이고, 손님 앱이 보는 `cafe_live` 와는 완전히 별개다.

**순수 파이썬만 쓴다.** numpy·cv2·requests 를 안 쓰므로 박스 밖 아무 데서나
로그 파일만 있으면 돌려볼 수 있다 — 지난 기록을 다시 채점하는 게 이 설계의
핵심이라(§5) 그게 중요하다.

두 가지를 분리해서 센다:

* **요약(②)은 모든 틱을 센다.**  고른 것만 세면 "모름 비율"이 통째로 틀어진다.
* **본체(①)는 고른 틱만 남긴다.**  잘 도는 97% 는 보낼 필요가 없다.

나가는 위치값은 **전부 그 자리를 1로 본 비율**이다.  화면 좌표는 나가지
않는다 — 방 안 어디였는지 복원이 안 되고, 판정을 배우는 데는 비율이면 충분하다.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

TICKS_TABLE = "seatnow_seat_ticks"
DAILY_TABLE = "seatnow_seat_daily"
RUNS_TABLE = "seatnow_runs"
SCENERY_TABLE = "seatnow_scenery"

#: 잘 돌아간 틱 중 이 비율만큼을 대조군으로 남긴다.  어려운 것만 모으면
#: 그 데이터로 배운 규칙이 편향된다 (§10-3).
DEFAULT_CONTROL_RATE = 0.01

_COUNTED_STATES = ("occupied", "empty", "unknown", "ignore")
_BAR_KIND = "counted_zone"


# ---------------------------------------------------------------------------
# 영업시간 — 4-1(b) 장식 지도와 4-3(4) "평소와 다른 것" 이 둘 다 여기 달려 있다
# ---------------------------------------------------------------------------


def parse_open_window(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """``"09:00-22:00"`` → ``(540, 1320)`` (자정부터의 분).

    끝이 시작보다 이르면 자정을 넘기는 영업으로 본다 (``"18:00-02:00"``).
    형식이 이상하면 ``None`` — 영업 여부를 모른다고 답하지, 추측하지 않는다.
    """
    if not spec:
        return None
    try:
        start_text, end_text = str(spec).split("-", 1)
        start = _minutes_of_day(start_text)
        end = _minutes_of_day(end_text)
    except (ValueError, TypeError):
        return None
    if start is None or end is None:
        return None
    return (start, end)


def _minutes_of_day(text: str) -> Optional[int]:
    parts = text.strip().split(":")
    if len(parts) != 2:
        return None
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 24 and 0 <= minute < 60):
        return None
    return hour * 60 + minute


def is_open_at(wall_clock: Optional[str], window: Optional[Tuple[int, int]]) -> Optional[bool]:
    """그 시각이 영업시간 안이었나.  모르면 ``None`` (거짓이 아니다)."""
    if window is None:
        return None
    moment = parse_wall_clock(wall_clock)
    if moment is None:
        return None
    now = moment.hour * 60 + moment.minute
    start, end = window
    if start == end:
        return True  # 24시간
    if start < end:
        return start <= now < end
    return now >= start or now < end  # 자정을 넘기는 영업


def parse_wall_clock(wall_clock: Optional[str]) -> Optional[datetime]:
    """``time.strftime("%Y-%m-%dT%H:%M:%S%z")`` 가 쓴 문자열을 되읽는다."""
    if not wall_clock:
        return None
    text = str(wall_clock).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for pattern in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# 기하 — 전부 "그 자리를 1로 본" 비율로 바꾼다
# ---------------------------------------------------------------------------


def _box(raw: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    if x2 < x1 or y2 < y1:
        return None
    return (x1, y1, x2, y2)


def seat_relative(box: Any, seat_box: Any) -> Optional[Dict[str, float]]:
    """물건 네모를 자리 기준 비율로.  1을 넘을 수 있다 (자리 밖으로 튀어나온 것)."""
    inner, outer = _box(box), _box(seat_box)
    if inner is None or outer is None:
        return None
    width = max(1e-6, outer[2] - outer[0])
    height = max(1e-6, outer[3] - outer[1])
    return {
        "cx": round(((inner[0] + inner[2]) / 2.0 - outer[0]) / width, 4),
        "cy": round(((inner[1] + inner[3]) / 2.0 - outer[1]) / height, 4),
        "w": round((inner[2] - inner[0]) / width, 4),
        "h": round((inner[3] - inner[1]) / height, 4),
    }


def overlap_fraction(box: Any, seat_box: Any) -> float:
    """물건의 몇 할이 그 자리 안에 있나 (물건 넓이 기준)."""
    inner, outer = _box(box), _box(seat_box)
    if inner is None or outer is None:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if area <= 0:
        return 0.0
    overlap_w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    overlap_h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (overlap_w * overlap_h) / area


def border_margin(seat_box: Any, frame_width: Any, frame_height: Any) -> Optional[float]:
    """자리가 화면 가장자리에서 얼마나 떨어졌나 (0 = 끝에 붙음).

    설치 결함과 코드 결함을 가르는 값이다 (§4-3).  IGNORE 가 많은 자리가
    전부 0 근처면 카메라를 다시 단다.  코드로 풀 문제가 아니다.
    """
    outer = _box(seat_box)
    try:
        width, height = float(frame_width or 0), float(frame_height or 0)
    except (TypeError, ValueError):
        return None
    if outer is None or width <= 0 or height <= 0:
        return None
    margins = (
        outer[0] / width,
        (width - outer[2]) / width,
        outer[1] / height,
        (height - outer[3]) / height,
    )
    return round(max(0.0, min(margins)), 4)


def aspect_ratio(box: Any) -> Optional[float]:
    """세로 ÷ 가로.  1.75 라는 손으로 그은 선을 다시 긋는 데 쓰는 값이다 (§5)."""
    inner = _box(box)
    if inner is None:
        return None
    width = max(1e-6, inner[2] - inner[0])
    return round((inner[3] - inner[1]) / width, 3)


# ---------------------------------------------------------------------------
# 실행 설정 — 이게 없으면 지난주와 비교할 수 없다
# ---------------------------------------------------------------------------

#: 매번 달라지지만 판정에는 영향이 없는 것들.  지문에서 뺀다.
_VOLATILE_RUN_KEYS = ("input", "decode", "started_at", "wall_clock", "host")


def settings_fingerprint(run_context: Optional[Mapping[str, Any]]) -> str:
    """판정 설정 전체의 지문.  같은 설정이면 재시작해도 같은 값이 나온다."""
    stable = {
        key: value
        for key, value in sorted((run_context or {}).items())
        if key not in _VOLATILE_RUN_KEYS
    }
    canonical = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    return str(uuid.uuid4())


def run_row(
    run_context: Optional[Mapping[str, Any]],
    *,
    run_id: str,
    cafe_id: str,
    started_at: str,
    ended_at: Optional[str] = None,
) -> Dict[str, Any]:
    context = dict(run_context or {})
    models = dict(context.get("models") or {})
    settings = dict(context.get("settings") or {})
    source = dict(context.get("input") or {})
    return {
        "run_id": run_id,
        "cafe_id": cafe_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "schema_version": SCHEMA_VERSION,
        "box_version": context.get("box_version") or context.get("version"),
        "profile": context.get("profile"),
        "det_model": models.get("detector"),
        "det_sha256": models.get("detector_sha256"),
        "pose_model": models.get("pose"),
        "imgsz": settings.get("imgsz"),
        "pose_imgsz": settings.get("pose_imgsz"),
        "tick_seconds": settings.get("sample_seconds"),
        "median_frames": settings.get("median_frames"),
        "settings_hash": settings_fingerprint(context),
        "layout_version": context.get("layout_version"),
        "frame_width": source.get("width"),
        "frame_height": source.get("height"),
    }


# ---------------------------------------------------------------------------
# 한 자리의 한 순간
# ---------------------------------------------------------------------------


def _seat_kind(layout_kind: Optional[str]) -> str:
    return "bar_seat" if layout_kind == _BAR_KIND else "table"


def burst_votes(table: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """5장 투표에서 ``(최종 답에 동의한 장 수, 전체 장 수)``.

    엔진이 이미 ``vote_counts`` 를 남기고 있다 (``{"occupied": 3, "empty": 2}``).
    **동의가 전체보다 적으면 갈린 것이다** — 그 순간이 §4-3(2) "한 장만 보여도
    인정" 을 시험할 재료다.  묶음 투표를 안 쓴 기록이면 둘 다 ``None``.
    """
    counts = table.get("vote_counts")
    if not isinstance(counts, Mapping) or not counts:
        return (None, None)
    try:
        total = sum(int(value) for value in counts.values())
    except (TypeError, ValueError):
        return (None, None)
    if total <= 0:
        return (None, None)
    answer = str(table.get("raw_state") or table.get("state") or "")
    try:
        seen = int(counts.get(answer, 0))
    except (TypeError, ValueError):
        seen = 0
    return (seen, total)


def person_rows(poses: Sequence[Mapping[str, Any]], seat_box: Any) -> List[Dict[str, Any]]:
    """이 자리에 걸친 사람들.  §5의 1.75 선을 다시 긋는 재료가 여기 다 있다."""
    rows: List[Dict[str, Any]] = []
    for pose in poses or []:
        if not isinstance(pose, Mapping):
            continue
        cover = overlap_fraction(pose.get("box"), seat_box)
        if cover <= 0.0:
            continue
        rows.append(
            {
                "hw_ratio": aspect_ratio(pose.get("box")),
                "pose_state": pose.get("state"),
                # 사유는 반드시 코드로 줄여서 내보낸다.  원문은 측정값이 붙어
                # 있고(``left_hka=95.0<110``) 감사되지 않은 숫자를 실어 나른다 (§10-4).
                # 값이 필요하면 바로 아래 ``angles`` 에 감사된 형태로 들어 있다.
                "pose_reason": _short_reason(pose.get("reason")),
                "conf": pose.get("confidence"),
                "angles": dict(pose.get("angles") or {}),
                "cover": round(cover, 4),
                "geom": seat_relative(pose.get("box"), seat_box),
            }
        )
    return rows


def object_rows(table: Mapping[str, Any], seat_box: Any) -> List[Dict[str, Any]]:
    """이 자리의 물건들.  ``kept=false`` 가 §4-2 "버려진 짐" 의 전부다.

    ``kept``/``drop_reason`` 은 엔진이 아직 안 넣어줄 수 있다.  그때는 통과한
    것만 보이고 ``kept`` 는 참이 된다 — 없는 걸 지어내지 않는다.
    """
    rows: List[Dict[str, Any]] = []
    for item in table.get("objects") or []:
        if not isinstance(item, Mapping):
            continue
        kept = item.get("kept")
        rows.append(
            {
                "class": item.get("class"),
                "conf": item.get("confidence"),
                "share": item.get("share"),
                "kept": True if kept is None else bool(kept),
                "drop_reason": item.get("drop_reason"),
                "geom": seat_relative(item.get("box"), seat_box),
            }
        )
    return rows


def _is_hard(table: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> bool:
    """§10-3 "남길 것" — 틀렸거나, 갈렸거나, 뭔가를 버린 순간."""
    state = str(table.get("state") or "")
    if state == "unknown":
        return True
    raw = table.get("raw_state")
    settled = table.get("persistent_state")
    if raw is not None and settled is not None and raw != settled:
        return True
    if any(item.get("kept") is False for item in objects):
        return True
    seen, total = burst_votes(table)
    if seen is not None and total is not None and seen < total:
        return True  # 5장 투표가 만장일치가 아니었다
    return False


class TelemetrySelector:
    """틱을 받아 ① 고른 줄을 내놓고, 동시에 ② 요약을 모든 틱으로 센다.

    상태를 가지는 이유는 둘뿐이다: 전이(앱에 나간 값이 바뀐 순간)를 알려면
    직전 값을 기억해야 하고, 대조군을 뽑으려면 난수가 필요하다.
    """

    def __init__(
        self,
        cafe_id: str,
        run_id: str,
        *,
        open_window: Optional[Tuple[int, int]] = None,
        control_rate: float = DEFAULT_CONTROL_RATE,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.cafe_id = cafe_id
        self.run_id = run_id
        self.open_window = open_window
        self.control_rate = max(0.0, min(1.0, float(control_rate)))
        self._rng = rng or random.Random()
        self._previous_shown: Dict[str, Optional[str]] = {}
        self._daily: Dict[Tuple[str, str], Counter] = {}
        self._reasons: Dict[Tuple[str, str], Counter] = {}

    # -- ① 본체 -------------------------------------------------------------

    def observe(self, record: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """틱 하나를 받아 **보낼 줄만** 돌려준다.  요약은 조용히 같이 센다."""
        wall_clock = record.get("wall_clock")
        moment = parse_wall_clock(wall_clock)
        tick_at = moment.isoformat() if moment else None
        day = moment.date().isoformat() if moment else None
        source = dict((record.get("run") or {}).get("input") or {})
        open_now = is_open_at(wall_clock, self.open_window)

        rows: List[Dict[str, Any]] = []
        for table in record.get("tables") or []:
            if not isinstance(table, Mapping):
                continue
            seat_id = str(table.get("layout_name") or table.get("label") or "?")
            state = str(table.get("state") or "unknown")
            if day is not None:
                self._count(seat_id, day, state, table.get("reason"))

            seat_box = table.get("box")
            objects = object_rows(table, seat_box)
            votes_seen, votes_total = burst_votes(table)
            shown = table.get("shown_state")
            previous = self._previous_shown.get(seat_id, _UNSET)
            self._previous_shown[seat_id] = shown

            kind = self._sample_kind(table, objects, shown, previous)
            if kind is None or tick_at is None:
                continue
            rows.append(
                {
                    "cafe_id": self.cafe_id,
                    "seat_id": seat_id,
                    "tick_at": tick_at,
                    "run_id": self.run_id,
                    "schema_version": SCHEMA_VERSION,
                    "seat_kind": _seat_kind(table.get("layout_kind")),
                    "capacity": table.get("layout_capacity"),
                    "zone": table.get("layout_zone_name"),
                    "raw_state": str(table.get("raw_state") or state),
                    "settled_state": str(table.get("persistent_state") or state),
                    "shown_state": shown,
                    "reason_code": _reason_code(table),
                    "votes_seen": votes_seen,
                    "votes_total": votes_total,
                    "vote_counts": dict(table.get("vote_counts") or {}),
                    "is_open": open_now,
                    "border_margin": border_margin(
                        seat_box, source.get("width"), source.get("height")
                    ),
                    "novelty": table.get("novelty"),
                    "chairs_linked": len(table.get("connected_chairs") or []),
                    "seated_people": table.get("seated_people"),
                    "confidence": table.get("confidence"),
                    "persons": person_rows(record.get("poses") or [], seat_box),
                    "objects": objects,
                    "sample_kind": kind,
                }
            )
        return rows

    def _sample_kind(
        self,
        table: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
        shown: Any,
        previous: Any,
    ) -> Optional[str]:
        if previous is not _UNSET and previous != shown:
            return "transition"  # 드물고 제일 비싸다.  무조건 남긴다
        if _is_hard(table, objects):
            return "hard"
        if self.control_rate > 0.0 and self._rng.random() < self.control_rate:
            return "control"  # 편향을 막는 대조군 (§10-3)
        return None

    # -- ② 요약 -------------------------------------------------------------

    def _count(self, seat_id: str, day: str, state: str, reason: Any) -> None:
        key = (seat_id, day)
        counter = self._daily.setdefault(key, Counter())
        counter["ticks"] += 1
        counter[state if state in _COUNTED_STATES else "unknown"] += 1
        if state == "unknown":
            self._reasons.setdefault(key, Counter())[_short_reason(reason)] += 1

    def daily_rows(self) -> List[Dict[str, Any]]:
        """지금까지 본 모든 틱의 요약.  **고른 것만이 아니라 전부를 센 값이다.**"""
        rows: List[Dict[str, Any]] = []
        for (seat_id, day), counter in sorted(self._daily.items()):
            ticks = counter["ticks"] or 1
            rows.append(
                {
                    "cafe_id": self.cafe_id,
                    "seat_id": seat_id,
                    "day": day,
                    "ticks": counter["ticks"],
                    "occupied": counter["occupied"],
                    "empty": counter["empty"],
                    "unknown": counter["unknown"],
                    "ignored": counter["ignore"],
                    "reason_counts": dict(self._reasons.get((seat_id, day), {})),
                    "unknown_rate": round(counter["unknown"] / ticks, 4),
                    "ignore_rate": round(counter["ignore"] / ticks, 4),
                }
            )
        return rows

    def reset_daily(self) -> None:
        """요약을 보내고 난 뒤 부른다."""
        self._daily.clear()
        self._reasons.clear()


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return "<unset>"


_UNSET = _Unset()


def _reason_code(table: Mapping[str, Any]) -> Optional[str]:
    if str(table.get("state") or "") != "unknown":
        return None
    return _short_reason(table.get("reason"))


def _short_reason(reason: Any) -> str:
    """``compact_occluded_pose=1.42`` → ``compact_occluded_pose``.

    측정값이 붙은 사유를 그대로 두면 사유마다 값이 달라 셀 수가 없다.
    값은 `persons` 안에 이미 숫자로 들어 있다.
    """
    text = str(reason or "").strip()
    if not text:
        return "unspecified"
    for separator in ("=", ":", " "):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text or "unspecified"


def rows_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    cafe_id: str,
    run_id: str,
    open_window: Optional[Tuple[int, int]] = None,
    control_rate: float = DEFAULT_CONTROL_RATE,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """로그 여러 줄 → ``(고른 틱 줄들, 요약 줄들)``.

    지난 기록을 다시 훑을 때 쓰는 입구다.  박스 없이 돈다.
    """
    selector = TelemetrySelector(
        cafe_id, run_id, open_window=open_window, control_rate=control_rate, rng=rng
    )
    ticks: List[Dict[str, Any]] = []
    for record in records:
        ticks.extend(selector.observe(record))
    return ticks, selector.daily_rows()
