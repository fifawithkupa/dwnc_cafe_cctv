"""오답 난 자리만 골라 "모델이 못 본 건가, 코드가 버린 건가"를 가른다.

    python -m checks.diagnose_miss results/angle1_layout

우리 로그의 `raw_detections` 는 **이미 신뢰도 0.15로 걸러진 뒤**다.  그것만
보면 모델이 0.05로는 봤던 것을 "모델이 못 봤다"로 잘못 분류하고, **임계값만
내리면 잡히는 것을 파인튜닝 숙제로 넘긴다.**

그래서 오답 난 자리만 골라 이미 저장된 `clean/*.jpg` 를 **신뢰도 0.01로
다시 추론**한다.  영상을 다시 디코딩하지 않으니 사진 몇 장이면 몇 초다.
결과는 넷으로 갈린다.  **고치는 사람이 서로 다르다.**

| 딱지 | 뜻 | 누가 고치나 |
|---|---|---|
| **로직** | 우리 임계값 위에서 이미 잡고 있었다 | 코드 — 규칙이 버렸다 |
| **상자밖** | 물건이 자리 상자 바로 밖에 있다 | 설치 검수 — 상자를 다시 그린다 |
| **임계값경계** | 임계값의 1/3 위에서만 보인다 | 코드 — 전역 기본값을 내려보고 전체 점수로 판단 |
| **가구오인** | 상판 위의 물건을 `chair`·`tv` 같은 가구로 불렀다 | 모델 — **못 본 게 아니라 이름이 틀렸다** |
| **파인튜닝** | 그보다 낮거나 아예 안 보인다 | 모델 — 코드로는 못 푼다 |

두 가지를 일부러 안 한다.

* **의자·테이블은 배경이다.**  어느 자리 위에나 늘 보이므로 손님 짐의
  답이 될 수 없다.  이걸 "규칙이 버렸다"로 세면 탐지 실패가 전부 로직
  문제로 둔갑한다.  단, **자리보다 훨씬 작은 가구가 상판 위에 서 있고
  사람이 그린 의자와도 안 맞으면** 그건 배경이 아니라 이름을 틀린
  손님 짐이다 (`가구오인`)
* **이미 그 자리에 배정한 물건은 뺀다.**  이미 센 것을 '놓친 것'으로 다시
  세면, 멀쩡한 자리에도 딱지가 붙는다

설정(모델 경로·`imgsz`·임계값)은 **로그의 `run` 기록에서 읽는다.**  플래그로
따로 주면 진단이 실제 실행과 조용히 어긋난다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from engine.frame_dump import CLEAN_DIR, frame_stem


PROJECT_DIR = Path(__file__).resolve().parents[1]

# 재추론에 쓸 바닥 신뢰도.  "모델이 정말 아무것도 못 봤나"를 물으려면
# 우리 임계값보다 훨씬 아래에서 봐야 한다.
FLOOR_CONFIDENCE = 0.01

LABEL_LOGIC = "로직"
LABEL_MISLABEL = "가구오인"
LABEL_BORDERLINE = "임계값경계"
LABEL_BOX = "상자밖"
LABEL_FINETUNE = "파인튜닝"
LABEL_UNSCORED = "판단보류"

# 임계값경계로 부를 수 있는 바닥.  임계값의 1/3 아래에서 겨우 보인 것을
# "임계값만 내리면 잡힌다"고 부르면 거짓말이 된다 — 그 값까지 내리면 모든
# 자리에 유령 그릇이 쏟아진다.  그 아래는 모델이 못 보는 것으로 센다.
BORDERLINE_FLOOR_RATIO = 1.0 / 3.0

# 자리 상자 밖을 얼마나 넓혀 볼 것인가.  물건이 상자 **바로 밖**에 있으면
# 그건 탐지 문제가 아니라 상자를 잘못 그린 것이고, 고치는 사람이 다르다.
#
# 0.6 은 너무 넓었다: T6 의자5(89x264)를 그만큼 넓히면 옆 테이블 T1 위의
# 가방까지 들어와서, 모델이 아예 못 본 배낭을 "상자를 다시 그리면 된다"로
# 잘못 진단했다.  "바로 밖"은 말 그대로 바로 밖이어야 한다.
NEARBY_MARGIN = 0.25

# 같은 물건으로 볼 겹침.  이미 이 자리에 배정된 물건은 "놓친 것"이 아니다.
SAME_OBJECT_IOU = 0.5

# 손님 짐을 가구로 잘못 부른 것으로 의심할 조건.  자리 넓이의 이만큼도
# 안 되는 "의자"가 상판 위에 서 있고, 사람이 그린 어느 의자와도 안 맞으면
# 그건 의자가 아니라 그 위에 놓인 물건이다.
#
# 이걸 안 보면 진단이 "chair, dining table 뿐 — 모델이 못 본다"로 끝나서,
# 정작 **모델은 매번 정확히 봤고 이름만 틀렸다**는 사실이 묻힌다.
# (angle1 30초 T5: 세워둔 노트북을 chair 0.196 으로 불렀다.)
MISLABEL_MAX_AREA_FRACTION = 0.30
MISLABEL_MIN_SHARE = 0.50
MISLABEL_CHAIR_IOU = 0.30

# 근거 글자가 가리키는 곳.  무엇을 놓쳤느냐에 따라 볼 영역이 다르다.
EVIDENCE_REGION = {
    "t": "seat",     # 책상에 짐 → 좌석 상자 안
    "c": "chairs",   # 의자에 짐 → 연결된 의자 상자 안
    "s": "both",     # 사람이 앉음 → 둘 다
}

# 물건 후보로 안 세는 클래스.  모델이 봤어도 규칙이 일부러 버린다.
def _excluded_classes() -> frozenset:
    from engine.seatnow_core import EXCLUDED_OBJECT_CLASSES

    return frozenset(EXCLUDED_OBJECT_CLASSES)


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def share_inside(obj_box: Sequence[float], region: Sequence[float]) -> float:
    """물건의 몇 %가 이 영역 안에 있나 (engine 의 `object_seat_share` 와 같다)."""
    area = box_area(obj_box)
    if area <= 0:
        return 0.0
    return intersection_area(obj_box, region) / area


@dataclass
class Sighting:
    """저신뢰 재추론에서 이 자리 위에 보인 것 하나."""

    name: str
    confidence: float
    share: float
    where: str
    box: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "class": self.name,
            "confidence": round(self.confidence, 4),
            "share": round(self.share, 3),
            "where": self.where,
            "box": [round(value, 1) for value in self.box],
        }


@dataclass
class Diagnosis:
    timestamp: float
    seat: str
    category: str
    direction: Optional[str]
    looking_for: str
    label: str
    why: str
    best_confidence: Optional[float]
    sightings: List[Sighting] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "seat": self.seat,
            "category": self.category,
            "direction": self.direction,
            "looking_for": self.looking_for,
            "label": self.label,
            "why": self.why,
            "best_confidence": self.best_confidence,
            "sightings": [sighting.as_dict() for sighting in self.sightings],
        }


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    union = box_area(a) + box_area(b) - intersection_area(a, b)
    return intersection_area(a, b) / union if union > 0 else 0.0


def expand_region(box: Sequence[float], ratio: float = NEARBY_MARGIN) -> List[float]:
    """자리 상자를 사방으로 넓힌다.  '상자 바로 밖'을 보기 위한 것이다."""
    width = (box[2] - box[0]) * ratio
    height = (box[3] - box[1]) * ratio
    return [box[0] - width, box[1] - height, box[2] + width, box[3] + height]


def already_counted(
    detection: Mapping[str, object], miss: Mapping[str, object]
) -> bool:
    """우리가 이 자리에 **이미 배정한** 물건인가.

    이미 센 것을 '놓친 것'으로 다시 세면, 놓친 근거가 없는 자리에도
    "규칙이 버렸다" 딱지가 붙는다.
    """
    box = [float(value) for value in (detection.get("box") or [])]
    if not box:
        return False
    name = str(detection.get("class", ""))
    for counted in list(miss.get("objects") or []) + list(miss.get("chair_objects") or []):
        if str(counted.get("class", "")) != name:
            continue
        counted_box = [float(value) for value in (counted.get("box") or [])]
        if not counted_box:
            # 옛 로그의 의자 짐에는 상자가 없다.  클래스가 같으면 같은 것으로 본다.
            return True
        if box_iou(box, counted_box) >= SAME_OBJECT_IOU:
            return True
    return False


def regions_for(miss: Mapping[str, object], wanted: str) -> List[Tuple[str, List[float]]]:
    """이 오답을 진단하려면 사진의 어디를 봐야 하나."""
    seat_box = [float(value) for value in (miss.get("seat_box") or [])]
    chairs = [
        [float(value) for value in (chair.get("box") or [])]
        for chair in (miss.get("connected_chairs") or [])
        if chair.get("box")
    ]
    where = EVIDENCE_REGION.get(wanted, "both")
    regions: List[Tuple[str, List[float]]] = []
    if where in ("seat", "both") and seat_box:
        regions.append(("자리", seat_box))
    if where in ("chairs", "both"):
        regions += [(f"의자{index + 1}", box) for index, box in enumerate(chairs)]
    return regions


def collect_sightings(
    detections: Sequence[Mapping[str, object]],
    regions: Sequence[Tuple[str, List[float]]],
    minimum_share: float = 0.10,
) -> List[Sighting]:
    """이 영역들 위에 걸친 탐지를 전부 모은다. 겹침이 큰 것부터."""
    excluded = _excluded_classes()
    sightings: List[Sighting] = []
    for detection in detections:
        name = str(detection.get("class", ""))
        box = [float(value) for value in (detection.get("box") or [])]
        if not box:
            continue
        best: Optional[Tuple[float, str]] = None
        for where, region in regions:
            if not region:
                continue
            share = share_inside(box, region)
            if best is None or share > best[0]:
                best = (share, where)
        if best is None or best[0] < minimum_share:
            continue
        # 가구·고정 집기는 규칙이 일부러 버리는 클래스라 따로 표시한다.
        where = best[1] + ("(제외클래스)" if name in excluded else "")
        sightings.append(
            Sighting(
                name=name,
                confidence=float(detection.get("confidence", 0.0)),
                share=best[0],
                where=where,
                box=box,
            )
        )
    sightings.sort(key=lambda item: (-item.confidence, -item.share))
    return sightings


def relevant_sightings(sightings: Sequence[Sighting], wanted: str) -> List[Sighting]:
    """이 숙제에 답이 될 수 있는 탐지만 남긴다.

    '사람이 앉음'을 찾을 때는 person 만 본다 — 가방을 사람으로 셀 수 없다.
    나머지는 규칙이 물건 후보로 쓰는 클래스만 본다.  의자·테이블은 어느
    자리에나 늘 보이는 **배경**이지 손님 짐이 아니므로 답이 될 수 없다.
    """
    excluded = _excluded_classes()
    if wanted == "s":
        return [item for item in sightings if item.name == "person"]
    return [item for item in sightings if item.name not in excluded]


def furniture_mistaken_for_belongings(
    sightings: Sequence[Sighting],
    seat_box: Sequence[float],
    drawn_chairs: Sequence[Sequence[float]],
    threshold: float = 0.0,
) -> List[Sighting]:
    """가구 이름을 달고 있지만 실은 손님 짐인 것들.

    탐지기가 세워둔 노트북을 ``chair`` 라고 부르면, 그 클래스가 제외
    목록에 있어서 통째로 버려진다.  진단이 이걸 "배경"으로 넘기면
    "모델이 못 본다"는 잘못된 결론이 나온다 — 실제로는 **매번 정확한
    자리에서 봤고 이름만 틀렸다.**  고치는 방법이 완전히 다르다.
    """
    seat_area = box_area(seat_box)
    if seat_area <= 0:
        return []
    # 상판 위에 "서 있는" 물건은 원근 때문에 그린 상자 위로 솟는다.
    region = expand_region(seat_box)
    suspects: List[Sighting] = []
    for item in sightings:
        if item.name not in _excluded_classes() or item.name == "person":
            continue
        if item.confidence < threshold:
            # 임계값 아래의 가구는 탐지기의 잡음이다.  그걸 "봤는데 이름을
            # 틀렸다"고 부르면, 정말 못 보는 것까지 이름 문제로 둔갑한다.
            continue
        if box_area(item.box) > MISLABEL_MAX_AREA_FRACTION * seat_area:
            continue  # 진짜 가구는 자리만큼 크다.
        if share_inside(item.box, region) < MISLABEL_MIN_SHARE:
            continue  # 상판 위(또는 바로 그 위)에 있어야 한다.
        if any(box_iou(item.box, chair) >= MISLABEL_CHAIR_IOU for chair in drawn_chairs):
            continue  # 사람이 그린 의자와 맞으면 진짜 의자다.
        suspects.append(item)
    return sorted(suspects, key=lambda item: -item.confidence)


def label_miss(
    sightings: Sequence[Sighting],
    threshold: float,
    wanted: str,
    nearby: Sequence[Sighting] = (),
    mislabelled: Sequence[Sighting] = (),
) -> Tuple[str, str, Optional[float]]:
    """딱지 하나를 붙인다 — 고치는 사람이 서로 다르다."""
    relevant = relevant_sightings(sightings, wanted)
    best = max(relevant, key=lambda item: item.confidence) if relevant else None

    if best is not None and best.confidence >= threshold:
        return (
            LABEL_LOGIC,
            f"임계값({threshold:.2f}) 위에서 이미 {best.name} {best.confidence:.2f} "
            f"(겹침 {best.share * 100:.0f}%)를 잡고 있었다 — 규칙이 버렸다",
            best.confidence,
        )

    # 상자 안에서 못 찾았으면, 물건이 상자 **바로 밖**에 있는지 본다.
    # 그건 탐지 문제가 아니라 자리 상자를 잘못 그린 것이다.
    outside = [
        item
        for item in relevant_sightings(nearby, wanted)
        if item.confidence >= threshold
    ]
    if outside:
        near = max(outside, key=lambda item: item.confidence)
        return (
            LABEL_BOX,
            f"{near.name} {near.confidence:.2f} 를 자리 **바로 밖**에서 잡고 있다 "
            f"({near.where}) — 상자를 다시 그릴 일이다",
            near.confidence,
        )

    if best is not None and best.confidence >= threshold * BORDERLINE_FLOOR_RATIO:
        return (
            LABEL_BORDERLINE,
            f"{best.name} 를 {best.confidence:.2f} 로만 봤다 "
            f"(임계값 {threshold:.2f}, 겹침 {best.share * 100:.0f}%) "
            "— 전역 기본값을 내려볼 수 있는 범위다",
            best.confidence,
        )

    # 이름만 틀린 경우가 "안 보인다"보다 먼저다.  둘은 학습 데이터
    # 요구사항이 완전히 다르다.
    if wanted != "s" and mislabelled:
        worst = mislabelled[0]
        return (
            LABEL_MISLABEL,
            f"자리 위의 물건을 **{worst.name} {worst.confidence:.2f}** 로 불렀다 "
            f"(겹침 {worst.share * 100:.0f}%) — 못 본 게 아니라 **이름을 틀렸다.** "
            "그 클래스는 가구라 통째로 버려진다",
            worst.confidence,
        )

    if best is not None:
        return (
            LABEL_FINETUNE,
            f"{best.name} 가 {best.confidence:.3f} 로 스쳤을 뿐이다 "
            f"(임계값 {threshold:.2f}의 1/3에도 못 미친다) — 여기까지 내리면 "
            "모든 자리에 유령이 쏟아진다",
            best.confidence,
        )

    background = ", ".join(sorted({item.name for item in sightings})) or "아무것도"
    return (
        LABEL_FINETUNE,
        f"신뢰도 {FLOOR_CONFIDENCE}에서도 "
        + ("사람이" if wanted == "s" else "짐이")
        + f" 안 보인다 (보이는 것은 {background} 뿐)",
        None,
    )


def wanted_evidence(miss: Mapping[str, object]) -> str:
    """이 칸에서 무엇을 찾아야 하나.  놓친 근거 > 상태 방향 순으로 본다."""
    missed = str(miss.get("missed_evidence") or "")
    if missed:
        # 여러 개를 놓쳤으면 사람 > 책상 > 의자 순으로 가장 확실한 것부터.
        for letter in "stc":
            if letter in missed:
                return letter
    if miss.get("truth") == "occupied":
        return "t"
    return ""


def run_detector(
    frame_path: Path, model_path: Path, imgsz: int, confidence: float
) -> List[Dict[str, object]]:
    """사진 한 장을 아주 낮은 신뢰도로 다시 훑는다."""
    import cv2
    import numpy as np

    from engine.seatnow_core import load_model

    # 한글이 든 경로에서 cv2.imread 가 조용히 실패한다 (plan.md §7).
    data = np.frombuffer(frame_path.read_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"사진을 못 읽었다: {frame_path}")

    model = load_model(model_path, "detect")
    result = model.predict(frame, imgsz=imgsz, conf=confidence, verbose=False)[0]
    names = result.names
    items: List[Dict[str, object]] = []
    for box in result.boxes:
        items.append(
            {
                "class": names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    return items


def claimed_by_another_seat(
    box: Sequence[float],
    seat: str,
    seat_boxes: Sequence[Tuple[str, Sequence[float]]],
) -> bool:
    """이 물건을 더 많이 덮는 다른 자리가 있나.

    "상자밖"은 *이 자리* 것인데 상자를 벗어났다는 뜻이다.  옆자리 위에
    멀쩡히 놓인 물건을 끌어오면, 모델이 아예 못 본 짐까지 "상자를 다시
    그리면 된다"로 둔갑한다 (angle1 30초 T6 에서 실제로 그랬다).
    """
    mine = max(
        (share_inside(box, other) for name, other in seat_boxes if name == seat),
        default=0.0,
    )
    theirs = max(
        (share_inside(box, other) for name, other in seat_boxes if name != seat),
        default=0.0,
    )
    return theirs > mine


def diagnose(
    misses: Sequence[Mapping[str, object]],
    detections_by_time: Mapping[float, Sequence[Mapping[str, object]]],
    threshold: float,
    seat_boxes_by_time: Optional[Mapping[float, Sequence[Tuple[str, Sequence[float]]]]] = None,
) -> List[Diagnosis]:
    """오답 목록에 딱지를 붙인다.  모델 호출은 이미 끝난 상태로 들어온다."""
    results: List[Diagnosis] = []
    for miss in misses:
        timestamp = float(miss.get("timestamp", 0.0))
        wanted = wanted_evidence(miss)
        imagined = str(miss.get("imagined_evidence") or "")

        if not wanted and imagined:
            # 헛것은 "모델이 못 봤나"를 물을 일이 아니다.  우리가 무엇을
            # 보고 그렇게 판단했는지가 이미 로그에 있고, 그건 규칙 문제다.
            seen = [
                f"{obj.get('class')} {float(obj.get('confidence', 0)):.2f}"
                for obj in list(miss.get("objects") or [])
                + list(miss.get("chair_objects") or [])
            ]
            results.append(
                Diagnosis(
                    timestamp=timestamp,
                    seat=str(miss.get("seat", "?")),
                    category=str(miss.get("category", "")),
                    direction=miss.get("direction"),  # type: ignore[arg-type]
                    looking_for=f"헛본 근거 {imagined}",
                    label=LABEL_LOGIC,
                    why="없는 것을 봤다 — 우리가 쓴 것: " + (", ".join(seen) or "기록 없음"),
                    best_confidence=None,
                )
            )
            continue

        if not wanted:
            results.append(
                Diagnosis(
                    timestamp=timestamp,
                    seat=str(miss.get("seat", "?")),
                    category=str(miss.get("category", "")),
                    direction=miss.get("direction"),  # type: ignore[arg-type]
                    looking_for="—",
                    label=LABEL_UNSCORED,
                    why="무엇을 찾아야 할지 정답지가 말해주지 않는다",
                    best_confidence=None,
                )
            )
            continue

        regions = regions_for(miss, wanted)
        # 이미 이 자리에 배정한 물건은 놓친 것이 아니다.
        fresh = [
            detection
            for detection in detections_by_time.get(timestamp, [])
            if not already_counted(detection, miss)
        ]
        sightings = collect_sightings(fresh, regions)
        inside = {id(item) for item in sightings}
        wider = [(f"{name} 주변", expand_region(box)) for name, box in regions]
        seat_boxes = (seat_boxes_by_time or {}).get(timestamp, [])
        seat_name = str(miss.get("seat", "?"))
        around = [
            item
            for item in collect_sightings(fresh, wider)
            if not any(
                other.name == item.name and box_iou(other.box, item.box) >= SAME_OBJECT_IOU
                for other in sightings
            )
        ]
        # "상자밖"은 **소유권** 질문이라 옆자리가 더 많이 덮으면 그 자리 것이다.
        # "가구오인"은 **이름** 질문이라 소유권과 무관하다 — 원근 때문에 옆
        # 테이블 상자가 이 자리 상판 위를 덮는 일이 흔하고(angle1 T4/T5),
        # 거기서 이름을 틀린 물건까지 놓치면 안 된다.
        nearby = [
            item
            for item in around
            if not claimed_by_another_seat(item.box, seat_name, seat_boxes)
        ]
        seat_box = [float(value) for value in (miss.get("seat_box") or [])]
        drawn_chairs = [
            [float(value) for value in (chair.get("box") or [])]
            for chair in (miss.get("connected_chairs") or [])
            if chair.get("box")
        ]
        # 상자 바로 밖도 같이 본다.  angle1 30초 T5 의 노트북은 상판 안쪽
        # 끝에 서 있어서 그린 상자를 벗어나는데, 이름까지 틀린 경우다.
        mislabelled = (
            furniture_mistaken_for_belongings(
                list(sightings) + list(around), seat_box, drawn_chairs, threshold
            )
            if seat_box
            else []
        )
        label, why, best = label_miss(
            sightings, threshold, wanted, nearby, mislabelled
        )
        results.append(
            Diagnosis(
                timestamp=timestamp,
                seat=str(miss.get("seat", "?")),
                category=str(miss.get("category", "")),
                direction=miss.get("direction"),  # type: ignore[arg-type]
                looking_for=f"{wanted} ({ {'s': '사람이 앉음', 't': '책상에 짐', 'c': '의자에 짐'}[wanted] })",
                label=label,
                why=why,
                best_confidence=best,
                sightings=sightings[:6],
            )
        )
    return results


def render_report(diagnoses: Sequence[Diagnosis], threshold: float) -> str:
    counts: Dict[str, int] = {}
    for item in diagnoses:
        counts[item.label] = counts.get(item.label, 0) + 1

    lines = ["# 원인 진단 — 모델이 못 본 건가, 코드가 버린 건가", ""]
    lines.append("이미 그 자리에 배정한 물건은 뺐다. 남은 것만 '놓친 것'이다.")
    lines.append("")
    lines.append(
        f"신뢰도 **{FLOOR_CONFIDENCE}** 로 다시 훑어봤다 "
        f"(실제 실행 임계값은 {threshold:.2f})."
    )
    lines.append("")
    lines += ["| 딱지 | 칸 | 누가 고치나 |", "|---|---:|---|"]
    for label, who in (
        (LABEL_LOGIC, "**코드** — 규칙이 이미 있는 것을 버렸다"),
        (LABEL_BOX, "**설치 검수** — 자리 상자를 다시 그린다"),
        (LABEL_BORDERLINE, "**코드** — 전역 임계값을 내려보고 전체 점수로 판단"),
        (LABEL_MISLABEL, "**모델** — 봤는데 **이름을 틀렸다.** 가구로 불러서 버려진다"),
        (LABEL_FINETUNE, "**모델** — 아예 못 본다"),
        (LABEL_UNSCORED, "판단 보류"),
    ):
        if counts.get(label):
            lines.append(f"| {label} | {counts[label]} | {who} |")
    lines.append("")

    for label in (LABEL_LOGIC, LABEL_BOX, LABEL_BORDERLINE, LABEL_MISLABEL, LABEL_FINETUNE, LABEL_UNSCORED):
        rows = [item for item in diagnoses if item.label == label]
        if not rows:
            continue
        lines += [f"## {label}", "", "| 시각 | 자리 | 등급 | 찾던 것 | 왜 그렇게 봤나 |", "|---|---|---|---|---|"]
        for item in sorted(rows, key=lambda r: (r.timestamp, r.seat)):
            lines.append(
                f"| {item.timestamp:g}초 | {item.seat} | {item.category} "
                f"| {item.looking_for} | {item.why} |"
            )
        lines.append("")
        detailed = [item for item in rows if item.sightings]
        if detailed:
            lines += ["<details><summary>그 자리 위에 실제로 보인 것들</summary>", ""]
            for item in sorted(detailed, key=lambda r: (r.timestamp, r.seat)):
                lines.append(f"**{item.timestamp:g}초 {item.seat}**")
                lines.append("")
                lines.append("| 클래스 | 신뢰도 | 겹침 | 어디 |")
                lines.append("|---|---:|---:|---|")
                for sighting in item.sightings:
                    lines.append(
                        f"| {sighting.name} | {sighting.confidence:.3f} "
                        f"| {sighting.share * 100:.0f}% | {sighting.where} |"
                    )
                lines.append("")
            lines += ["</details>", ""]

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="오답의 원인을 모델에게 되물어본다")
    parser.add_argument("run_dir", type=Path, help="results/<이름> 폴더")
    parser.add_argument("--misses", type=Path, help="기본: <run_dir>/오답.json")
    parser.add_argument("--log", type=Path, help="기본: <run_dir>/log.jsonl")
    parser.add_argument("--output", type=Path, help="기본: <run_dir>/원인진단.md")
    parser.add_argument(
        "--include-evidence-gaps",
        action="store_true",
        help="상태는 맞았지만 근거를 놓친 칸도 진단한다 (탐지 숙제를 다 보려면)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    misses_path = args.misses or run_dir / "오답.json"
    log_path = args.log or run_dir / "log.jsonl"

    if not misses_path.exists():
        print(f"오답 목록이 없다: {misses_path}  (먼저 checks.score_answers 를 돌려라)")
        return 2

    payload = json.loads(misses_path.read_text(encoding="utf-8"))
    misses = list(payload.get("misses") or [])
    if args.include_evidence_gaps:
        misses += list(payload.get("evidence_gaps") or [])
    if not misses:
        print("진단할 오답이 없다")
        return 0

    from checks.verify_seatnow import load_jsonl

    records = load_jsonl(log_path)
    config = (records[0].get("run") or {}).get("config") or {}
    models = (records[0].get("run") or {}).get("models") or {}
    imgsz = int(config.get("imgsz", 1280))
    threshold = float(config.get("object_confidence", 0.15))
    detector = Path(str(models.get("detector") or PROJECT_DIR / "yolov8n.pt"))
    if not detector.exists():
        detector = PROJECT_DIR / Path(detector).name

    timestamps = sorted({float(miss.get("timestamp", 0.0)) for miss in misses})
    detections_by_time: Dict[float, List[Dict[str, object]]] = {}
    for timestamp in timestamps:
        frame = run_dir / CLEAN_DIR / f"{frame_stem(timestamp)}.jpg"
        if not frame.exists():
            print(f"  사진이 없다, 건너뜀: {frame}")
            detections_by_time[timestamp] = []
            continue
        print(f"  {timestamp:g}s 재추론 (conf={FLOOR_CONFIDENCE}, imgsz={imgsz})...")
        detections_by_time[timestamp] = run_detector(
            frame, detector, imgsz, FLOOR_CONFIDENCE
        )

    seat_boxes_by_time: Dict[float, List[Tuple[str, List[float]]]] = {}
    for record in records:
        stamp = float(record.get("timestamp", 0.0))
        seat_boxes_by_time[stamp] = [
            (
                str(table.get("layout_name") or table.get("label") or "?"),
                [float(value) for value in (table.get("box") or [])],
            )
            for table in (record.get("tables") or [])
            if table.get("box")
        ]

    diagnoses = diagnose(misses, detections_by_time, threshold, seat_boxes_by_time)
    output = args.output or run_dir / "원인진단.md"
    output.write_text(render_report(diagnoses, threshold), encoding="utf-8")
    (run_dir / "원인진단.json").write_text(
        json.dumps(
            {"threshold": threshold, "floor": FLOOR_CONFIDENCE,
             "diagnoses": [item.as_dict() for item in diagnoses]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    counts: Dict[str, int] = {}
    for item in diagnoses:
        counts[item.label] = counts.get(item.label, 0) + 1
    print(f"진단: {output}")
    print("  " + " | ".join(f"{label}={count}" for label, count in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
