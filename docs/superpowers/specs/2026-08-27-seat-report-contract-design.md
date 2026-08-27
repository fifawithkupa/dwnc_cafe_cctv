# 좌석 리포트 출력 계약 재정의 (seat_report)

> 작성: 2026-08-27
> 기준 커밋: `JINOI/main` = `6d0d078`
> 관련: `plan.md` T3, `CLAUDE.md`, `docs/superpowers/specs/2026-07-11-manual-seat-layout-design.md`

## 1. 배경과 목표

파일럿 카페에 카메라를 설치하고 15초 tick마다 좌석 가용성을 낸다. 목표는
**설치 시 1회 육안 검수만으로 운영이 시작되고, 그 뒤로 사람이 개입하지 않는 것**이다.

전제는 `CLAUDE.md`에 규칙으로 고정했다.

- 카메라의 기종·위치·각도는 SeatNow 팀이 정하고 직접 설치한다
- 매장당 카메라 **1대**
- 매장에 요구할 수 있는 수작업은 설치 당일 `calibrate.py` 검수 1회가 전부다
  (`_preseed`가 자동으로 채운 것에서 **잘못 잡힌 것 지우기 + 못 잡은 것 추가하기**)

이 문서가 푸는 문제는 세 가지다.

1. **일자형/벽 책상이 출력에 존재하지 않는다.** 모델이 `dining table`로 잡지 못하고,
   사람이 그려 넣어도 판정 단위가 "테이블 1개 = 상태 1개"라 "6석 중 3석"을 표현할 수 없다.
2. **앱이 쓸 계약이 없다.** JSONL 최상위는 트랙 단위 진단 정보(`tables[]`)이고,
   `summary.unknown`은 트랙 개수라서 "몇 자리 남음"으로 변환되지 않는다.
3. **UNKNOWN을 줄이려 해도 어디를 잡을지 모른다.** 원인이 되는 포즈 사유가
   테이블로 올라오면서 `nearby_person_pose_unknown` 한 줄로 뭉개진다
   (`seatnow_core.py:1611-1615`).

## 2. 판정 결과의 원칙

애매한 것을 `empty`로 반올림하지 않는다. `occupied` / `empty` / `unknown`을
그대로 내보내고, **UNKNOWN 비율을 줄이는 것을 개발의 방향**으로 삼는다.

`IGNORE`와 `UNKNOWN`을 섞지 않는다. 의미가 다르고 푸는 방법이 다르다.

| | 뜻 | 푸는 방법 |
|---|---|---|
| `IGNORE` | 화각 밖 / 애초에 판정 대상 아님 | 카메라 재설치 (설치 품질 지표) |
| `UNKNOWN` | 보이는데 판정 실패 | 코드 (엔지니어링 지표) |

카메라를 우리가 설치하므로 IGNORE는 설치 검수에서 걸러야 할 항목이다.
따라서 `totals.capacity`는 **설치 검수 때 확정된 좌석 수로 고정**하고,
운영 중 IGNORE가 튀면 앱 표시를 바꾸는 게 아니라 **설치가 틀어졌다는 알람**으로 다룬다.

## 3. 출력 계약 `seat_report`

tick마다 JSONL 최상위에 `seat_report`를 추가한다. 기존 `tables[]`는
진단용으로 그대로 둔다 (하위 호환, `verify_seatnow.py` 무영향).

```jsonc
"seat_report": {
  "schema_version": 1,
  "tick_at": 1234.5,
  "seats": [
    { "seat_id": "T3", "kind": "table", "capacity": 1,
      "state": "occupied", "reason_code": "person_seated", "confidence": 0.86 },

    { "seat_id": "BAR", "kind": "counted_zone", "capacity": 6,
      "occupied": 3, "free": 2, "unknown": 1,
      "reason_codes": { "occluded_lower_body": 1 } }
    // ... 나머지 좌석 생략 (capacity 3, occupied 1, free 1, unknown 1)
  ],
  "totals": { "capacity": 10, "occupied": 5, "free": 3, "unknown": 2 }
}
```

필드는 `kind`에 따라 갈린다. `table`은 `state` / `reason_code` / `confidence`를,
`counted_zone`은 `occupied` / `free` / `unknown` / `reason_codes`(사유별 개수)를 갖는다.
`reason_code`는 UNKNOWN 전용이 아니라 **모든 상태에 붙는다** — `occupied`와 `empty`도
근거를 남긴다 (§5.2).

**`totals.free`는 확신하는 empty만 센다.** UNKNOWN을 free로 반올림하지 않는다.
앱은 "3자리 남음 (2자리 확인불가)"로 표시할 수 있다.

`tables[]`를 직접 개조하지 않는 이유: `track_to_dict`는 `layout_version`,
`predicted`, `raw_state` 등 디버깅 필드가 많아 앱 계약과 진단이 뒤엉킨다.
분리하면 앱은 `seat_report`만, 개발은 `tables[]`를 본다.

## 4. 구역 카운트 `counted_zone`

### 4.1 레이아웃 스키마 (v2)

```jsonc
{ "id": 7, "name": "BAR", "kind": "counted_zone", "capacity": 6,
  "box": [x1, y1, x2, y2], "chairs": [] }
```

`kind` 생략 시 `"table"`로 간주한다. **기존 v1 레이아웃 파일은 무수정으로 로드된다.**

### 4.2 카운트 규칙

구역을 좌석 경계로 쪼개지 **않는다.** 원근 때문에 일자 책상은 화면상 좌석 간격이
불균등해서, 박스를 capacity등분하면 어긋난다. 대신 증거 클러스터를 세고
capacity로 상한을 건다.

| 증거 | 판정 |
|---|---|
| 착석 포즈 사람의 anchor가 구역 안 | `occupied` +1 |
| 비제외 클래스 객체가 구역 안 + 근처에 사람 없음 | `occupied` +1 (짐 점유) |
| 사람은 있는데 포즈 판정 불가 | `unknown` +1 |
| 사람과 짐이 서로 가까움 | 같은 자리로 병합 (중복 카운트 방지) |
| **서 있는 사람**(`PoseState.STANDING`) | **세지 않는다** — 지나가는 손님·주문 대기 |

`free = max(0, capacity - occupied - unknown)`

"근처에 사람 없음", "서로 가까움"의 판정은 새 임계값을 만들지 않고
`associate_objects_to_chairs` / `associate_seated_people_to_chairs`가 쓰는
기존 연결 규칙을 구역 안에서 재사용한다. 조정 손잡이를 늘리지 않는 것이
`CLAUDE.md`의 "매장별 튜닝 금지" 규칙과 맞다.

이 방식은 **탐지 모델이 제일 잘하는 일(사람 박스 세기)에 기댄다.** 좌석 경계
추정은 모델이 못 하는 일이고, 사람 수 세기는 잘한다. 짐 증거는 기존
`EXCLUDED_OBJECT_CLASSES` 필터를 구역 안에서 그대로 재사용한다.

**수용하는 한계**: 두 명이 딱 붙어 앉으면 1명으로 셀 수 있다. 카메라 1대 + 가림
전제에서 피할 수 없다. `occupied`를 과소 계상하는 방향이라 "빈자리를 부풀리는"
쪽으로 틀린다. 사유 코드로 관측만 해두고, 데이터로 확인된 뒤에 대응한다.

### 4.3 카운트 디바운싱

기존 확정 임계값을 그대로 적용한다. 카운트 **증가는 2회**, **감소는 3회** 연속
관측 후 반영 — 상태 디바운싱과 같은 비대칭(`occupy_confirmations=2`,
`empty_confirmations=3`)이라 규칙이 하나로 유지된다.

## 5. UNKNOWN 사유 코드

### 5.1 핵심 변경

`nearby_person_pose_unknown`으로 덮어쓰지 말고, **원인이 된 포즈의 `reason`을
위로 승격**시킨다. 새 정보를 만드는 게 아니라 이미 있는 정보를 버리지 않는 것이다.

기존 어휘: `compact_occluded_pose`(`seatnow_core.py:484`),
`insufficient_keypoints`(`:487`), `nearby_person_pose_unknown`(`:1614`),
`no_customer_evidence`(`:1619`), `border_cropped`(`:1559`),
`temporarily_occluded:` 접두사(`:3256`).

### 5.2 분류 축은 "무엇으로 푸는가"

| 그룹 | 코드 | 푸는 방법 |
|---|---|---|
| 설치 (→IGNORE) | `border_cropped`, `scene_change` | 카메라 재배치. 코드로 풀지 않는다 |
| 기하·가림 (→UNKNOWN) | `occluded_lower_body`, `ambiguous_association` | 구제 경로 (의자 연결·짐 증거) |
| 모델 (→UNKNOWN) | `pose_low_keypoints`, `table_not_detected` | 파인튜닝(T11) 또는 imgsz 상향 |
| 시간 (→UNKNOWN) | `track_predicted`, `pending_confirmation` | 아무것도 안 한다. 다음 tick에 해소 |
| 확정 (→OCCUPIED) | `person_seated`, `belongings`, `occupied_chair` | 해당 없음 (정상 판정) |
| 확정 (→EMPTY) | `no_customer_evidence` | 해당 없음 (정상 판정) |
| 이상 | `count_exceeds_capacity` | 레이아웃 `capacity`가 틀렸다는 신호 → 검수 |

마지막 그룹의 분리가 중요하다. `pending_confirmation`(확정 대기)은 **정상 동작인데
지금은 UNKNOWN에 섞인다.** 분리하지 않으면 "UNKNOWN 30%"의 상당 부분이 사실은
고칠 게 없는 대기 상태인데 개선 대상으로 오해된다.

`reason_code`는 `Enum`으로 닫는다. 기존 자유 문자열 `reason`
(`seated:2;objects:cup` 같은 상세)은 진단용으로 `tables[]`에 그대로 둔다.

### 5.3 집계

`verify_seatnow.py`의 기존 커버리지 집계에 합류시켜
**"UNKNOWN 34% = 가림 22% + 모델 8% + 대기 4%"** 형태로 뽑는다.
다음에 뭘 할지가 논쟁이 아니라 표에서 나온다.

## 6. 모듈 경계와 데이터 흐름

`seatnow_core.py`가 이미 3,373줄이다 (다른 모듈은 전부 300~600줄대). 여기에
더 넣지 않고 **신규 모듈 `seatnow_report.py`**로 분리한다.

```
frame
  └─ analyze()                  seatnow_core.py   증거 연결 + counted_zone 카운트
       └─ TableObservation      (+ occupied_count, unknown_count 필드)
  └─ OccupancyTracker.update()  seatnow_core.py   디바운싱 (카운트 포함)
       └─ TrackerUpdate
  └─ build_seat_report()        seatnow_report.py ← 신규, 순수 함수
       └─ seat_report dict
  └─ frame_log_record()         seatnow_core.py   JSONL 기록
```

카운트를 `analyze()`에 두는 이유: 증거 연결(사람·짐→좌석)이 이미 거기 있다.
리포트 모듈로 빼면 원본 detection을 통째로 넘겨야 해서 경계가 지저분해진다.
반대로 `build_seat_report()`는 **dict in / dict out 순수 함수**라 모델 없이
전부 테스트된다 — 기존 171개 테스트의 "모델 없이 순수 로직 검증" 패턴 그대로다.

## 7. 에러 처리

레이아웃은 **로드 시점에 강하게 실패**시킨다. 런타임에 조용히 틀리는 것보다 낫다.

| 상황 | 처리 |
|---|---|
| `schema_version: 1` 레이아웃 | 그대로 로드, `kind="table"` 기본값 |
| `counted_zone`인데 `capacity` 없음/0 이하 | `LayoutError` |
| 모르는 `kind` 값 | `LayoutError` |
| 카운트 > capacity | capacity로 클램프 + `count_exceeds_capacity` 기록 |

`build_seat_report()`는 **예외를 삼키지 않는다.** 검증된 데이터에 대한 순수
포매팅이라 여기서 터지면 버그지 운영 상황이 아니다. 24/7 복원력은 T10의
systemd 재시작이 담당하는 층이다.

## 8. 테스트

신규 `tests/test_seatnow_report.py`:

- v1 레이아웃 하위 호환 — 기존 `layouts/*.json` 2개가 무수정 로드
- 카운트 규칙 — 사람만 / 짐만 / 사람+짐 병합 / capacity 초과 / unknown 혼재
- 사유 코드 — 4개 그룹의 각 분기가 정확한 코드를 내는지
- `totals.free`가 UNKNOWN을 절대 포함하지 않는지 (계약의 핵심)

## 9. 연쇄 변경

라벨링 도구도 같이 가야 파일럿 매장 채점이 된다.

- `make_labels.py` — `counted_zone`은 occupied/empty/ignore가 아니라
  **인원 수**를 받는 스켈레톤 생성
- `verify_seatnow.py` — 카운트 정답 채점 + 사유 코드별 UNKNOWN 분포 집계

## 10. 범위 밖

- **판정 임계값의 각도 불변화** (`sitting_angle: 110.0` 등). 근본적이지만 지금은
  근거가 없다. 어느 임계값이 실제 병목인지는 라벨링된 평가셋에서 사유 코드
  분포가 나온 뒤에 판단한다. 설치 후 각도는 어차피 고정이다.
- **카메라 2대**. `CLAUDE.md` 규칙대로 설계에서 배제한다.
- **파인튜닝(T11)**. 사유 코드의 "모델" 그룹 비중이 확인된 뒤의 일이다.
